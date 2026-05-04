import numpy as np
import optuna
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
import json
import joblib
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- Param spaces: solo params tuneables (sin fixed como random_state, n_jobs) ---

def _xgboost_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
    }


def _lightgbm_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
    }


def _random_forest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _logistic_regression_params(trial):
    return {
        "C": trial.suggest_float("C", 0.001, 10, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
    }


MODEL_REGISTRY = {
    "xgboost": {"suggest_params": _xgboost_params},
    "lightgbm": {"suggest_params": _lightgbm_params},
    "random_forest": {"suggest_params": _random_forest_params},
    "logistic_regression": {"suggest_params": _logistic_regression_params},
}


def _build_model(model_type, tunable_params, scale_pos_weight):
    """Construye el modelo con params tuneables + params fijos. LR incluye StandardScaler."""
    if model_type == "xgboost":
        return XGBClassifier(
            **tunable_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
    elif model_type == "lightgbm":
        return LGBMClassifier(
            **tunable_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            **tunable_params,
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                **tunable_params,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=42,
            )),
        ])


def _log_model(model, model_type):
    if model_type == "xgboost":
        mlflow.xgboost.log_model(model, "model")
    elif model_type == "lightgbm":
        mlflow.lightgbm.log_model(model, "model")
    else:
        mlflow.sklearn.log_model(model, "model")


def prepare_X(df, target_col="Churn", feature_cols=None):
    X = df.drop(columns=[target_col])
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes
    if feature_cols is not None:
        available = [c for c in feature_cols if c in X.columns]
        X = X[available]
    return X


def split_data(df, target_col="Churn", feature_cols=None, test_size=0.2, random_state=42):
    X = prepare_X(df, target_col, feature_cols)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_objective_cv(X_trainval, y_trainval, scale_pos_weight, model_type, cv_folds):
    suggest_params = MODEL_REGISTRY[model_type]["suggest_params"]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial):
        tunable = suggest_params(trial)
        threshold = trial.suggest_float("threshold", 0.3, 0.7)
        model = _build_model(model_type, tunable, scale_pos_weight)

        fold_f1s = []
        for train_idx, val_idx in cv.split(X_trainval, y_trainval):
            X_tr, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
            y_tr, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]
            y_pred = (proba >= threshold).astype(int)
            fold_f1s.append(f1_score(y_val, y_pred))

        cv_f1_mean = float(np.mean(fold_f1s))
        cv_f1_std = float(np.std(fold_f1s))

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params({**tunable, "threshold": threshold})
            mlflow.log_metrics({"cv_f1_mean": cv_f1_mean, "cv_f1_std": cv_f1_std})

        return cv_f1_mean

    return objective


def evaluate_model(model, X_test, y_test, threshold):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, proba),
    }, y_pred, proba


def save_artifacts(model, threshold, metrics, best_params, run_id, model_type, feature_set, model_dir="models"):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    slug = f"{model_type}_{feature_set}"
    joblib.dump(model, model_dir / f"best_model_{slug}.pkl")
    with open(model_dir / f"metadata_{slug}.json", "w") as f:
        json.dump({
            "model_type": model_type,
            "feature_set": feature_set,
            "threshold": threshold,
            "metrics": metrics,
            "best_params": best_params,
            "run_id": run_id,
        }, f, indent=2)


def train_model(
    df,
    model_type="xgboost",
    experiment_name="churn_model_comparison",
    feature_set="full",
    feature_cols=None,
    n_trials=100,
    cv_folds=10,
    target_col="Churn",
    test_size=0.2,
    random_state=42,
    save_dir="models",
):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type '{model_type}' no existe. Opciones: {list(MODEL_REGISTRY)}")

    mlflow.set_experiment(experiment_name)

    X_trainval, X_test, y_trainval, y_test = split_data(
        df, target_col, feature_cols, test_size, random_state
    )
    scale_pos_weight = (y_trainval == 0).sum() / (y_trainval == 1).sum()

    with mlflow.start_run(run_name=f"{model_type}_{feature_set}") as parent_run:
        mlflow.set_tags({"model_type": model_type, "feature_set": feature_set})
        mlflow.log_params({
            "n_features": X_trainval.shape[1],
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "test_size": test_size,
            "scale_pos_weight": scale_pos_weight,
        })

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            build_objective_cv(X_trainval, y_trainval, scale_pos_weight, model_type, cv_folds),
            n_trials=n_trials,
        )

        best = study.best_params.copy()
        best_threshold = best.pop("threshold")

        # Reconstruir con _build_model para incluir params fijos (scale_pos_weight, etc.)
        best_model = _build_model(model_type, best, scale_pos_weight)
        best_model.fit(X_trainval, y_trainval)

        metrics, _, _ = evaluate_model(best_model, X_test, y_test, best_threshold)

        mlflow.log_params({**best, "best_threshold": best_threshold})
        mlflow.log_metrics({f"best_{k}": v for k, v in metrics.items()})
        _log_model(best_model, model_type)

        # feature_importances solo en modelos no-pipeline
        raw_model = best_model.named_steps.get("model", best_model) if isinstance(best_model, Pipeline) else best_model
        if hasattr(raw_model, "feature_importances_"):
            importance = dict(zip(X_trainval.columns, raw_model.feature_importances_))
            mlflow.log_dict(importance, "feature_importance.json")

        save_artifacts(
            model=best_model,
            threshold=best_threshold,
            metrics=metrics,
            best_params=best,
            run_id=parent_run.info.run_id,
            model_type=model_type,
            feature_set=feature_set,
            model_dir=save_dir,
        )

        print(f"\n{'=' * 55}")
        print(f"  {model_type} | features={feature_set} | n={X_trainval.shape[1]} cols")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  threshold: {best_threshold:.3f}")
        print(f"{'=' * 55}")

        return {
            "model_type": model_type,
            "feature_set": feature_set,
            "model": best_model,
            "threshold": best_threshold,
            "metrics": metrics,
            "best_params": best,
            "X_trainval": X_trainval,
            "X_test": X_test,
            "y_trainval": y_trainval,
            "y_test": y_test,
            "study": study,
            "run_id": parent_run.info.run_id,
        }
