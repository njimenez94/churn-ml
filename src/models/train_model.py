import optuna
import mlflow
import mlflow.xgboost
import json
import joblib
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def split_data(df, target_col="Churn", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_objective(X_train, y_train, X_test, y_test, scale_pos_weight):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }
        threshold = trial.suggest_float("threshold", 0.3, 0.7)

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= threshold).astype(int)

            metrics = {
                "f1": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "accuracy": accuracy_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, proba),
            }

            mlflow.log_params({**params, "threshold": threshold})
            mlflow.log_metrics(metrics)

            return metrics["f1"]

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


def save_artifacts(model, threshold, metrics, best_params, run_id, feature_set, model_dir="models"):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    joblib.dump(model, model_dir / "best_model.pkl")
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({
            "threshold": threshold,
            "metrics": metrics,
            "best_params": best_params,
            "run_id": run_id,
            "feature_set": feature_set,
        }, f, indent=2)


def train_model(
    df,
    experiment_name="churn_xgboost",
    feature_set="default",
    n_trials=100,
    target_col="Churn",
    test_size=0.2,
    random_state=42,
    save_dir="models",
):
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = split_data(df, target_col, test_size, random_state)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    with mlflow.start_run(run_name=f"optuna_{feature_set}") as parent_run:
        mlflow.set_tag("feature_set", feature_set)
        mlflow.log_params({
            "n_features": X_train.shape[1],
            "n_trials": n_trials,
            "test_size": test_size,
            "scale_pos_weight": scale_pos_weight,
        })

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            build_objective(X_train, y_train, X_test, y_test, scale_pos_weight),
            n_trials=n_trials,
        )

        best = study.best_params.copy()
        best_threshold = best.pop("threshold")

        best_model = XGBClassifier(
            **best,
            random_state=random_state,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        best_model.fit(X_train, y_train)

        metrics, y_pred, proba = evaluate_model(best_model, X_test, y_test, best_threshold)

        mlflow.log_params({**best, "best_threshold": best_threshold})
        mlflow.log_metrics({f"best_{k}": v for k, v in metrics.items()})
        mlflow.xgboost.log_model(best_model, "model")

        importance = dict(zip(X_train.columns, best_model.feature_importances_))
        mlflow.log_dict(importance, "feature_importance.json")

        save_artifacts(
            model=best_model,
            threshold=best_threshold,
            metrics=metrics,
            best_params=best,
            run_id=parent_run.info.run_id,
            feature_set=feature_set,
            model_dir=save_dir,
        )

        print("=" * 50)
        print(f"Feature set: {feature_set}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(f"Threshold: {best_threshold:.3f}")
        print(f"Modelo guardado en {save_dir}/")
        print("=" * 50)

        return {
            "model": best_model,
            "threshold": best_threshold,
            "metrics": metrics,
            "best_params": best,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "study": study,
            "run_id": parent_run.info.run_id,
        }