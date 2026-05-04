from src.data.clean_data import clean_data
from src.data.validate_data import validate_data
from src.features.build_features import build_features
from src.models.train_model import train_model
import pandas as pd

FILE = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
PROCESSED_PATH = 'data/processed/data_clean.csv'

MODELS = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]
N_TRIALS = 100
CV_FOLDS = 10

# Columnas originales sin feature engineering (solo numericas + binarias basicas)
_BASELINE_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Baseline + features construidas (ratios, buckets, flags)
_ENGINEERED_COLS = _BASELINE_COLS + [
    "services_count", "tenure_bucket", "is_new",
    "avg_monthly_spend", "charges_ratio", "high_value_short_tenure",
]

FEATURE_SETS = {
    "baseline": _BASELINE_COLS,
    "with_engineered": _ENGINEERED_COLS,
    "full": None,  # todas: baseline + engineered + Contract/PaymentMethod dummies
}


def print_comparison(results):
    print("\n" + "=" * 72)
    print("COMPARACION FINAL: MODELOS x FEATURE SETS")
    print("=" * 72)
    print(f"{'Modelo':<22} {'Features':<18} {'F1':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6}")
    print("-" * 72)
    for r in sorted(results, key=lambda r: r["metrics"]["f1"], reverse=True):
        m = r["metrics"]
        print(
            f"{r['model_type']:<22} {r['feature_set']:<18} "
            f"{m['f1']:>6.4f} {m['auc']:>6.4f} "
            f"{m['precision']:>6.4f} {m['recall']:>6.4f}"
        )
    print("=" * 72)
    best = max(results, key=lambda r: r["metrics"]["f1"])
    print(f"\nMejor: {best['model_type']} + {best['feature_set']}  F1={best['metrics']['f1']:.4f}")
    print("Ver todos los runs: mlflow ui")


def main():
    df = pd.read_csv(FILE)
    df = clean_data(df)

    success, _ = validate_data(df)
    if not success:
        raise ValueError("Data validation failed.")

    df = build_features(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Pipeline de datos completo. Shape: {df.shape}")

    total = len(MODELS) * len(FEATURE_SETS)
    results = []

    for i, model_type in enumerate(MODELS):
        for j, (fs_name, fs_cols) in enumerate(FEATURE_SETS.items()):
            run_num = i * len(FEATURE_SETS) + j + 1
            print(f"\n[{run_num}/{total}] {model_type} | features={fs_name} | {N_TRIALS} trials x {CV_FOLDS}-fold CV")
            result = train_model(
                df,
                model_type=model_type,
                feature_set=fs_name,
                feature_cols=fs_cols,
                n_trials=N_TRIALS,
                cv_folds=CV_FOLDS,
            )
            results.append(result)

    print_comparison(results)


if __name__ == '__main__':
    main()
