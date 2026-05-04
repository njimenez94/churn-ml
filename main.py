from src.data.clean_data import clean_data
from src.data.validate_data import validate_data
from src.features.build_features import build_features
from src.models.train_model import train_model
import pandas as pd

FILE = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
PROCESSED_PATH = 'data/processed/data_clean.csv'


def main():
    df = pd.read_csv(FILE)
    df = clean_data(df)

    success, _ = validate_data(df)
    if not success:
        raise ValueError("Data validation failed. Check the output above for details.")

    df = build_features(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Pipeline de datos completo. Shape: {df.shape}")

    print("🚀 Iniciando entrenamiento del modelo...")
    train_model(df, feature_set="v3_clean", n_trials=100)


if __name__ == '__main__':
    main()