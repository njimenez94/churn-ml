from src.data.clean_data import clean_data
from src.data.validate_data import validate_data
from src.features.build_features import build_features
import pandas as pd

FILE = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'


def main():
    df = pd.read_csv(FILE)
    df = clean_data(df)

    success, _ = validate_data(df)
    if not success:
        raise ValueError("Data validation failed. Check the output above for details.")

    df = build_features(df)
    df.to_csv('data/processed/data_clean.csv', index=False)
    print(f"✅ Pipeline completo. Output: data/processed/data_clean.csv | Shape: {df.shape}")


if __name__ == '__main__':
    main()