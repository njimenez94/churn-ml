import pandas as pd

def _convert_total_charges(df, col='TotalCharges'):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0).clip(lower=0)
    return df

def _convert_binary_to_numeric(df, cols= ['Churn']):
    for col in cols:
        df[col] = df[col].replace({'Yes': 1, 'No': 0})
    return df

def clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['customerID'])
    df = _convert_total_charges(df)
    df = _convert_binary_to_numeric(df)
    print(f"Data cleaned. Shape: {df.shape}")
    return df