import pandas as pd

COLS_NO_SERVICE = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'MultipleLines'
]

BINS_COL = [
    'Churn',
    'Partner', 'Dependents', 'PhoneService',
    'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling'
]

def _string_to_numeric(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def _clean_no_service(df, cols):
    replace_map = {
        'No internet service': None,
        'No phone service': None
    }
    for col in cols:
        df[col] = df[col].replace(replace_map)  
    return df

def _convert_binary_to_numeric(df, cols):
    for col in cols:
        df[col] = df[col].replace({'Yes': 1, 'No': 0})
    return df

def clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0).clip(lower=0)
    df = _clean_no_service(df, COLS_NO_SERVICE)
    df = _convert_binary_to_numeric(df, BINS_COL)
    print(f"Data cleaned. Shape: {df.shape}")
    return df