import pandas as pd


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy()

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for c in service_cols:
        if c in df.columns:
            df[c] = (df[c].astype(str) == 'Yes').astype(int)

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    for c in binary_cols:
        vals = sorted(df[c].dropna().astype(str).unique())
        df[c] = df[c].astype(str).map({vals[0]: 0, vals[1]: 1}).fillna(0).astype(int)

    available = [c for c in service_cols if c in df.columns]
    if available:
        df['services_count'] = df[available].sum(axis=1)

    if 'tenure' in df.columns:
        df['tenure_bucket'] = pd.cut(
            df['tenure'], bins=[-1, 6, 12, 24, 48, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        df['is_new'] = (df['tenure'] < 6).astype(int)

    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['avg_monthly_spend'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

    if 'MonthlyCharges' in df.columns and 'avg_monthly_spend' in df.columns:
        df['charges_ratio'] = df['MonthlyCharges'] / df['avg_monthly_spend'].replace(0, 1)

    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['high_value_short_tenure'] = (
            (df['MonthlyCharges'] > df['MonthlyCharges'].median()) &
            (df['tenure'] < 12)
        ).astype(int)

    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True, dtype=int)

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    return df