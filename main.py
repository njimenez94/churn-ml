from src.data.clean_data import clean_data

FILE = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

def main():
    df = clean_data(FILE)
    df.to_csv('data/processed/data_clean.csv', index=False)

if __name__ == '__main__':
    main()