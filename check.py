import traceback
import pandas as pd

CSV_COLS = ['TransactionID', 'TransactionAmt', 'TransactionDT', 'ProductCD', 'card1', 'card4', 'addr1', 'addr2', 'P_emaildomain', 'isFraud']
TRANSACTION_FILE = 'data/train_transaction.csv'
LIVE_CSV_PATH = 'data/live_transactions.csv'

try:
    print("Loading df_transactions...")
    df_transactions = pd.read_csv(TRANSACTION_FILE, usecols=CSV_COLS, nrows=50_000, low_memory=False)
    print("Filling NAs in df_transactions...")
    for col in df_transactions.columns:
        fill_val = "" if col in ["ProductCD", "card4", "P_emaildomain"] else 0
        df_transactions[col] = df_transactions[col].fillna(fill_val)
    
    print("Loading df_live...")
    import os
    if os.path.exists(LIVE_CSV_PATH):
        df_live = pd.read_csv(LIVE_CSV_PATH, low_memory=False)
        print("Filling NAs in df_live...")
        for col in df_live.columns:
            fill_val = "" if col in ["ProductCD", "card4", "P_emaildomain"] else 0
            df_live[col] = df_live[col].fillna(fill_val)
        print("Concatenating...")
        df_transactions = pd.concat([df_live, df_transactions], ignore_index=True)

    print("Precomputing baseline stats...")
    total = len(df_transactions)
    print("summing isFraud...")
    fraud_count = int(df_transactions["isFraud"].sum())
    print("summing amt...")
    legit_count = total - fraud_count
    total_volume = float(df_transactions["TransactionAmt"].sum())
    
    print("SUCCESS!")
except Exception as e:
    traceback.print_exc()
