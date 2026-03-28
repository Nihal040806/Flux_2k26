import pandas as pd
import numpy as np
import random
import os

os.makedirs('data', exist_ok=True)
n = 1000
df = pd.DataFrame({
    'TransactionID': np.arange(100000, 100000+n),
    'TransactionAmt': np.random.lognormal(mean=3, sigma=1, size=n),
    'TransactionDT': np.arange(80000, 80000 + n * 30, 30),
    'ProductCD': np.random.choice(['W', 'C', 'R', 'H'], size=n),
    'card1': np.random.randint(1000, 15000, size=n),
    'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover'], size=n),
    'addr1': np.random.randint(100, 500, size=n),
    'addr2': 87,
    'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com'], size=n),
    'isFraud': np.random.choice([0, 1], p=[0.95, 0.05], size=n)
})
df.to_csv('data/train_transaction.csv', index=False)
print("done")
