"""
preprocess.py - Data Preprocessing for Fraud Detection LSTM Model
=================================================================
Handles loading, cleaning, merging, and sequence generation from
the IEEE-CIS Fraud Detection dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.dirname(__file__)

TRANSACTION_FILE = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_FILE = os.path.join(DATA_DIR, "train_identity.csv")

SELECTED_FEATURES = ["TransactionAmt", "TransactionDT", "card1", "addr1", "isFraud"]
SEQUENCE_FEATURES = ["TransactionAmt", "addr1"]  # features used inside each timestep

SEQUENCE_LENGTH = 10   # number of past transactions per sequence
SAMPLE_SIZE = 50_000   # rows to sample for fast training
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load & Merge
# ---------------------------------------------------------------------------
def load_and_merge() -> pd.DataFrame:
    """Load transaction and identity CSVs and merge on TransactionID."""
    print("[1/5] Loading datasets...")
    txn = pd.read_csv(TRANSACTION_FILE)
    idn = pd.read_csv(IDENTITY_FILE)

    print(f"       Transactions shape: {txn.shape}")
    print(f"       Identity shape:     {idn.shape}")

    # Left join keeps all transactions even without identity info
    merged = txn.merge(idn, on="TransactionID", how="left")
    print(f"       Merged shape:       {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# 2. Select & Clean
# ---------------------------------------------------------------------------
def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant features, fill NaN, sort, and sample."""
    print("[2/5] Selecting features & cleaning...")
    df = df[SELECTED_FEATURES].copy()

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Sort by user (card1) then time so sequences are chronological
    df.sort_values(by=["card1", "TransactionDT"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Sample for fast training
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
        df.sort_values(by=["card1", "TransactionDT"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"       Sampled down to {SAMPLE_SIZE} rows")

    print(f"       Final shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 3. Build Sequences
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame):
    """
    Group transactions by card1 (proxy for user ID) and create
    sliding-window sequences of length SEQUENCE_LENGTH.

    Returns:
        X : np.ndarray of shape (num_samples, SEQUENCE_LENGTH, num_features)
        y : np.ndarray of shape (num_samples,)
    """
    print("[3/5] Building sequences (window={})...".format(SEQUENCE_LENGTH))

    sequences = []
    labels = []

    # Group by card1 (user proxy)
    grouped = df.groupby("card1")

    for card_id, group in grouped:
        # Extract feature columns and label column
        features = group[SEQUENCE_FEATURES].values  # shape: (n_txns, 2)
        fraud = group["isFraud"].values

        # Sliding window over the user's transactions
        for i in range(len(features) - SEQUENCE_LENGTH):
            seq = features[i : i + SEQUENCE_LENGTH]
            label = fraud[i + SEQUENCE_LENGTH]  # label of the NEXT transaction
            sequences.append(seq)
            labels.append(label)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    print(f"       X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"       Fraud ratio: {y.mean():.4f}")
    return X, y


# ---------------------------------------------------------------------------
# 4. Scale Features
# ---------------------------------------------------------------------------
def scale_features(X: np.ndarray):
    """
    Fit a StandardScaler on the flattened feature matrix and
    transform X in-place. Saves scaler to disk.

    Returns:
        X_scaled : np.ndarray (same shape as input)
        scaler   : fitted StandardScaler
    """
    print("[4/5] Scaling features...")
    num_samples, timesteps, num_features = X.shape

    # Reshape to 2D for scaler: (num_samples * timesteps, num_features)
    X_flat = X.reshape(-1, num_features)

    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)

    # Reshape back to 3D
    X_scaled = X_flat.reshape(num_samples, timesteps, num_features)

    # Persist scaler for inference
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"       Scaler saved to {scaler_path}")

    return X_scaled, scaler


# ---------------------------------------------------------------------------
# 5. Full Pipeline
# ---------------------------------------------------------------------------
def run_preprocessing():
    """Execute the complete preprocessing pipeline and return X, y."""
    print("=" * 60)
    print("  Fraud Detection - Preprocessing Pipeline")
    print("=" * 60)

    df = load_and_merge()
    df = select_and_clean(df)
    X, y = build_sequences(df)
    X, scaler = scale_features(X)

    print("[5/5] Preprocessing complete!\n")
    return X, y, scaler


# ---------------------------------------------------------------------------
# Entry point (for standalone testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y, scaler = run_preprocessing()
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"Fraud samples: {int(y.sum())} / {len(y)}")
