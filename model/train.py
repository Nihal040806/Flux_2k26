"""
train.py - LSTM Model Training for Fraud Detection
===================================================
Builds, compiles, trains, and saves a Keras LSTM model using
preprocessed transaction sequences.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Local import
from preprocess import run_preprocessing


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.h5")

EPOCHS = 5
BATCH_SIZE = 256
TEST_SPLIT = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Build LSTM Model
# ---------------------------------------------------------------------------
def build_model(input_shape: tuple) -> Sequential:
    """
    Construct the LSTM-based fraud detection model.

    Architecture:
        LSTM(64) → Dense(32, relu) → Dense(1, sigmoid)
    """
    model = Sequential([
        # LSTM layer: learns temporal patterns in transaction sequences
        LSTM(64, input_shape=input_shape),

        # Dropout to reduce overfitting
        Dropout(0.3),

        # Hidden dense layer for feature extraction
        Dense(32, activation="relu"),

        # Dropout before final output
        Dropout(0.2),

        # Output layer: binary classification (fraud / not fraud)
        Dense(1, activation="sigmoid"),
    ])

    # Compile with binary crossentropy (binary classification task)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("\n📐 Model Summary:")
    model.summary()
    return model


# ---------------------------------------------------------------------------
# 2. Train Pipeline
# ---------------------------------------------------------------------------
def train():
    """Full training pipeline: preprocess → build → train → save."""

    # ── Step 1: Preprocess data ──────────────────────────────────────────
    X, y, scaler = run_preprocessing()

    # ── Step 2: Train/test split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n📊 Split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"   Train fraud ratio: {y_train.mean():.4f}")
    print(f"   Test  fraud ratio: {y_test.mean():.4f}")

    # ── Step 3: Build model ──────────────────────────────────────────────
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_model(input_shape)

    # ── Step 4: Compute class weights to handle imbalanced data ──────────
    n_total = len(y_train)
    n_fraud = int(y_train.sum())
    n_legit = n_total - n_fraud

    if n_fraud > 0:
        weight_legit = n_total / (2 * n_legit)
        weight_fraud = n_total / (2 * n_fraud)
        class_weight = {0: weight_legit, 1: weight_fraud}
        print(f"\n⚖️  Class weights: legit={weight_legit:.2f}, fraud={weight_fraud:.2f}")
    else:
        class_weight = None
        print("\n⚠️  No fraud samples found; training without class weights.")

    # ── Step 5: Train ────────────────────────────────────────────────────
    print(f"\n🚀 Training for {EPOCHS} epochs (batch_size={BATCH_SIZE})...\n")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weight,
        verbose=1,
    )

    # ── Step 6: Evaluate ─────────────────────────────────────────────────
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test Loss:     {loss:.4f}")
    print(f"✅ Test Accuracy: {accuracy:.4f}")

    # ── Step 7: Save model ───────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")

    return model, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Fraud Detection - LSTM Model Training")
    print("=" * 60)
    model, history = train()
    print("\n🎉 Training complete!")
