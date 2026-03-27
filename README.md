# 🛡️ Fraud Detection System using Deep Learning (LSTM)

> Real-time fraud detection powered by LSTM neural networks trained on sequential transaction patterns.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [API Usage](#api-usage)
- [Explainability](#explainability)
- [Tech Stack](#tech-stack)

---

## 🔍 Overview

This project implements an **LSTM-based fraud detection system** that analyzes sequences of past transactions to identify fraudulent behavior in real time. Unlike traditional rule-based systems, our model learns **temporal patterns** across transaction histories to detect anomalies.

### Key Features

- 🧠 **Deep Learning**: LSTM model captures sequential transaction patterns
- ⚡ **Real-time API**: FastAPI backend for instant fraud scoring
- 📊 **Explainability**: Rule-based reasoning for every prediction
- 🎯 **Hackathon-optimized**: Fast training (~5 epochs), minimal setup

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Raw Data   │ ──► │ Preprocessor │ ──► │  LSTM Model     │
│  (CSV)      │     │ (Sequences)  │     │  (TensorFlow)   │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Frontend   │ ◄── │  FastAPI     │ ◄── │  Saved Model    │
│  (Client)   │     │  Backend     │     │  (.h5 + scaler) │
└─────────────┘     └──────────────┘     └─────────────────┘
```

---

## 📁 Project Structure

```
fraud-project/
│
├── data/
│   ├── train_transaction.csv      # IEEE-CIS transaction data
│   └── train_identity.csv         # IEEE-CIS identity data
│
├── model/
│   ├── preprocess.py              # Data loading, cleaning, sequence creation
│   ├── train.py                   # LSTM model building & training
│   ├── fraud_model.h5             # Trained model (generated)
│   └── scaler.pkl                 # Feature scaler (generated)
│
├── backend/
│   └── app.py                     # FastAPI REST API
│
├── frontend/                      # Placeholder for future UI
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install Dependencies

```bash
cd fraud-project
pip install -r requirements.txt
```

---

## 📦 Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle.

### Download Instructions

1. Go to: [https://www.kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place both files in the `data/` directory

```bash
fraud-project/data/
├── train_transaction.csv
└── train_identity.csv
```

> **Note**: You need a Kaggle account to download the dataset. Alternatively, use the Kaggle CLI:
> ```bash
> kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/
> kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/
> ```

---

## 🏋️ Training the Model

```bash
cd model
python train.py
```

### What happens during training:

1. **Load & Merge**: Combines transaction and identity data on `TransactionID`
2. **Feature Selection**: Uses `TransactionAmt`, `TransactionDT`, `card1`, `addr1`, `isFraud`
3. **Sampling**: Takes 50,000 rows for fast training
4. **Sequence Creation**: Builds sliding windows of 10 transactions per user (`card1`)
5. **Scaling**: StandardScaler normalization (saved as `scaler.pkl`)
6. **Training**: 5 epochs with class-weight balancing for imbalanced data
7. **Output**: Saves `fraud_model.h5` and `scaler.pkl`

### Expected Output

```
============================================================
  Fraud Detection - LSTM Model Training
============================================================
[1/5] Loading datasets...
[2/5] Selecting features & cleaning...
[3/5] Building sequences (window=10)...
[4/5] Scaling features...
[5/5] Preprocessing complete!

📊 Split: Train=XXXX, Test=XXXX
🚀 Training for 5 epochs (batch_size=256)...

✅ Test Accuracy: ~0.96+
💾 Model saved to model/fraud_model.h5
🎉 Training complete!
```

---

## 🚀 Running the API

```bash
cd backend
python app.py
```

The server starts at: **http://localhost:8000**

- Swagger docs: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**

---

## 📡 API Usage

### Health Check

```bash
GET /
```

```json
{
  "status": "running",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### Predict Fraud

```bash
POST /predict
Content-Type: application/json
```

**Request Body:**

```json
{
  "sequence": [
    [150.0, 315.0],
    [22.5, 315.0],
    [89.99, 315.0],
    [450.0, 204.0],
    [12.0, 315.0],
    [67.5, 315.0],
    [200.0, 315.0],
    [33.0, 315.0],
    [99.0, 315.0],
    [1500.0, 100.0]
  ]
}
```

Each inner list = `[TransactionAmt, addr1]` (one timestep).

**Response:**

```json
{
  "fraud_probability": 0.8234,
  "is_fraud": true,
  "reasons": [
    "High amount spike: last transaction ($1500.00) is >3x the average ($124.89)",
    "New location detected: addr1=100 differs from usual addr1=315"
  ]
}
```

### cURL Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      [150.0, 315.0], [22.5, 315.0], [89.99, 315.0],
      [450.0, 204.0], [12.0, 315.0], [67.5, 315.0],
      [200.0, 315.0], [33.0, 315.0], [99.0, 315.0],
      [1500.0, 100.0]
    ]
  }'
```

---

## 🧠 Explainability

The system provides **rule-based explanations** alongside model predictions:

| Rule | Trigger | Explanation |
|------|---------|-------------|
| 💰 Amount Spike | Last txn > 3× average | "High amount spike detected" |
| 📍 Location Change | Last addr ≠ usual addr | "New location detected" |
| 📈 Rapid High-Value | ≥3 txns above 90th percentile | "Multiple high-value transactions" |
| 🌐 Location Diversity | ≥4 unique addresses | "High location diversity" |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow / Keras |
| Model | LSTM (Long Short-Term Memory) |
| API Framework | FastAPI |
| Data Processing | Pandas, NumPy |
| Feature Scaling | scikit-learn (StandardScaler) |
| Serialization | joblib |
| Server | Uvicorn |

---

## 📝 License

MIT License - Feel free to use for hackathons, learning, and production.

---

## 👥 Team

Built for hackathon submission — **Fraud Detection using Deep Learning**.

---

*Built with ❤️ and LSTM magic*
