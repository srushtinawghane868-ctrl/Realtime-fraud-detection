"""
train_model.py - Train Anomaly Detection Model for Real-Time Fraud Detection

This script generates a reproducible synthetic dataset of financial transactions,
preprocesses the features, trains a scikit-learn Isolation Forest model,
evaluates its baseline performance against synthetic labels, and serializes
the trained model bundle (pipeline, scaler, feature list) using joblib.

Model Features:
  - amount (float): Monetary value of transaction in USD
  - transaction_hour (int): Hour of transaction (0-23)
  - distance_from_home_km (float): Distance from customer's home location (km)
  - transaction_frequency_1h (int): Number of transactions in the previous hour
  - merchant_risk_score (float): Historical risk score of the merchant (0.0 to 1.0)
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Canonical feature ordering used across training, batch tests, and streaming inference
FEATURE_NAMES = [
    "amount",
    "transaction_hour",
    "distance_from_home_km",
    "transaction_frequency_1h",
    "merchant_risk_score",
]


def generate_synthetic_transactions(n_samples: int = 15000, anomaly_ratio: float = 0.04, random_state: int = 42):
    """
    Generate realistic synthetic financial transactions.

    Returns:
        pd.DataFrame: DataFrame containing features, metadata, and ground truth synthetic labels.
    """
    np.random.seed(random_state)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    print(f"[*] Generating {n_samples:,} synthetic transactions ({n_normal:,} normal, {n_anomalies:,} anomalies)...")

    # --- 1. Normal Transactions ---
    # Amounts: Log-normal distribution (median ~$35, typical everyday purchases)
    normal_amounts = np.random.lognormal(mean=3.5, sigma=0.8, size=n_normal)
    normal_amounts = np.clip(normal_amounts, 2.0, 350.0)

    # Hours: Bimodal peak around lunch (12-14) and evening (18-20)
    hour_choices = np.concatenate([
        np.random.normal(loc=13.0, scale=2.5, size=n_normal // 2),
        np.random.normal(loc=19.0, scale=2.0, size=n_normal - n_normal // 2)
    ])
    normal_hours = np.clip(hour_choices.astype(int), 0, 23)

    # Distances: Most near home (half-normal, median ~4km, 95% < 25km)
    normal_distances = np.abs(np.random.normal(loc=4.0, scale=6.0, size=n_normal))
    normal_distances = np.clip(normal_distances, 0.1, 45.0)

    # Frequency: Low frequency in past hour (Poisson lambda=1.1 + 1)
    normal_frequencies = np.random.poisson(lam=1.1, size=n_normal) + 1
    normal_frequencies = np.clip(normal_frequencies, 1, 4)

    # Merchant risk: Low risk for typical legitimate merchants
    normal_merchant_risk = np.random.beta(a=2, b=10, size=n_normal)  # mean ~ 0.16

    df_normal = pd.DataFrame({
        "amount": np.round(normal_amounts, 2),
        "transaction_hour": normal_hours,
        "distance_from_home_km": np.round(normal_distances, 2),
        "transaction_frequency_1h": normal_frequencies,
        "merchant_risk_score": np.round(normal_merchant_risk, 3),
        "is_synthetic_anomaly": 0,
    })

    # --- 2. Anomalous Transactions ---
    # Anomalies exhibit abnormal amounts, unusual night hours, sudden distance jumps,
    # transaction velocity bursts, or high-risk merchants.
    anomaly_types = np.random.choice(
        ["high_amount", "far_distance", "velocity_burst", "sketchy_night_merchant"],
        size=n_anomalies,
        p=[0.35, 0.30, 0.20, 0.15]
    )

    anom_amounts = []
    anom_hours = []
    anom_distances = []
    anom_frequencies = []
    anom_risks = []

    for atype in anomaly_types:
        if atype == "high_amount":
            # Very large unusual transaction ($1,200 to $6,500)
            anom_amounts.append(np.random.uniform(1200.0, 6500.0))
            anom_hours.append(np.random.randint(0, 24))
            anom_distances.append(np.random.uniform(1.0, 50.0))
            anom_frequencies.append(np.random.randint(1, 3))
            anom_risks.append(np.random.uniform(0.1, 0.6))
        elif atype == "far_distance":
            # Transaction thousands of km away (e.g. international / impossible travel)
            anom_amounts.append(np.random.uniform(50.0, 800.0))
            anom_hours.append(np.random.randint(0, 24))
            anom_distances.append(np.random.uniform(400.0, 4500.0))
            anom_frequencies.append(np.random.randint(1, 3))
            anom_risks.append(np.random.uniform(0.3, 0.7))
        elif atype == "velocity_burst":
            # High card testing / rapid consecutive transactions
            anom_amounts.append(np.random.uniform(20.0, 400.0))
            anom_hours.append(np.random.randint(0, 24))
            anom_distances.append(np.random.uniform(1.0, 30.0))
            anom_frequencies.append(np.random.randint(8, 20))
            anom_risks.append(np.random.uniform(0.4, 0.85))
        else:  # sketchy_night_merchant
            # Late night (2-4 AM), high-risk merchant, high amount
            anom_amounts.append(np.random.uniform(500.0, 2500.0))
            anom_hours.append(np.random.choice([1, 2, 3, 4]))
            anom_distances.append(np.random.uniform(10.0, 200.0))
            anom_frequencies.append(np.random.randint(2, 6))
            anom_risks.append(np.random.uniform(0.75, 0.98))

    df_anom = pd.DataFrame({
        "amount": np.round(anom_amounts, 2),
        "transaction_hour": anom_hours,
        "distance_from_home_km": np.round(anom_distances, 2),
        "transaction_frequency_1h": anom_frequencies,
        "merchant_risk_score": np.round(anom_risks, 3),
        "is_synthetic_anomaly": 1,
    })

    # Combine and shuffle
    df = pd.concat([df_normal, df_anom], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Attach synthetic identifiers for realistic dataset structure
    df.insert(0, "transaction_id", [f"TXN-{100000 + i}" for i in range(len(df))])
    df.insert(1, "customer_id", [f"CUST-{np.random.randint(1000, 9999)}" for _ in range(len(df))])
    df.insert(2, "timestamp", datetime.utcnow().isoformat())

    return df


def train_and_evaluate(df: pd.DataFrame, contamination: float = 0.04, random_state: int = 42):
    """
    Train Isolation Forest model on transaction features and evaluate performance.
    """
    X = df[FEATURE_NAMES].copy()
    y_true = df["is_synthetic_anomaly"].values

    print("\n[*] Feature Summary Statistics:")
    print(X.describe().T[["mean", "std", "min", "50%", "max"]])

    # Feature scaling: standardize features for consistent distance calculations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize Isolation Forest
    print(f"\n[*] Training Isolation Forest (n_estimators=150, contamination={contamination})...")
    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Evaluation
    # Isolation Forest: -1 for anomaly, +1 for normal.
    raw_preds = model.predict(X_scaled)
    # Map to 1 for anomaly, 0 for normal
    y_pred = np.where(raw_preds == -1, 1, 0)
    decision_scores = model.decision_function(X_scaled)

    print("\n" + "=" * 60)
    print("  MODEL BASELINE EVALUATION (SYNTHETIC BENCHMARK)")
    print("=" * 60)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], digits=4))

    auc = roc_auc_score(y_true, -decision_scores)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("=" * 60)

    # Bundle artifacts
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "contamination": contamination,
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": "IsolationForest",
        "n_samples_trained": len(df),
    }

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train Fraud Detection Anomaly Model")
    parser.add_argument("--samples", type=int, default=15000, help="Number of synthetic samples to generate")
    parser.add_argument("--contamination", type=float, default=0.04, help="Expected anomaly contamination ratio")
    parser.add_argument("--output-model", type=str, default="models/fraud_detection_pipeline.joblib", help="Output model path")
    parser.add_argument("--output-data", type=str, default="data/synthetic_transactions.csv", help="Output CSV dataset path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)

    # 1. Generate Dataset
    df = generate_synthetic_transactions(n_samples=args.samples, anomaly_ratio=args.contamination)
    df.to_csv(args.output_data, index=False)
    print(f"[+] Saved synthetic dataset to: {args.output_data}")

    # 2. Train and Evaluate
    bundle = train_and_evaluate(df, contamination=args.contamination)

    # 3. Serialize Model Bundle
    joblib.dump(bundle, args.output_model)
    print(f"[+] Successfully serialized trained model bundle to: {args.output_model}")


if __name__ == "__main__":
    main()

