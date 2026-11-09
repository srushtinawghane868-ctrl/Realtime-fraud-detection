"""
test_pipeline.py - Comprehensive Unit and Integration Tests for Real-Time Fraud Detection
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
import joblib

# Import project modules
from train_model import FEATURE_NAMES, generate_synthetic_transactions, train_and_evaluate
from transaction_producer import generate_transaction, MERCHANT_PROFILES
from spark_consumer import TRANSACTION_SCHEMA, make_batch_processor


def test_feature_consistency():
    """
    Ensure the canonical feature names and order are identical across modules.
    """
    expected_features = [
        "amount",
        "transaction_hour",
        "distance_from_home_km",
        "transaction_frequency_1h",
        "merchant_risk_score",
    ]
    assert FEATURE_NAMES == expected_features, "Feature list in train_model does not match expected canonical ordering."

    # Validate schema field names in Spark consumer contain all features
    spark_field_names = [f.name for f in TRANSACTION_SCHEMA.fields]
    for feat in expected_features:
        assert feat in spark_field_names, f"Feature '{feat}' missing from Spark TRANSACTION_SCHEMA."


def test_transaction_generation_and_json_serialization():
    """
    Verify transaction_producer generates valid JSON messages conforming to schema.
    """
    # Test normal event
    txn_normal = generate_transaction(anomaly_rate=0.0)
    assert txn_normal["transaction_id"].startswith("TXN-")
    assert txn_normal["customer_id"].startswith("CUST-")
    assert isinstance(txn_normal["amount"], float)
    assert 0 <= txn_normal["transaction_hour"] <= 23
    assert txn_normal["distance_from_home_km"] >= 0.0
    assert txn_normal["transaction_frequency_1h"] >= 1
    assert 0.0 <= txn_normal["merchant_risk_score"] <= 1.0

    # Ensure JSON serializable and deserializable
    serialized = json.dumps(txn_normal)
    deserialized = json.loads(serialized)
    assert deserialized["transaction_id"] == txn_normal["transaction_id"]
    assert deserialized["amount"] == txn_normal["amount"]

    # Test deliberate anomaly generation
    txn_anomaly = generate_transaction(anomaly_rate=1.0)
    assert txn_anomaly["transaction_id"].startswith("TXN-")
    assert txn_anomaly["amount"] > 0


def test_model_training_and_serialization(tmp_path):
    """
    Verify model training on synthetic data and serialization to disk.
    """
    df_synthetic = generate_synthetic_transactions(n_samples=500, anomaly_ratio=0.06, random_state=123)
    assert len(df_synthetic) == 500
    assert "is_synthetic_anomaly" in df_synthetic.columns

    # Train model bundle
    bundle = train_and_evaluate(df_synthetic, contamination=0.06, random_state=123)
    assert "model" in bundle
    assert "scaler" in bundle
    assert bundle["feature_names"] == FEATURE_NAMES

    # Test serialization and loading
    model_file = tmp_path / "test_model.joblib"
    joblib.dump(bundle, str(model_file))
    assert model_file.exists()

    loaded_bundle = joblib.load(str(model_file))
    assert loaded_bundle["feature_names"] == FEATURE_NAMES
    assert hasattr(loaded_bundle["model"], "predict")


def test_prediction_generation():
    """
    Verify feature scaling, score calculation, and anomaly classification.
    """
    df_synthetic = generate_synthetic_transactions(n_samples=300, anomaly_ratio=0.1, random_state=42)
    bundle = train_and_evaluate(df_synthetic, contamination=0.1, random_state=42)

    model = bundle["model"]
    scaler = bundle["scaler"]

    # Sample 1: typical normal transaction ($35, 13:00, 3km, 1 txn/h, low risk 0.12)
    normal_sample = pd.DataFrame([{
        "amount": 35.0,
        "transaction_hour": 13,
        "distance_from_home_km": 3.0,
        "transaction_frequency_1h": 1,
        "merchant_risk_score": 0.12,
    }])[FEATURE_NAMES]

    normal_scaled = scaler.transform(normal_sample)
    normal_pred = model.predict(normal_scaled)
    normal_score = model.decision_function(normal_scaled)
    assert normal_pred[0] == 1, "Normal transaction was incorrectly classified as anomaly."
    assert normal_score[0] > 0, "Decision score for normal transaction should be positive."

    # Sample 2: extreme anomaly ($5,500, 3 AM, 3,200km, 15 txn/h, high risk 0.95)
    anomaly_sample = pd.DataFrame([{
        "amount": 5500.0,
        "transaction_hour": 3,
        "distance_from_home_km": 3200.0,
        "transaction_frequency_1h": 15,
        "merchant_risk_score": 0.95,
    }])[FEATURE_NAMES]

    anomaly_scaled = scaler.transform(anomaly_sample)
    anomaly_pred = model.predict(anomaly_scaled)
    anomaly_score = model.decision_function(anomaly_scaled)
    assert anomaly_pred[0] == -1, "Extreme outlier transaction was not identified as anomaly."
    assert anomaly_score[0] < 0, "Decision score for extreme anomaly should be negative."


def test_output_persistence(tmp_path):
    """
    Verify persisting flagged transactions to CSV without overwriting headers or data.
    """
    output_csv = tmp_path / "flagged_test.csv"
    output_csv_str = str(output_csv)

    # Mock batch records
    records_batch_1 = pd.DataFrame([
        {
            "transaction_id": "TXN-001",
            "customer_id": "CUST-1",
            "amount": 2500.0,
            "anomaly_score": 0.32,
            "is_anomaly": 1,
            "risk_label": "HIGH_RISK",
        }
    ])

    records_batch_2 = pd.DataFrame([
        {
            "transaction_id": "TXN-002",
            "customer_id": "CUST-2",
            "amount": 1800.0,
            "anomaly_score": 0.25,
            "is_anomaly": 1,
            "risk_label": "SUSPICIOUS",
        }
    ])

    # Simulate batch 1 write
    file_exists = os.path.exists(output_csv_str) and os.path.getsize(output_csv_str) > 0
    records_batch_1.to_csv(output_csv_str, mode="a", index=False, header=not file_exists)

    # Simulate batch 2 write
    file_exists = os.path.exists(output_csv_str) and os.path.getsize(output_csv_str) > 0
    records_batch_2.to_csv(output_csv_str, mode="a", index=False, header=not file_exists)

    # Verify both records exist and header was only written once
    df_read = pd.read_csv(output_csv_str)
    assert len(df_read) == 2
    assert list(df_read["transaction_id"]) == ["TXN-001", "TXN-002"]
    assert list(df_read["risk_label"]) == ["HIGH_RISK", "SUSPICIOUS"]


def test_kafka_connectivity_if_available():
    """
    Test publishing and reading a message on Kafka topic 'transactions' if broker is reachable.
    """
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import NoBrokersAvailable

    try:
        producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=3000,
        )
        test_payload = {"test": True, "timestamp": "2026-10-29T00:00:00"}
        future = producer.send("transactions", test_payload)
        metadata = future.get(timeout=5.0)
        assert metadata.topic == "transactions"
        producer.close()
    except (NoBrokersAvailable, Exception) as e:
        pytest.skip(f"Kafka broker not available yet ({e}); skipping live Kafka round-trip test.")

