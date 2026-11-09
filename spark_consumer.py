"""
spark_consumer.py - Apache Spark Structured Streaming Consumer for Real-Time Fraud Detection

Subscribes to the Kafka topic 'transactions', parses the streaming JSON events,
validates the schema, runs inference using the pre-trained Isolation Forest model,
displays real-time transaction evaluations in the terminal, and persistently
saves flagged suspicious transactions to a CSV file.
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Configure Java 21 environment for Apache Spark on Windows if JAVA_HOME is not set to JDK 21
DEFAULT_JDK21 = r"C:\Program Files\Android\openjdk\jdk-21.0.8"
if os.path.exists(DEFAULT_JDK21):
    current_java = os.environ.get("JAVA_HOME", "")
    if "jdk-21" not in current_java:
        os.environ["JAVA_HOME"] = DEFAULT_JDK21
        os.environ["PATH"] = f"{DEFAULT_JDK21}\\bin;" + os.environ.get("PATH", "")

# Configure Hadoop and winutils on Windows
DEFAULT_HADOOP = r"C:\hadoop"
if os.path.exists(DEFAULT_HADOOP):
    os.environ["HADOOP_HOME"] = DEFAULT_HADOOP
    os.environ["hadoop.home.dir"] = DEFAULT_HADOOP
    os.environ["PATH"] = f"{DEFAULT_HADOOP}\\bin;" + os.environ.get("PATH", "")

# Ensure PySpark workers use the exact active Python virtual environment binary
# Convert to 8.3 short path on Windows to avoid space issues in batch launchers
def get_safe_python_path(path: str) -> str:
    if os.name == "nt":
        import ctypes
        buffer = ctypes.create_unicode_buffer(500)
        ctypes.windll.kernel32.GetShortPathNameW(path, buffer, 500)
        if buffer.value:
            return buffer.value
    return path

safe_python = get_safe_python_path(sys.executable)
os.environ["PYSPARK_PYTHON"] = safe_python
os.environ["PYSPARK_DRIVER_PYTHON"] = safe_python




from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, expr
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)

# Explicit JSON Schema for incoming Kafka transactions
TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("timestamp", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("merchant_id", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("amount", DoubleType(), False),
    StructField("transaction_hour", IntegerType(), False),
    StructField("distance_from_home_km", DoubleType(), False),
    StructField("transaction_frequency_1h", IntegerType(), False),
    StructField("merchant_risk_score", DoubleType(), False),
])

# Global cumulative counters
cumulative_stats = {
    "total_processed": 0,
    "total_flagged": 0,
    "batches_processed": 0,
}


def load_model_bundle(model_path: str):
    """
    Load serialized model pipeline bundle.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please train the model first by running: python train_model.py"
        )
    print(f"[*] Loading pre-trained model bundle from: {model_path}")
    bundle = joblib.load(model_path)
    print(f"[+] Loaded {bundle.get('model_type', 'Model')} trained on {bundle.get('trained_at', 'unknown date')}")
    print(f"[+] Canonical Feature Order: {bundle['feature_names']}")
    return bundle


def create_spark_session(app_name: str = "FraudDetectionStreaming") -> SparkSession:
    """
    Create and configure SparkSession with Kafka connector package and Windows-friendly settings.
    """
    scala_version = "2.12"
    spark_version = "3.5.1"
    kafka_package = f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}"

    # Required JVM flags for Java 17/21 compatibility with Spark
    java_options = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
    )

    print(f"[*] Initializing SparkSession (Spark {spark_version}, Scala {scala_version})...")
    print(f"[*] Kafka connector package: {kafka_package}")

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages", kafka_package)
        .config("spark.driver.extraJavaOptions", java_options)
        .config("spark.executor.extraJavaOptions", java_options)
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.ui.enabled", "false")  # Disable web UI for local streaming efficiency
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print("[+] SparkSession successfully initialized.")
    return spark


def make_batch_processor(model_bundle: dict, output_csv_path: str):
    """
    Factory creating the foreachBatch handler with closed-over model bundle and output path.
    """
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]

    def process_micro_batch(batch_df, batch_id: int):
        global cumulative_stats
        start_time = time.time()

        if batch_df.isEmpty():
            return

        cumulative_stats["batches_processed"] += 1
        pdf = batch_df.toPandas()
        batch_size = len(pdf)

        if batch_size == 0:
            return

        # 1. Feature Preprocessing (strictly aligned with model training)
        X = pdf[feature_names].copy()
        X_scaled = scaler.transform(X)

        # 2. Model Inference
        # Isolation Forest: -1 for anomaly, +1 for inlier
        raw_preds = model.predict(X_scaled)
        # Decision function: lower values mean more abnormal
        decision_scores = model.decision_function(X_scaled)

        # 3. Anomaly scoring and risk categorization
        # Inverse decision score for an intuitive risk score (higher = riskier)
        anomaly_scores = np.round(-decision_scores, 4)
        is_anomaly = (raw_preds == -1).astype(int)

        risk_labels = []
        for score, pred in zip(decision_scores, is_anomaly):
            if score < -0.10:
                risk_labels.append("HIGH_RISK")
            elif pred == 1:
                risk_labels.append("SUSPICIOUS")
            else:
                risk_labels.append("NORMAL")

        # 4. Enrich DataFrame
        pdf["anomaly_score"] = anomaly_scores
        pdf["is_anomaly"] = is_anomaly
        pdf["risk_label"] = risk_labels
        pdf["evaluation_timestamp"] = datetime.utcnow().isoformat()

        # Update cumulative counters
        flagged_count = int(is_anomaly.sum())
        cumulative_stats["total_processed"] += batch_size
        cumulative_stats["total_flagged"] += flagged_count

        elapsed = time.time() - start_time

        # 5. Live Terminal Dashboard Output
        print("\n" + "=" * 90)
        print(
            f" [STREAMING BATCH #{batch_id}] | "
            f"Transactions: {batch_size} | "
            f"Flagged Anomalies: {flagged_count} | "
            f"Latency: {elapsed * 1000:.1f}ms"
        )
        print("=" * 90)
        print(
            f"{'TXN ID':<17} | {'CUSTOMER':<10} | {'AMOUNT':>8} | "
            f"{'DIST':>7} | {'FREQ':>4} | {'SCORE':>7} | {'STATUS':<11}"
        )
        print("-" * 90)

        for _, row in pdf.iterrows():
            status_str = row["risk_label"]
            if row["risk_label"] == "HIGH_RISK":
                status_display = f"! {status_str} !"
            elif row["risk_label"] == "SUSPICIOUS":
                status_display = f"? {status_str}"
            else:
                status_display = f"  {status_str}"

            print(
                f"{row['transaction_id']:<17} | "
                f"{row['customer_id']:<10} | "
                f"${row['amount']:>7.2f} | "
                f"{row['distance_from_home_km']:>5.1f}km | "
                f"{row['transaction_frequency_1h']:>4d} | "
                f"{row['anomaly_score']:>7.4f} | "
                f"{status_display:<11}"
            )

        print("-" * 90)
        overall_flag_rate = (
            (cumulative_stats["total_flagged"] / cumulative_stats["total_processed"] * 100.0)
            if cumulative_stats["total_processed"] > 0
            else 0.0
        )
        print(
            f" Cumulative Stats: "
            f"Total Processed: {cumulative_stats['total_processed']} | "
            f"Total Flagged: {cumulative_stats['total_flagged']} ({overall_flag_rate:.1f}%) | "
            f"Batches: {cumulative_stats['batches_processed']}"
        )
        print("=" * 90)
        sys.stdout.flush()

        # 6. Persistent Output for Flagged Suspicious Transactions
        flagged_df = pdf[pdf["is_anomaly"] == 1]
        if not flagged_df.empty:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            file_exists = os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0
            flagged_df.to_csv(output_csv_path, mode="a", index=False, header=not file_exists)
            print(f"[+] Persisted {len(flagged_df)} flagged transaction(s) to: {output_csv_path}")
            sys.stdout.flush()

    return process_micro_batch


def main():
    parser = argparse.ArgumentParser(description="Spark Structured Streaming Fraud Detector")
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", type=str, default="transactions", help="Kafka topic to consume from")
    parser.add_argument("--model-path", type=str, default="models/fraud_detection_pipeline.joblib", help="Model file path")
    parser.add_argument("--output-csv", type=str, default="output/flagged_transactions.csv", help="Flagged output CSV path")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/spark_fraud_stream", help="Checkpoint directory")
    parser.add_argument("--trigger-interval", type=str, default="2 seconds", help="Micro-batch trigger interval")
    parser.add_argument("--starting-offsets", type=str, default="latest", help="Kafka starting offsets (latest or earliest)")
    args = parser.parse_args()

    # 1. Load ML Model
    model_bundle = load_model_bundle(args.model_path)

    # 2. Initialize Spark Session
    spark = create_spark_session()

    # 3. Read from Kafka Streaming Source
    print(f"[*] Subscribing to Kafka topic: '{args.topic}' at {args.bootstrap_servers}")
    raw_kafka_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", args.starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )

    # 4. Parse Kafka Message Value as JSON
    parsed_stream = (
        raw_kafka_stream.selectExpr("CAST(value AS STRING) as json_payload")
        .select(from_json(col("json_payload"), TRANSACTION_SCHEMA).alias("data"))
        .select("data.*")
    )

    # 5. Filter out null or corrupted records
    valid_stream = parsed_stream.filter(
        col("transaction_id").isNotNull()
        & col("amount").isNotNull()
        & col("distance_from_home_km").isNotNull()
    )

    # 6. Windows-friendly Checkpoint Directory
    checkpoint_abs = os.path.abspath(args.checkpoint_dir).replace("\\", "/")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    # 7. Start Structured Streaming with foreachBatch
    batch_processor = make_batch_processor(model_bundle, args.output_csv)

    print(f"[*] Starting streaming query with trigger: {args.trigger_interval}")
    print(f"[*] Checkpoint location: {checkpoint_abs}")
    print(f"[*] Flagged transactions will be appended to: {args.output_csv}")
    print("[+] Waiting for incoming transactions... (Press Ctrl+C to terminate)")

    query = (
        valid_stream.writeStream
        .foreachBatch(batch_processor)
        .trigger(processingTime=args.trigger_interval)
        .option("checkpointLocation", checkpoint_abs)
        .start()
    )

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[*] Gracefully stopping Spark Structured Streaming query...")
        query.stop()
        spark.stop()
        print("[+] Streaming query stopped cleanly.")


if __name__ == "__main__":
    main()
