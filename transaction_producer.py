"""
transaction_producer.py - Live Transaction Event Generator for Apache Kafka

This script continuously generates realistic financial transactions (both normal
and deliberately anomalous) and publishes them as JSON messages to the Kafka
topic 'transactions'.

Features generated per event:
  - transaction_id (str)
  - timestamp (str - ISO 8601)
  - customer_id (str)
  - merchant_id (str)
  - merchant_category (str)
  - amount (float)
  - transaction_hour (int)
  - distance_from_home_km (float)
  - transaction_frequency_1h (int)
  - merchant_risk_score (float)
"""

import sys
import time
import json
import uuid
import random
import argparse
import signal
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Merchant categories and typical base risk profiles
MERCHANT_PROFILES = {
    "grocery": {"risk_range": (0.05, 0.20), "amount_range": (15.0, 150.0)},
    "dining": {"risk_range": (0.10, 0.30), "amount_range": (12.0, 120.0)},
    "gas_station": {"risk_range": (0.15, 0.40), "amount_range": (20.0, 85.0)},
    "online_retail": {"risk_range": (0.20, 0.50), "amount_range": (10.0, 300.0)},
    "travel_airline": {"risk_range": (0.30, 0.65), "amount_range": (150.0, 900.0)},
    "electronics": {"risk_range": (0.25, 0.60), "amount_range": (50.0, 800.0)},
    "luxury_goods": {"risk_range": (0.40, 0.75), "amount_range": (300.0, 2500.0)},
    "crypto_exchange": {"risk_range": (0.65, 0.95), "amount_range": (100.0, 3500.0)},
}

CATEGORIES = list(MERCHANT_PROFILES.keys())

# Customers with fixed baseline homes
CUSTOMER_POOL = [f"CUST-{1000 + i}" for i in range(25)]


def create_producer(bootstrap_servers: str, max_retries: int = 15, retry_interval: float = 2.0) -> KafkaProducer:
    """
    Connect to Kafka broker with automatic retry logic for container startup readiness.
    """
    print(f"[*] Connecting to Kafka cluster at: {bootstrap_servers}")
    for attempt in range(1, max_retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                request_timeout_ms=10000,
            )
            print(f"[+] Successfully connected to Kafka on attempt {attempt}/{max_retries}")
            return producer
        except NoBrokersAvailable:
            print(f"[!] Kafka not ready yet (attempt {attempt}/{max_retries}). Retrying in {retry_interval}s...")
            time.sleep(retry_interval)
        except Exception as e:
            print(f"[!] Connection error: {e}. Retrying in {retry_interval}s...")
            time.sleep(retry_interval)

    print(f"[-] ERROR: Failed to connect to Kafka at {bootstrap_servers} after {max_retries} attempts.")
    sys.exit(1)


def generate_transaction(anomaly_rate: float = 0.15) -> dict:
    """
    Generate a single transaction event.
    With probability anomaly_rate, generate an outlier transaction.
    Notice: No ground-truth label is leaked in the output schema!
    """
    is_anomaly = random.random() < anomaly_rate
    customer_id = random.choice(CUSTOMER_POOL)
    txn_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"
    now = datetime.utcnow()
    hour = now.hour

    if not is_anomaly:
        # Normal Everyday Pattern
        category = random.choices(
            CATEGORIES,
            weights=[0.30, 0.25, 0.15, 0.15, 0.05, 0.05, 0.03, 0.02],
            k=1
        )[0]
        profile = MERCHANT_PROFILES[category]
        amount = round(random.uniform(profile["amount_range"][0], profile["amount_range"][1]), 2)
        # Normal distance: close to billing address (median ~ 3-10 km)
        distance = round(abs(random.gauss(5.0, 4.0)) + 0.5, 2)
        # Normal frequency: 1 to 3 transactions in past hour
        freq = random.choice([1, 1, 1, 2, 2, 3])
        # Merchant risk score
        risk_score = round(random.uniform(profile["risk_range"][0], profile["risk_range"][1]), 3)
    else:
        # Deliberate Anomaly Patterns (simulating card theft, velocity burst, international fraud, or high-risk merchant)
        anomaly_scenario = random.choice(["high_amount", "far_distance", "velocity_spike", "compromised_night"])
        if anomaly_scenario == "high_amount":
            category = random.choice(["luxury_goods", "electronics", "online_retail"])
            amount = round(random.uniform(1500.0, 6000.0), 2)
            distance = round(random.uniform(5.0, 60.0), 2)
            freq = random.randint(1, 3)
            risk_score = round(random.uniform(0.35, 0.70), 3)
        elif anomaly_scenario == "far_distance":
            category = random.choice(CATEGORIES)
            amount = round(random.uniform(40.0, 850.0), 2)
            distance = round(random.uniform(500.0, 4500.0), 2)  # Far away / foreign country
            freq = random.randint(1, 3)
            risk_score = round(random.uniform(0.30, 0.75), 3)
        elif anomaly_scenario == "velocity_spike":
            category = random.choice(["online_retail", "crypto_exchange", "gas_station"])
            amount = round(random.uniform(25.0, 350.0), 2)
            distance = round(random.uniform(2.0, 35.0), 2)
            freq = random.randint(9, 22)  # High rapid frequency
            risk_score = round(random.uniform(0.50, 0.85), 3)
        else:  # compromised_night
            category = "crypto_exchange"
            amount = round(random.uniform(800.0, 3200.0), 2)
            distance = round(random.uniform(150.0, 1200.0), 2)
            freq = random.randint(4, 8)
            risk_score = round(random.uniform(0.75, 0.98), 3)
            hour = random.choice([2, 3, 4])  # unusual dead of night

    txn = {
        "transaction_id": txn_id,
        "timestamp": now.isoformat(),
        "customer_id": customer_id,
        "merchant_id": f"MERC-{random.randint(500, 999)}",
        "merchant_category": category,
        "amount": amount,
        "transaction_hour": hour,
        "distance_from_home_km": distance,
        "transaction_frequency_1h": freq,
        "merchant_risk_score": risk_score,
    }
    return txn


def main():
    parser = argparse.ArgumentParser(description="Live Transaction Event Producer for Kafka")
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--topic", type=str, default="transactions", help="Kafka topic to publish to")
    parser.add_argument("--interval", type=float, default=1.0, help="Delay in seconds between transactions")
    parser.add_argument("--max-events", type=int, default=None, help="Stop after publishing N events (None for continuous)")
    parser.add_argument("--anomaly-rate", type=float, default=0.18, help="Ratio of intentionally abnormal transactions")
    args = parser.parse_args()

    producer = create_producer(args.bootstrap_servers)

    # Graceful shutdown handler
    running = True

    def sig_handler(sig, frame):
        nonlocal running
        print("\n[*] Received shutdown signal. Closing producer...")
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("=" * 70)
    print("  LIVE TRANSACTION PRODUCER STARTED")
    print(f"  Target Topic  : {args.topic}")
    print(f"  Broker        : {args.bootstrap_servers}")
    print(f"  Interval      : {args.interval}s")
    print(f"  Anomaly Rate  : {args.anomaly_rate * 100:.1f}%")
    print(f"  Max Events    : {args.max_events or 'Continuous (Ctrl+C to stop)'}")
    print("=" * 70)

    count = 0
    try:
        while running:
            if args.max_events and count >= args.max_events:
                print(f"[+] Reached maximum events limit ({args.max_events}). Stopping.")
                break

            txn = generate_transaction(anomaly_rate=args.anomaly_rate)
            # Send to Kafka (keyed by customer_id for partition ordering)
            future = producer.send(args.topic, key=txn["customer_id"], value=txn)
            # Flush asynchronously or wait confirmation
            record_metadata = future.get(timeout=10.0)

            count += 1
            print(
                f"[{count:04d}] Sent {txn['transaction_id']} | "
                f"{txn['customer_id']} | "
                f"${txn['amount']:>7.2f} | "
                f"Dist: {txn['distance_from_home_km']:>6.1f}km | "
                f"Freq: {txn['transaction_frequency_1h']:>2d}/h | "
                f"Risk: {txn['merchant_risk_score']:0.2f} | "
                f"Partition: {record_metadata.partition} @ Offset: {record_metadata.offset}"
            )

            time.sleep(args.interval)

    except Exception as e:
        print(f"[!] Error while publishing: {e}")
    finally:
        print("[*] Flushing remaining messages and closing producer...")
        producer.flush(timeout=5.0)
        producer.close()
        print(f"[+] Producer stopped. Total events published: {count}")


if __name__ == "__main__":
    main()

