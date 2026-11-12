"""
Central configuration for the streaming fraud-detection platform.

Every tunable lives here and can be overridden from the environment, so the
same code runs on a laptop (SQLite broker + local Parquet lake) and in
production (Kafka + object storage) without edits.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", PROJECT_ROOT / "data"))
LAKE_DIR = DATA_DIR / "lake"                 # Parquet lakehouse (hive-partitioned)
SCORED_DIR = LAKE_DIR / "scored"             # one row per scored transaction
LABELS_DIR = LAKE_DIR / "labels"             # delayed chargeback / confirmed-fraud labels
METRICS_DIR = LAKE_DIR / "metrics"           # per-micro-batch pipeline telemetry
DLQ_DIR = LAKE_DIR / "dlq"                   # dead-letter queue (schema-invalid events)
GRAPH_DIR = LAKE_DIR / "graph"               # fraud-ring snapshots (src/graph.py)
MONITOR_DIR = DATA_DIR / "monitoring"        # drift + champion/challenger reports (src/monitoring.py)
REFERENCE_DIR = DATA_DIR / "reference"       # card / merchant reference data
STATE_DIR = DATA_DIR / "state"               # feature-store checkpoints
MODEL_DIR = PROJECT_ROOT / "models"
BROKER_DB = DATA_DIR / "broker.sqlite"       # local broker log

for _d in (SCORED_DIR, LABELS_DIR, METRICS_DIR, DLQ_DIR, GRAPH_DIR, MONITOR_DIR, REFERENCE_DIR, STATE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Broker
# --------------------------------------------------------------------------- #
# "sqlite" (default, zero-dependency, multi-process safe) or "kafka".
BROKER_BACKEND = os.getenv("BROKER_BACKEND", "sqlite").lower()
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_TRANSACTIONS = os.getenv("TOPIC_TRANSACTIONS", "transactions.v1")
TOPIC_LABELS = os.getenv("TOPIC_LABELS", "fraud-labels.v1")
TOPIC_ALERTS = os.getenv("TOPIC_ALERTS", "fraud-alerts.v1")
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", "4"))
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "fraud-scorer")

# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
SIM_SEED = int(os.getenv("SIM_SEED", "2024"))
SIM_N_CARDS = int(os.getenv("SIM_N_CARDS", "100000"))     # ~20 txns/card/day at 25 TPS (realistic velocity)
SIM_N_MERCHANTS = int(os.getenv("SIM_N_MERCHANTS", "1500"))
SIM_FRAUD_RATE = float(os.getenv("SIM_FRAUD_RATE", "0.004"))     # P(fraud episode starts) per legit event -> ~2% fraud events
SIM_MALFORMED_RATE = float(os.getenv("SIM_MALFORMED_RATE", "0.001"))
SIM_LATE_EVENT_RATE = float(os.getenv("SIM_LATE_EVENT_RATE", "0.01"))
SIM_LABEL_DELAY_MIN = (5, 90)                                     # chargeback arrives 5-90 min later
SIM_RING_RATE = float(os.getenv("SIM_RING_RATE", "0.0004"))      # P(a fraud ring compromises the card) per legit event
SIM_HOUSEHOLD_RATE = float(os.getenv("SIM_HOUSEHOLD_RATE", "0.02"))  # legit CNP payment from a shared household device

# --------------------------------------------------------------------------- #
# Consumer / micro-batching
# --------------------------------------------------------------------------- #
BATCH_MAX_EVENTS = int(os.getenv("BATCH_MAX_EVENTS", "2000"))     # poll size
FLUSH_MAX_ROWS = int(os.getenv("FLUSH_MAX_ROWS", "20000"))        # Parquet file size target
FLUSH_MAX_SECONDS = float(os.getenv("FLUSH_MAX_SECONDS", "5"))    # max staleness of the lake
CHECKPOINT_EVERY_S = float(os.getenv("CHECKPOINT_EVERY_S", "30"))
COMPACT_EVERY_S = float(os.getenv("COMPACT_EVERY_S", "300"))
COMPACT_MIN_FILES = int(os.getenv("COMPACT_MIN_FILES", "12"))

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "50"))     # risk score -> manual review
DECLINE_THRESHOLD = float(os.getenv("DECLINE_THRESHOLD", "80"))   # risk score -> auto-decline
IF_WEIGHT = 60.0                                                  # max points from the Isolation Forest
RULES_CAP = 70.0                                                  # max points from rules
LABEL_MATURITY_MIN = 90                                           # chargebacks arrive within 90 min

# --------------------------------------------------------------------------- #
# Monitoring, graph and copilot (2026)
# --------------------------------------------------------------------------- #
PSI_WATCH, PSI_DRIFT = 0.10, 0.25                                 # population-stability bands
PROMOTE_MIN_FRAUD = int(os.getenv("PROMOTE_MIN_FRAUD", "50"))     # labelled frauds needed before a verdict
PROMOTE_MIN_GAIN = float(os.getenv("PROMOTE_MIN_GAIN", "0.02"))   # challenger PR-AUC lead required
RING_MIN_CARDS = int(os.getenv("RING_MIN_CARDS", "3"))            # smallest device-linked cluster called a ring
GRAPH_WINDOW_H = float(os.getenv("GRAPH_WINDOW_H", "24"))
COPILOT_MODEL = os.getenv("COPILOT_MODEL", "claude-opus-5-5")
COPILOT_EFFORT = os.getenv("COPILOT_EFFORT", "medium")

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")

LOG_LEVEL = os.getenv("FRAUD_LOG_LEVEL", "INFO")


def get_logger(name: str) -> logging.Logger:
    """Module logger with a consistent, grep-friendly format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    return logger
