"""
Dashboard data access: DuckDB over the Parquet lake, cached per refresh tick.

Design
------
* The time window is anchored on the **latest event in the lake**, not the
  wall clock: the dashboard is equally meaningful while streaming live and
  when replaying a backfill.
* Hive partition pruning (``dt``) plus an ``event_time`` predicate keep scans
  proportional to the window, not the lake.
* One filtered window per (filters, tick) is cached (``ttl`` a little below the
  refresh interval); every chart on a page aggregates that frame, so a
  refresh costs one lake scan.
* Labels arrive late (chargebacks): "confirmed fraud" is a *lower bound* for
  recent minutes. Pages say so where it matters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from src.config import DLQ_DIR, LABELS_DIR, METRICS_DIR, MODEL_DIR, SCORED_DIR
from src.storage import lake_summary

WINDOWS = {"Last 15 min": 15, "Last 1 hour": 60, "Last 6 hours": 360, "Last 24 hours": 1440, "All data": None}

WINDOW_COLS = [
    "txn_id", "event_time", "card_id", "merchant_name", "merchant_category", "txn_type", "channel", "amount",
    "currency", "amount_usd", "country", "city", "lat", "lon", "home_country", "home_lat", "home_lon", "device_id",
    "ip_country", "amount_zscore", "amount_to_avg_ratio", "txn_count_1m", "txn_count_10m", "distinct_merchants_10m",
    "small_txn_count_10m", "speed_kmh", "km_from_home", "new_device", "geo_mismatch", "ip_mismatch", "is_night",
    "mcc_risk", "if_percentile", "if_points", "rule_points", "rules_triggered", "risk_score", "decision",
    "top_features", "model_version", "processed_at", "pipeline_latency_ms", "partition", "offset",
]
# Added in 2026 (challenger shadow score + graph features); older lake files simply return NULL for them.
from src.features import FEATURE_COLUMNS, GRAPH_FEATURES  # noqa: E402

OPTIONAL_COLS = ["challenger_score"] + [c for c in FEATURE_COLUMNS + GRAPH_FEATURES if c not in WINDOW_COLS]


class LakeEmpty(RuntimeError):
    """No scored data yet – the consumer hasn't flushed anything."""


@dataclass(frozen=True)
class Filters:
    window: str = "Last 1 hour"
    threshold: float = 50.0
    txn_types: tuple[str, ...] = ()
    currencies: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    min_amount: float = 0.0


def _glob(base) -> str:
    return (base / "**" / "*.parquet").as_posix()


def _has(base) -> bool:
    return any(base.glob("dt=*/hr=*/*.parquet"))


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("SET threads TO 4")
    return con


def _run(sql: str, params: list | None = None, retries: int = 3) -> pd.DataFrame:
    """Execute with retries – compaction may delete a file between glob and read."""
    last = None
    for _ in range(retries):
        con = _con()
        try:
            return con.execute(sql, params or []).df()
        except (duckdb.IOException, duckdb.InvalidInputException) as exc:
            last = exc
        finally:
            con.close()
    raise last  # type: ignore[misc]


@st.cache_data(ttl=2, show_spinner=False)
def latest_state() -> dict:
    """Newest event time and last write time (cheap: Parquet min/max statistics)."""
    if not _has(SCORED_DIR):
        raise LakeEmpty("No scored transactions in the lake yet.")
    df = _run(f"SELECT max(event_time) AS max_et, min(event_time) AS min_et, max(processed_at) AS last_write, "
              f"any_value(model_version) AS model FROM read_parquet('{_glob(SCORED_DIR)}', hive_partitioning=true)")
    r = df.iloc[0]
    return {"max_et": utc(r["max_et"]), "min_et": utc(r["min_et"]), "last_write": utc(r["last_write"]), "model": r["model"]}


def utc(ts) -> pd.Timestamp:
    """DuckDB returns TIMESTAMPTZ in the local zone; normalise everything to UTC."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def window_bounds(f: Filters) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = latest_state()
    minutes = WINDOWS[f.window]
    end = s["max_et"]
    start = s["min_et"] if minutes is None else end - timedelta(minutes=minutes)
    return start, end


@st.cache_data(ttl=60, show_spinner=False)
def _scored_columns() -> set[str]:
    df = _run(f"DESCRIBE SELECT * FROM read_parquet('{_glob(SCORED_DIR)}', hive_partitioning=true, union_by_name=true)")
    return set(df["column_name"])


@st.cache_data(ttl=20, show_spinner=False)
def rings_snapshot() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.graph import load_rings
    return load_rings()


@st.cache_data(ttl=20, show_spinner=False)
def monitoring_report() -> dict:
    from src.monitoring import read_report
    return read_report()


@st.cache_data(ttl=300, show_spinner=False)
def reference_profile() -> dict:
    from src.monitoring import load_profile
    return load_profile()


@st.cache_data(ttl=3, max_entries=24, show_spinner=False)
def load_window(f: Filters) -> pd.DataFrame:
    """Filtered, de-duplicated scored transactions in the window, joined to (delayed) labels."""
    start, end = window_bounds(f)
    where = ["dt >= ?", "event_time >= ?", "event_time <= ?"]
    params: list = [start.strftime("%Y-%m-%d"), start.to_pydatetime(), end.to_pydatetime()]
    for col, vals in (("txn_type", f.txn_types), ("currency", f.currencies), ("channel", f.channels),
                      ("merchant_category", f.categories), ("country", f.countries), ("decision", f.decisions)):
        if vals:
            where.append(f"{col} IN ({','.join('?' * len(vals))})")
            params += list(vals)
    if f.min_amount > 0:
        where.append("amount_usd >= ?")
        params.append(f.min_amount)
    cols = ", ".join(f's."{c}"' for c in WINDOW_COLS + [c for c in OPTIONAL_COLS if c in _scored_columns()])
    label_join = (f"LEFT JOIN (SELECT txn_id, any_value(fraud_type) AS fraud_type FROM "
                  f"read_parquet('{_glob(LABELS_DIR)}', hive_partitioning=true) GROUP BY txn_id) l USING (txn_id)"
                  if _has(LABELS_DIR) else "")
    label_cols = ", l.fraud_type" if label_join else ", NULL::VARCHAR AS fraud_type"
    sql = f"""
        WITH s AS (
            SELECT * FROM read_parquet('{_glob(SCORED_DIR)}', hive_partitioning=true, union_by_name=true)
            WHERE {' AND '.join(where)}
            QUALIFY row_number() OVER (PARTITION BY txn_id ORDER BY processed_at) = 1)
        SELECT {cols}{label_cols} FROM s {label_join}"""
    df = _run(sql, params)
    if df.empty:
        return df
    for c in ("event_time", "processed_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["is_fraud"] = df["fraud_type"].notna()
    df["flagged"] = df["risk_score"] >= f.threshold
    for c in ("merchant_category", "txn_type", "channel", "currency", "country", "city", "decision", "home_country"):
        df[c] = df[c].astype("category")
    return df


@st.cache_data(ttl=3, show_spinner=False)
def load_metrics(minutes: int | None = 360) -> pd.DataFrame:
    if not _has(METRICS_DIR):
        return pd.DataFrame()
    where = "" if minutes is None else f"WHERE event_time >= now() - INTERVAL {int(minutes)} MINUTE"
    df = _run(f"SELECT * EXCLUDE (dt, hr) FROM read_parquet('{_glob(METRICS_DIR)}', hive_partitioning=true, "
              f"union_by_name=true) {where} ORDER BY event_time")
    if df.empty and minutes is not None:  # e.g. backfill finished hours ago: show the last batches anyway
        df = _run(f"SELECT * EXCLUDE (dt, hr) FROM read_parquet('{_glob(METRICS_DIR)}', hive_partitioning=true, "
                  f"union_by_name=true) ORDER BY event_time DESC LIMIT 500").sort_values("event_time")
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


@st.cache_data(ttl=5, show_spinner=False)
def load_dlq(limit: int = 500) -> pd.DataFrame:
    if not _has(DLQ_DIR):
        return pd.DataFrame(columns=["received_at", "partition", "offset", "error", "raw"])
    return _run(f"SELECT received_at, partition, \"offset\", error, raw FROM read_parquet('{_glob(DLQ_DIR)}', "
                f"hive_partitioning=true) ORDER BY received_at DESC LIMIT {int(limit)}")


@st.cache_data(ttl=5, show_spinner=False)
def dlq_counts() -> pd.DataFrame:
    if not _has(DLQ_DIR):
        return pd.DataFrame(columns=["error_type", "n"])
    return _run(f"SELECT split_part(error, ':', 1) AS error_type, count(*) AS n FROM "
                f"read_parquet('{_glob(DLQ_DIR)}', hive_partitioning=true) GROUP BY 1 ORDER BY 2 DESC")


@st.cache_data(ttl=3, show_spinner=False)
def card_history(card_id: str, limit: int = 200) -> pd.DataFrame:
    if not card_id or not _has(SCORED_DIR):
        return pd.DataFrame()
    df = _run(f"""
        SELECT * FROM read_parquet('{_glob(SCORED_DIR)}', hive_partitioning=true, union_by_name=true)
        WHERE card_id = ? QUALIFY row_number() OVER (PARTITION BY txn_id ORDER BY processed_at) = 1
        ORDER BY event_time DESC LIMIT {int(limit)}""", [card_id])
    if not df.empty and _has(LABELS_DIR):
        lb = _run(f"SELECT DISTINCT txn_id, fraud_type FROM read_parquet('{_glob(LABELS_DIR)}', hive_partitioning=true) "
                  f"WHERE card_id = ?", [card_id])
        df = df.merge(lb, on="txn_id", how="left")
    elif not df.empty:
        df["fraud_type"] = None
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def model_card() -> dict:
    path = MODEL_DIR / "model_card.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(ttl=60, show_spinner=False)
def feature_importances() -> pd.DataFrame:
    card = model_card()
    try:
        from src.detection import ModelBundle
        imp = ModelBundle.load().importances
    except Exception:  # noqa: BLE001 – fall back to the JSON card
        imp = card.get("top_importances", {})
    return pd.DataFrame({"feature": list(imp), "importance": list(imp.values())})


@st.cache_data(ttl=10, show_spinner=False)
def lake_stats() -> dict:
    return lake_summary()


def time_bucket(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Pick a bucket so a time-series has ~60-120 points."""
    minutes = max((end - start).total_seconds() / 60, 1)
    for label, m in (("10s", 1 / 6), ("30s", 0.5), ("1min", 1), ("5min", 5), ("15min", 15), ("1h", 60)):
        if minutes / m <= 140:
            return label
    return "1h"


def split_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the pipe-separated rules column into one row per (txn, rule)."""
    r = df.loc[df["rules_triggered"].fillna("") != "", ["txn_id", "rules_triggered", "is_fraud", "risk_score"]]
    if r.empty:
        return pd.DataFrame(columns=["txn_id", "rule", "is_fraud", "risk_score"])
    r = r.assign(rule=r["rules_triggered"].str.split("|")).explode("rule")
    return r.drop(columns="rules_triggered")


def threshold_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall / F1 / flag-rate for every integer threshold 0-100 (vs confirmed labels)."""
    y = df["is_fraud"].to_numpy()
    s = df["risk_score"].to_numpy()
    rows = []
    for t in range(0, 101):
        pred = s >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * p * r / (p + r) if p == p and r == r and (p + r) else np.nan
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1, "flag_rate": pred.mean() if len(s) else np.nan})
    return pd.DataFrame(rows)
