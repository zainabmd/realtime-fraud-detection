"""
Model health: feature/score drift (PSI) and champion-vs-challenger evaluation on matured labels.

Drift
-----
At training time ``build_reference_profile`` stores decile bins of every model input and of both
scores. Live, ``drift_report`` bins the recent window with the *same* edges and computes the
Population Stability Index per column::

    PSI = Σ (live% - ref%) · ln(live% / ref%)        < 0.10 stable · 0.10-0.25 watch · > 0.25 drift

Champion / challenger
---------------------
Labels arrive 5-90 minutes late, so only events older than ``LABEL_MATURITY_MIN`` are scored. The
challenger is compared to the champion **at the same analyst workload**: its threshold is set so it
flags the same share of traffic as the champion's review threshold. That isolates "catches more fraud"
from "flags more transactions". Promotion needs enough labelled fraud and a clear PR-AUC lead.

    python -m src.monitoring                 # one report -> data/monitoring/report.json
    python -m src.monitoring --watch 300     # refresh every 5 minutes (run_local starts this)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (LABEL_MATURITY_MIN, MODEL_DIR, MONITOR_DIR, PROMOTE_MIN_FRAUD, PROMOTE_MIN_GAIN, PSI_DRIFT,
                        PSI_WATCH, REVIEW_THRESHOLD, get_logger)
from src.features import FEATURE_COLUMNS, GRAPH_FEATURES
from src.utils import atomic_replace

log = get_logger("monitoring")

PROFILE_PATH = MODEL_DIR / "reference_profile.json"
REPORT_PATH = MONITOR_DIR / "report.json"
MONITORED = FEATURE_COLUMNS + GRAPH_FEATURES + ["risk_score", "challenger_score"]
# Time-of-day encodings shift in any window shorter than a day by design: reported, never alerted on.
SEASONAL = {"hour_sin", "hour_cos", "is_night"}
EPS = 1e-4
DECIMALS = 4      # the lake stores features rounded to 4 dp; bin edges and values must use the same grid,
                  # or point masses (e.g. the cold-start "30 days since last txn" sentinel) straddle an edge


# --------------------------------------------------------------------------- reference + PSI
def build_reference_profile(df: pd.DataFrame, columns: list[str] = MONITORED) -> dict:
    """Decile bin edges and the share of training rows in each bin, per column."""
    prof = {}
    for col in columns:
        if col not in df:
            continue
        x = np.round(df[col].to_numpy(dtype=float), DECIMALS)
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        edges = np.unique(np.round(np.quantile(x, np.linspace(0.1, 0.9, 9)), DECIMALS))
        counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
        prof[col] = {"edges": edges.tolist(), "expected": (counts / counts.sum()).round(6).tolist(),
                     "mean": float(x.mean())}
    return prof


def save_profile(profile: dict) -> None:
    tmp = PROFILE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile))
    atomic_replace(tmp, PROFILE_PATH)


def load_profile() -> dict:
    return json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else {}


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def band(value: float) -> str:
    return "drift" if value > PSI_DRIFT else "watch" if value > PSI_WATCH else "stable"


def drift_report(live: pd.DataFrame, profile: dict) -> pd.DataFrame:
    rows = []
    for col, ref in profile.items():
        if col not in live:
            continue
        x = np.round(live[col].to_numpy(dtype=float), DECIMALS)
        x = x[np.isfinite(x)]
        if len(x) < 200:            # too few rows for a stable histogram
            continue
        edges = np.round(np.asarray(ref["edges"]), DECIMALS)
        counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
        value = psi(ref["expected"], counts / counts.sum())
        rows.append({"feature": col, "psi": round(value, 4), "status": "seasonal" if col in SEASONAL else band(value),
                     "ref_mean": round(ref["mean"], 4), "live_mean": round(float(x.mean()), 4),
                     "kind": "score" if col in ("risk_score", "challenger_score") else
                             "graph" if col in GRAPH_FEATURES else "feature"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False) if rows else pd.DataFrame(
        columns=["feature", "psi", "status", "ref_mean", "live_mean", "kind"])


# --------------------------------------------------------------------------- champion / challenger
def _at_rate(y: np.ndarray, s: np.ndarray, rate: float) -> dict:
    """Precision / recall when flagging the top `rate` share of traffic by score `s`."""
    k = max(1, int(round(rate * len(s))))
    flag = np.zeros(len(s), dtype=bool)
    flag[np.argsort(-s, kind="stable")[:k]] = True
    tp = int((flag & y).sum())
    return {"precision": tp / k, "recall": tp / max(int(y.sum()), 1), "flagged": k}


def compare_models(df: pd.DataFrame, review: float = REVIEW_THRESHOLD) -> dict:
    """Champion vs challenger on labelled rows (`is_fraud`, `risk_score`, `challenger_score`, `fraud_type`)."""
    df = df.dropna(subset=["challenger_score"])
    y = df["is_fraud"].to_numpy(dtype=bool)
    n_fraud = int(y.sum())
    out: dict = {"n": int(len(df)), "n_fraud": n_fraud}
    if n_fraud == 0 or n_fraud == len(df):
        out["verdict"] = "insufficient_data"
        out["reason"] = "no labelled fraud (or no legit traffic) in the matured window yet"
        return out
    champ, chall = df["risk_score"].to_numpy(), df["challenger_score"].to_numpy()
    rate = float((champ >= review).mean()) or 1e-3
    out["flag_rate"] = rate
    for name, s in (("champion", champ), ("challenger", chall)):
        out[name] = {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
                     **_at_rate(y, s, rate)}
    by_type = {}
    for ftype, g in df[y].groupby("fraud_type"):
        idx = g.index
        res = {}
        for name, col in (("champion", "risk_score"), ("challenger", "challenger_score")):
            k = max(1, int(round(rate * len(df))))
            cut = np.sort(df[col].to_numpy())[::-1][k - 1]
            res[name] = float((df.loc[idx, col] >= cut).mean())
        by_type[ftype] = {**res, "n": int(len(g))}
    out["recall_by_type"] = by_type
    gain = out["challenger"]["pr_auc"] - out["champion"]["pr_auc"]
    out["pr_auc_gain"] = gain
    if n_fraud < PROMOTE_MIN_FRAUD:
        out["verdict"], out["reason"] = "insufficient_data", f"{n_fraud} labelled frauds < {PROMOTE_MIN_FRAUD} required"
    elif gain >= PROMOTE_MIN_GAIN and out["challenger"]["recall"] >= out["champion"]["recall"]:
        out["verdict"] = "promote"
        out["reason"] = (f"PR-AUC +{gain:.3f} and recall {out['challenger']['recall']:.1%} vs "
                         f"{out['champion']['recall']:.1%} at the same {rate:.2%} flag rate")
    elif gain <= -PROMOTE_MIN_GAIN:
        out["verdict"], out["reason"] = "retire_challenger", f"challenger PR-AUC {gain:+.3f} behind the champion"
    else:
        out["verdict"], out["reason"] = "keep_champion", f"PR-AUC difference {gain:+.3f} within ±{PROMOTE_MIN_GAIN}"
    return out


# --------------------------------------------------------------------------- live report
def live_frame(hours: float = 6.0) -> pd.DataFrame:
    """Recent scored events with their (possibly missing) labels, newest window anchored on the lake."""
    from src.storage import query   # local import: keeps this module importable without a lake
    cols = ", ".join(f's."{c}"' for c in FEATURE_COLUMNS)
    graph = ", ".join(f'COALESCE(s."{c}", 0) AS "{c}"' for c in GRAPH_FEATURES)
    return query(f"""
        WITH w AS (SELECT * FROM scored WHERE event_time >= (SELECT max(event_time) FROM scored) - INTERVAL {int(hours * 60)} MINUTE)
        SELECT s.txn_id, s.event_time, {cols}, {graph}, s.risk_score, s.challenger_score, l.fraud_type
        FROM w s LEFT JOIN labels l USING (txn_id)""")


def build_report(hours: float = 6.0, drift_hours: float = 3.0) -> dict:
    """Drift on the last `drift_hours` (recent enough to react), model comparison on the last `hours`
    (long enough for labels to mature)."""
    try:
        df = live_frame(hours)
    except Exception as exc:  # noqa: BLE001 – e.g. no labels table yet
        log.debug("live frame failed: %s", exc)
        df = pd.DataFrame()
    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "window_hours": hours,
                    "drift_window_hours": drift_hours}
    if df.empty:
        report["status"] = "no_data"
        return report
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["is_fraud"] = df["fraud_type"].notna()
    profile = load_profile()
    recent = df[df["event_time"] >= df["event_time"].max() - pd.Timedelta(hours=drift_hours)]
    drift = drift_report(recent, profile) if profile else pd.DataFrame()
    report["drift"] = drift.to_dict("records")
    alerting = drift[drift["status"] != "seasonal"] if not drift.empty else drift
    report["drift_summary"] = ({"drift": int((alerting["status"] == "drift").sum()),
                                "watch": int((alerting["status"] == "watch").sum()),
                                "max_psi": float(alerting["psi"].max()),
                                "worst": alerting.iloc[0]["feature"]} if not alerting.empty else {})
    mature = df[df["event_time"] <= df["event_time"].max() - pd.Timedelta(minutes=LABEL_MATURITY_MIN)]
    report["matured_rows"] = int(len(mature))
    report["comparison"] = compare_models(mature.reset_index(drop=True)) if len(mature) else {
        "verdict": "insufficient_data", "reason": "no events older than the label delay yet"}
    report["status"] = "ok"
    return report


def write_report(report: dict) -> None:
    tmp = REPORT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o)))
    atomic_replace(tmp, REPORT_PATH)


def read_report() -> dict:
    return json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drift + champion/challenger monitoring")
    ap.add_argument("--hours", type=float, default=6.0, help="look-back for the champion/challenger comparison")
    ap.add_argument("--drift-hours", type=float, default=3.0, help="look-back for drift (PSI)")
    ap.add_argument("--watch", type=float, default=None, metavar="SECONDS", help="refresh continuously")
    args = ap.parse_args(argv)
    while True:
        try:
            r = build_report(args.hours, args.drift_hours)
            write_report(r)
            c = r.get("comparison", {})
            log.info("report: status=%s drift=%s verdict=%s (%s)", r.get("status"), r.get("drift_summary", {}),
                     c.get("verdict"), c.get("reason"))
        except Exception as exc:  # noqa: BLE001 – a monitoring failure must never take the stack down
            log.exception("monitoring run failed: %s", exc)
            if not args.watch:
                return 1
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
