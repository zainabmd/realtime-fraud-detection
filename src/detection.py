"""
Hybrid anomaly detection: statistical rules + Isolation Forest -> one risk score.

    risk = min(100, IF points (0-60) + min(rule points, 70))

* **Rules** encode known attack signatures with explicit thresholds (velocity,
  impossible travel, z-score spikes, ATO pattern...). They are precise,
  explainable and catch known fraud on day one.
* **Isolation Forest** is unsupervised: it isolates rare combinations of the
  22 behavioural features without needing labels, catching *novel* patterns.
  Its raw score is mapped to a percentile of the training distribution, and
  only the most anomalous tail (top 15%) earns points.

Decisions: risk >= DECLINE_THRESHOLD -> decline, >= REVIEW_THRESHOLD -> review.
Every score ships with the triggered rules and the top deviating features, so
analysts see *why* a transaction was flagged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.config import DECLINE_THRESHOLD, IF_WEIGHT, MODEL_DIR, REVIEW_THRESHOLD, RULES_CAP, get_logger
from src.features import FEATURE_COLUMNS, as_matrix
from src.utils import atomic_replace

log = get_logger("detection")

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"


@dataclass(frozen=True)
class Rule:
    code: str
    description: str
    points: float
    predicate: Callable[[pd.DataFrame], pd.Series]


RULES: list[Rule] = [
    Rule("VELOCITY_1M", "≥4 transactions on the card in the last minute", 25,
         lambda f: f["txn_count_1m"] >= 4),
    Rule("SMALL_TXN_BURST", "≥3 sub-$5 charges in 10 min (card testing)", 30,
         lambda f: f["small_txn_count_10m"] >= 3),
    Rule("IMPOSSIBLE_TRAVEL", "Implied travel speed > 900 km/h over > 500 km", 45,
         lambda f: (f["speed_kmh"] > 900) & (f["km_from_last"] > 500)),
    Rule("AMOUNT_SPIKE", "Amount z-score > 3.5 vs card history and > $200", 25,
         lambda f: (f["amount_zscore"] > 3.5) & (np.expm1(f["amount_usd_log"]) > 200)),
    Rule("ATO_PATTERN", "New device + foreign IP on a card-not-present payment", 30,
         lambda f: (f["new_device"] == 1) & (f["ip_mismatch"] == 1)),
    Rule("MERCHANT_SPRAY", "≥5 distinct merchants in 10 min", 25,
         lambda f: f["distinct_merchants_10m"] >= 5),
    Rule("HIGH_RISK_MCC", "High-risk merchant category at > 3x usual spend", 15,
         lambda f: (f["mcc_risk"] >= 0.6) & (f["amount_to_avg_ratio"] > 3)),
    Rule("NIGHT_CASH", "ATM withdrawal at night, > 2x usual spend", 10,
         lambda f: (f["is_atm"] == 1) & (f["is_night"] == 1) & (f["amount_to_avg_ratio"] > 2)),
]
RULE_INDEX = {r.code: r for r in RULES}


def evaluate_rules(feats: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Vectorised rule evaluation -> (points per row, list of triggered codes per row)."""
    hits = np.column_stack([r.predicate(feats).to_numpy(dtype=bool) for r in RULES])
    pts = hits @ np.array([r.points for r in RULES])
    codes = [[RULES[j].code for j in np.flatnonzero(row)] for row in hits]
    return pts, codes


@dataclass
class ModelBundle:
    """Everything needed to score, versioned and persisted as one artefact."""

    model: object                         # fitted IsolationForest
    version: str
    train_scores: np.ndarray              # sorted anomaly scores from training (for percentiles)
    feature_center: np.ndarray            # median per feature
    feature_scale: np.ndarray             # IQR per feature (robust scale)
    importances: dict[str, float] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def save(self, path: Path = MODEL_PATH) -> None:
        tmp = path.with_suffix(".tmp")
        joblib.dump(self, tmp)
        atomic_replace(tmp, path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "ModelBundle":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found – run `python -m src.train_model` first")
        return joblib.load(path)


class Scorer:
    def __init__(self, bundle: ModelBundle | None = None, review: float = REVIEW_THRESHOLD,
                 decline: float = DECLINE_THRESHOLD):
        self.bundle = bundle or ModelBundle.load()
        self.review, self.decline = review, decline

    def if_percentile(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = -self.bundle.model.score_samples(X)          # higher = more anomalous
        pct = np.searchsorted(self.bundle.train_scores, raw) / len(self.bundle.train_scores)
        return raw, pct

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = as_matrix(feats)
        if not np.isfinite(X).all():  # defensive: never let NaN/inf reach the model
            X = np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0)
        raw, pct = self.if_percentile(X)
        if_pts = IF_WEIGHT * np.clip((pct - 0.85) / 0.15, 0, 1) ** 1.5
        rule_pts, codes = evaluate_rules(feats)
        risk = np.minimum(100.0, if_pts + np.minimum(rule_pts, RULES_CAP))
        decision = np.where(risk >= self.decline, "decline", np.where(risk >= self.review, "review", "approve"))

        # Explanations: the 3 features deviating most from the training median (robust z).
        z = (X - self.bundle.feature_center) / self.bundle.feature_scale
        top_idx = np.argsort(-np.abs(z), axis=1)[:, :3]
        top = [json.dumps({FEATURE_COLUMNS[j]: round(float(z[i, j]), 2) for j in row}) for i, row in enumerate(top_idx)]

        return pd.DataFrame({
            "if_anomaly": raw.round(5), "if_percentile": pct.round(5), "if_points": if_pts.round(2),
            "rule_points": rule_pts, "rules_triggered": ["|".join(c) for c in codes],
            "risk_score": risk.round(2), "decision": decision, "top_features": top,
            "model_version": self.bundle.version,
        })
