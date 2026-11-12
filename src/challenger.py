"""
Challenger model: supervised gradient boosting on behaviour + entity-graph features.

The champion (Isolation Forest + rules, ``src/detection.py``) is unsupervised and needs no labels,
which is why it ran first. Once chargebacks have accumulated, a supervised model can learn what fraud
actually looks like, and the graph features (cards per device, devices per card) let it see coordinated
rings that no single-card rule can.

Deployment pattern: the challenger is **shadow-scored** on every event by the consumer (column
``challenger_score``) but never drives decisions. ``src/monitoring.py`` compares champion and challenger
on matured labels and recommends promotion only when the challenger wins on the same analyst workload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import MODEL_DIR, get_logger
from src.features import FEATURE_COLUMNS, GRAPH_FEATURES
from src.utils import atomic_replace

log = get_logger("challenger")

CHALLENGER_PATH = MODEL_DIR / "challenger.joblib"
CHALLENGER_FEATURES = FEATURE_COLUMNS + GRAPH_FEATURES


@dataclass
class ChallengerBundle:
    model: HistGradientBoostingClassifier
    version: str
    features: list[str] = field(default_factory=lambda: list(CHALLENGER_FEATURES))
    metrics: dict = field(default_factory=dict)

    def save(self, path: Path = CHALLENGER_PATH) -> None:
        tmp = path.with_suffix(".tmp")
        joblib.dump(self, tmp)
        atomic_replace(tmp, path)

    @staticmethod
    def load(path: Path = CHALLENGER_PATH) -> "ChallengerBundle | None":
        """None when no challenger has been trained: shadow scoring is optional, never a startup failure."""
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as exc:  # noqa: BLE001 – a broken challenger must not stop the champion
            log.warning("Challenger at %s could not be loaded (%s); shadow scoring disabled", path, exc)
            return None


def fit_challenger(train_df: pd.DataFrame, seed: int = 7) -> ChallengerBundle:
    """Fit on labelled history (column ``is_fraud``). Fraud is ~2% of events, so classes are re-weighted."""
    X = train_df[CHALLENGER_FEATURES].to_numpy(dtype=np.float64)
    y = train_df["is_fraud"].to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
        class_weight={0: 1.0, 1: 8.0}, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
        random_state=seed)
    model.fit(X, y)
    version = "hgb-graph-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    log.info("Challenger %s fitted on %s rows (%d boosting rounds)", version, f"{len(X):,}", model.n_iter_)
    return ChallengerBundle(model=model, version=version)


class ChallengerScorer:
    def __init__(self, bundle: ChallengerBundle):
        self.bundle = bundle

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        """Fraud probability as a 0-100 score (comparable in range to the champion's risk score)."""
        X = feats[self.bundle.features].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0)
        return (100.0 * self.bundle.model.predict_proba(X)[:, 1]).round(2)
