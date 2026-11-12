"""
Offline training & evaluation of the anomaly model.

1. Simulate an *independent* traffic stream (different traffic seed than the
   live producer) over the same card population.
2. Compute features with the production FeatureEngine (no train/serve skew).
3. Time-based split: first 70% of events -> train, last 30% -> holdout.
4. Fit an Isolation Forest **without labels** (it never sees `is_fraud`).
5. Use labels only for evaluation: ROC-AUC / PR-AUC for IF alone, rules alone
   and the blended risk score; precision/recall at the review & decline
   thresholds; recall per fraud type.
6. Global feature importance via a surrogate random forest fitted to the
   IF anomaly score (IF has no native importances).
7. Persist a versioned ModelBundle + a human-readable model card.
8. (2026) Fit the supervised **challenger** (gradient boosting on behaviour + graph features) on the
   same training window, compare it with the champion on the holdout at equal workload, and store the
   **reference profile** (decile bins of every input and score) that drift monitoring compares against.

    python -m src.train_model --hours 6 --tps 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.challenger import ChallengerScorer, fit_challenger
from src.config import DECLINE_THRESHOLD, MODEL_DIR, REVIEW_THRESHOLD, get_logger
from src.detection import RULES, ModelBundle, Scorer, evaluate_rules
from src.features import FEATURE_COLUMNS, GRAPH_FEATURES, FeatureEngine, as_matrix
from src.monitoring import build_reference_profile, compare_models, save_profile
from src.schemas import Transaction
from src.simulator import TransactionSimulator

log = get_logger("train_model")


def build_dataset(hours: float, tps: float, traffic_seed: int, warmup_hours: float = 2.0) -> pd.DataFrame:
    """Simulated, labelled feature rows. The first `warmup_hours` only build feature state and are dropped:
    a live stream always has warm per-card state, so training on a cold start would skew every
    history-based feature (and the drift reference) towards "first time we see this card"."""
    sim = TransactionSimulator(traffic_seed=traffic_seed, malformed_rate=0.0)
    sim.write_reference()
    cards, _ = sim.reference_frames()
    engine = FeatureEngine(cards)
    start = time.time() - (hours + warmup_hours) * 3600 - 86400 * 2  # well before the live stream's window
    events, labels = [], []
    t = start
    while t < start + (hours + warmup_hours) * 3600:
        for ev, lb in sim.events_between(t, t + 1800, tps):
            events.append(ev)
            labels.append(lb["fraud_type"] if lb else "legit")
        t += 1800
    log.info("Simulated %s events (%.2f%% fraud)", f"{len(events):,}", 100 * np.mean([lb != "legit" for lb in labels]))
    # Validate exactly like the consumer does, then compute features in arrival order.
    ts = [Transaction.model_validate(ev).event_time.timestamp() for ev in events]
    feats = engine.compute_batch(events, ts)
    feats["fraud_type"] = labels
    feats["is_fraud"] = (feats["fraud_type"] != "legit").astype(int)
    feats["event_ts"] = ts
    feats = feats[feats["event_ts"] >= start + warmup_hours * 3600].reset_index(drop=True)
    log.info("Dropped the %.0f h warm-up; %s rows remain for training/evaluation", warmup_hours, f"{len(feats):,}")
    return feats


def train(df: pd.DataFrame, seed: int = 7):
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    X_train = as_matrix(train_df)

    t0 = time.perf_counter()
    model = IsolationForest(n_estimators=300, max_samples=4096, contamination="auto", random_state=seed, n_jobs=-1)
    model.fit(X_train)
    log.info("Isolation Forest fitted on %s rows in %.1fs", f"{len(train_df):,}", time.perf_counter() - t0)

    train_raw = np.sort(-model.score_samples(X_train))
    q = np.quantile(X_train, [0.25, 0.5, 0.75], axis=0)
    scale = np.where((q[2] - q[0]) > 1e-9, q[2] - q[0], X_train.std(axis=0) + 1e-6)

    # Surrogate importance: which features drive the anomaly score?
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(40_000, len(X_train)), replace=False)
    sur = RandomForestRegressor(n_estimators=120, max_depth=10, n_jobs=-1, random_state=seed)
    sur.fit(X_train[idx], -model.score_samples(X_train[idx]))
    importances = dict(sorted(zip(FEATURE_COLUMNS, sur.feature_importances_.round(5)), key=lambda kv: -kv[1]))

    version = "iforest-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    bundle = ModelBundle(model=model, version=version, train_scores=train_raw, feature_center=q[1],
                         feature_scale=scale, importances={k: float(v) for k, v in importances.items()})

    # ---- Evaluation on the time-based holdout ----
    scored = Scorer(bundle).score(test_df.reset_index(drop=True))
    y = test_df["is_fraud"].to_numpy()
    rule_pts, _ = evaluate_rules(test_df)
    metrics: dict = {
        "n_train": int(len(train_df)), "n_test": int(len(test_df)), "test_fraud_rate": float(y.mean()),
        "roc_auc": {"isolation_forest": float(roc_auc_score(y, scored["if_anomaly"])),
                    "rules": float(roc_auc_score(y, rule_pts)),
                    "blended_risk": float(roc_auc_score(y, scored["risk_score"]))},
        "pr_auc": {"isolation_forest": float(average_precision_score(y, scored["if_anomaly"])),
                   "rules": float(average_precision_score(y, rule_pts)),
                   "blended_risk": float(average_precision_score(y, scored["risk_score"]))},
        "thresholds": {},
        "recall_by_fraud_type": {},
        "rule_stats": {},
    }
    for name, thr in (("review", REVIEW_THRESHOLD), ("decline", DECLINE_THRESHOLD)):
        pred = scored["risk_score"].to_numpy() >= thr
        tp = int((pred & (y == 1)).sum())
        metrics["thresholds"][name] = {
            "threshold": thr, "precision": tp / max(pred.sum(), 1), "recall": tp / max(y.sum(), 1),
            "flag_rate": float(pred.mean())}
    flagged = scored["risk_score"].to_numpy() >= REVIEW_THRESHOLD
    for ftype, grp in test_df.reset_index(drop=True).groupby("fraud_type"):
        if ftype != "legit":
            metrics["recall_by_fraud_type"][ftype] = float(flagged[grp.index].mean())
    for r in RULES:
        hit = r.predicate(test_df).to_numpy()
        metrics["rule_stats"][r.code] = {"hit_rate": float(hit.mean()),
                                         "precision": float(y[hit].mean()) if hit.any() else None}
    p, rc, th = precision_recall_curve(y, scored["risk_score"])
    step = max(1, len(th) // 200)
    metrics["pr_curve"] = {"precision": p[:-1][::step].round(4).tolist(), "recall": rc[:-1][::step].round(4).tolist(),
                           "threshold": th[::step].round(2).tolist()}
    bundle.metrics = metrics

    # ---- Challenger: supervised, with graph features; same split, same holdout ----
    challenger = fit_challenger(train_df, seed=seed)
    ch_test = ChallengerScorer(challenger).score(test_df)
    holdout = test_df[["fraud_type", "is_fraud"]].reset_index(drop=True).assign(
        risk_score=scored["risk_score"].to_numpy(), challenger_score=ch_test)
    comparison = compare_models(holdout)
    challenger.metrics = comparison
    metrics["champion_vs_challenger"] = comparison
    log.info("Holdout @ equal %.2f%% flag rate: champion PR-AUC %.3f recall %.1f%% · challenger PR-AUC %.3f recall %.1f%% · %s",
             100 * comparison["flag_rate"], comparison["champion"]["pr_auc"], 100 * comparison["champion"]["recall"],
             comparison["challenger"]["pr_auc"], 100 * comparison["challenger"]["recall"], comparison["verdict"])

    # ---- Reference profile for drift monitoring (training distribution of inputs and both scores) ----
    train_scored = Scorer(bundle).score(train_df.reset_index(drop=True))
    ref = train_df[FEATURE_COLUMNS + GRAPH_FEATURES].reset_index(drop=True).assign(
        risk_score=train_scored["risk_score"].to_numpy(), challenger_score=ChallengerScorer(challenger).score(train_df))
    profile = build_reference_profile(ref)
    return bundle, metrics, challenger, profile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train + evaluate the Isolation Forest fraud model")
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--tps", type=float, default=25, help="match the live traffic rate (velocity features are rate-sensitive)")
    ap.add_argument("--traffic-seed", type=int, default=99)
    args = ap.parse_args(argv)
    try:
        df = build_dataset(args.hours, args.tps, args.traffic_seed)
        bundle, m, challenger, profile = train(df)
    except Exception as exc:  # noqa: BLE001
        log.exception("Training failed: %s", exc)
        return 1
    bundle.save()
    challenger.save()
    save_profile(profile)
    card = {k: v for k, v in m.items() if k != "pr_curve"}
    card.update({"version": bundle.version, "features": FEATURE_COLUMNS, "top_importances": dict(list(bundle.importances.items())[:10]),
                 "challenger": {"version": challenger.version, "features": challenger.features}})
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    log.info("Saved %s · ROC-AUC blended=%.3f IF=%.3f rules=%.3f · PR-AUC blended=%.3f",
             bundle.version, m["roc_auc"]["blended_risk"], m["roc_auc"]["isolation_forest"], m["roc_auc"]["rules"],
             m["pr_auc"]["blended_risk"])
    for k, v in m["thresholds"].items():
        log.info("  @%s (%.0f): precision=%.3f recall=%.3f flag_rate=%.4f", k, v["threshold"], v["precision"],
                 v["recall"], v["flag_rate"])
    log.info("  recall by type: %s", {k: round(v, 3) for k, v in m["recall_by_fraud_type"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
