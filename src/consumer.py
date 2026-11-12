"""
Streaming scoring service (the consumer pipeline).

Loop:
    poll ─▶ parse & validate ─┬─▶ DLQ (invalid)
                              └─▶ features (stateful) ─▶ score (champion) ─▶ buffer ─▶ Parquet lake
                                                       └─▶ shadow-score (challenger, never decides)
                                                               └─▶ alerts topic (review/decline)
    poll labels topic ─▶ labels lake (delayed ground truth)
    every CHECKPOINT_EVERY_S: flush ─▶ checkpoint feature state ─▶ commit offsets

Delivery semantics
------------------
Offsets are committed **only together with a feature-state checkpoint**, after
the data they cover is durably in the lake. After a crash the consumer
restores the checkpoint and replays from the committed offsets, so feature
state stays consistent with the stream (effectively-once state) while the lake
receives at-least-once writes that the de-duplicating views absorb.

    python -m src.consumer
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.broker import SQLiteBroker, get_broker
from src.config import (BATCH_MAX_EVENTS, CHECKPOINT_EVERY_S, COMPACT_EVERY_S, CONSUMER_GROUP, DLQ_DIR,
                        FLUSH_MAX_ROWS, FLUSH_MAX_SECONDS, LABELS_DIR, METRICS_DIR, SCORED_DIR, TOPIC_ALERTS,
                        TOPIC_LABELS, TOPIC_TRANSACTIONS, get_logger)
from src.challenger import ChallengerBundle, ChallengerScorer
from src.detection import Scorer
from src.features import FEATURE_COLUMNS, GRAPH_FEATURES, FeatureEngine
from src.schemas import FraudLabel, Transaction
from src.storage import compact, write_partitioned

log = get_logger("consumer")

EVENT_COLS = ["txn_id", "card_id", "merchant_id", "merchant_name", "merchant_category", "txn_type", "channel",
              "amount", "currency", "amount_usd", "country", "city", "lat", "lon", "device_id", "ip_country"]
EXTRA_FEATURES = ["speed_kmh", "km_from_last", "km_from_home"]


class ScoringService:
    def __init__(self):
        self.tx = get_broker()
        self.tx.subscribe(TOPIC_TRANSACTIONS, CONSUMER_GROUP)
        self.lb = get_broker()
        self.lb.subscribe(TOPIC_LABELS, CONSUMER_GROUP + "-labels")
        self.alerts = get_broker()
        self.engine = FeatureEngine()
        self.engine.restore()
        self.scorer = Scorer()
        ch = ChallengerBundle.load()
        self.challenger = ChallengerScorer(ch) if ch else None
        home = {cid: (p[0], p[1], p[2]) for cid, p in self.engine.profile.items()}
        self._home = home
        self.buf_scored: list[pd.DataFrame] = []
        self.buf_labels: list[dict] = []
        self.buf_metrics: list[dict] = []
        self.buf_dlq: list[dict] = []
        self._last_flush = self._last_ckpt = self._last_compact = self._last_retention = time.time()
        self._stop = False
        self.totals = {"events": 0, "dlq": 0, "alerts": 0, "labels": 0}
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)
        log.info("Scoring service ready · champion=%s · challenger=%s (shadow)", self.scorer.bundle.version,
                 self.challenger.bundle.version if self.challenger else "none")

    def _request_stop(self, *_):
        log.info("Shutdown requested – draining")
        self._stop = True

    # ------------------------------------------------------------------ processing
    def _validate(self, msgs) -> tuple[list[dict], list[float], list]:
        good, ts, meta = [], [], []
        now = datetime.now(timezone.utc).isoformat()
        for m in msgs:
            try:
                txn = Transaction.model_validate_json(m.value)
            except (ValidationError, ValueError) as exc:
                err = exc.errors()[0] if isinstance(exc, ValidationError) else {"msg": str(exc), "loc": ()}
                self.buf_dlq.append({"received_at": now, "partition": m.partition, "offset": m.offset,
                                     "error": f"{'.'.join(map(str, err.get('loc', ())))}: {err.get('msg')}",
                                     "raw": m.value[:2000], "event_time": now})
                continue
            d = txn.model_dump(mode="json")
            good.append(d)
            ts.append(txn.event_time.timestamp())
            meta.append(m)
        return good, ts, meta

    def process_batch(self, msgs) -> dict:
        t0 = time.perf_counter()
        events, ts, meta = self._validate(msgs)
        n_dlq = len(msgs) - len(events)
        if not events:
            return {"n": len(msgs), "dlq": n_dlq, "alerts": 0, "ms": (time.perf_counter() - t0) * 1000}

        feats = self.engine.compute_batch(events, ts)
        scores = self.scorer.score(feats)
        now = time.time()
        df = pd.DataFrame(events)[EVENT_COLS]
        df["event_time"] = pd.to_datetime(ts, unit="s", utc=True)
        home = [self._home.get(c, (None, np.nan, np.nan)) for c in df["card_id"]]
        df["home_country"] = [h[0] for h in home]
        df["home_lat"] = [h[1] for h in home]
        df["home_lon"] = [h[2] for h in home]
        df = pd.concat([df, feats[FEATURE_COLUMNS + GRAPH_FEATURES + EXTRA_FEATURES].round(4), scores], axis=1)
        if self.challenger:
            try:
                df["challenger_score"] = self.challenger.score(feats)
                df["challenger_version"] = self.challenger.bundle.version
            except Exception as exc:  # noqa: BLE001 – shadow scoring must never block the champion path
                log.warning("Challenger scoring failed (%s); continuing without it", exc)
                df["challenger_score"], df["challenger_version"] = np.nan, None
        df["partition"] = [m.partition for m in meta]
        df["offset"] = [m.offset for m in meta]
        df["broker_ts"] = pd.to_datetime([m.timestamp for m in meta], unit="s", utc=True)
        df["processed_at"] = pd.Timestamp(now, unit="s", tz="UTC")
        # Pipeline latency = broker append -> scored (what the SLO is about); valid in backfill and live.
        df["pipeline_latency_ms"] = ((now - np.array([m.timestamp for m in meta])) * 1000).round(1)
        self.buf_scored.append(df)

        flagged = df[df["decision"] != "approve"]
        if len(flagged):
            payload = flagged[["txn_id", "card_id", "event_time", "amount_usd", "merchant_category", "country",
                               "risk_score", "decision", "rules_triggered"]].assign(event_time=lambda x: x["event_time"].astype(str))
            self.alerts.produce_batch(TOPIC_ALERTS, [(r["card_id"], json.dumps(r)) for r in payload.to_dict("records")])

        ms = (time.perf_counter() - t0) * 1000
        lat = df["pipeline_latency_ms"].to_numpy()
        self.buf_metrics.append({
            "event_time": pd.Timestamp(now, unit="s", tz="UTC"), "batch_events": len(msgs), "valid_events": len(events),
            "dlq_events": n_dlq, "alerts": int(len(flagged)), "processing_ms": round(ms, 2),
            "events_per_s": round(len(msgs) / max(ms / 1000, 1e-6), 1),
            "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95)),
            "latency_p99_ms": float(np.percentile(lat, 99)),
            "max_event_time": df["event_time"].max(), "cards_in_state": len(self.engine.state),
        })
        return {"n": len(msgs), "dlq": n_dlq, "alerts": int(len(flagged)), "ms": ms}

    def poll_labels(self) -> int:
        msgs = self.lb.poll(5000)
        for m in msgs:
            try:
                lb = FraudLabel.model_validate_json(m.value)
                self.buf_labels.append({**lb.model_dump(), "label_time": lb.label_time})
            except ValidationError as exc:
                log.warning("Bad label at %s/%s: %s", m.partition, m.offset, exc.errors()[0]["msg"])
        return len(msgs)

    # ------------------------------------------------------------------ durability
    def flush(self) -> None:
        if self.buf_scored:
            df = pd.concat(self.buf_scored, ignore_index=True)
            write_partitioned(df, SCORED_DIR, "event_time")
            self.buf_scored = []
        if self.buf_labels:
            write_partitioned(pd.DataFrame(self.buf_labels), LABELS_DIR, "label_time")
            self.buf_labels = []
        if self.buf_dlq:
            write_partitioned(pd.DataFrame(self.buf_dlq), DLQ_DIR, "event_time")
            self.buf_dlq = []
        if self.buf_metrics:
            m = pd.DataFrame(self.buf_metrics)
            lag = self.processing_lag()
            m["lag_total"] = sum(lag.values())
            m["lag_by_partition"] = json.dumps(lag)
            write_partitioned(m, METRICS_DIR, "event_time")
            self.buf_metrics = []
        self._last_flush = time.time()

    def processing_lag(self) -> dict[int, int]:
        """Lag vs. in-memory positions (what is still unprocessed), not vs. committed offsets."""
        if isinstance(self.tx, SQLiteBroker):
            ends = self.tx.end_offsets(TOPIC_TRANSACTIONS)
            return {p: max(0, ends[p] - self.tx._positions.get(p, 0)) for p in ends}
        return self.tx.lag()

    def checkpoint(self) -> None:
        """flush -> snapshot state -> commit: the order that makes replay safe."""
        self.flush()
        self.engine.checkpoint()
        self.tx.commit()
        self.lb.commit()
        self._last_ckpt = time.time()

    def _housekeeping(self) -> None:
        now = time.time()
        if now - self._last_compact >= COMPACT_EVERY_S:
            for base, key in ((SCORED_DIR, "txn_id"), (LABELS_DIR, "txn_id"), (METRICS_DIR, None), (DLQ_DIR, None)):
                compact(base, key)
            self._last_compact = now
        if isinstance(self.tx, SQLiteBroker) and now - self._last_retention >= 600:
            for topic in (TOPIC_TRANSACTIONS, TOPIC_LABELS, TOPIC_ALERTS):
                self.tx.apply_retention(topic, max_age_s=6 * 3600)
            self._last_retention = now

    # ------------------------------------------------------------------ main loop
    def run(self, max_idle_s: float | None = None) -> None:
        idle_since = None
        last_log = time.time()
        while not self._stop:
            msgs = self.tx.poll(BATCH_MAX_EVENTS)
            self.totals["labels"] += self.poll_labels()
            if msgs:
                idle_since = None
                try:
                    r = self.process_batch(msgs)
                except Exception:  # noqa: BLE001 – rewind to last commit, then let the supervisor restart
                    log.exception("Batch failed – rewinding to last committed offsets")
                    self.buf_scored.clear()
                    if isinstance(self.tx, SQLiteBroker):
                        self.tx.seek_to_committed()
                    raise
                self.totals["events"] += r["n"]
                self.totals["dlq"] += r["dlq"]
                self.totals["alerts"] += r["alerts"]
            else:
                idle_since = idle_since or time.time()
                if max_idle_s and time.time() - idle_since > max_idle_s:
                    log.info("Idle for %.0fs – exiting (--exit-when-idle)", max_idle_s)
                    break
                time.sleep(0.2)

            rows = sum(len(b) for b in self.buf_scored)
            if rows >= FLUSH_MAX_ROWS or (time.time() - self._last_flush) >= FLUSH_MAX_SECONDS:
                self.flush()
            if time.time() - self._last_ckpt >= CHECKPOINT_EVERY_S:
                self.checkpoint()
            self._housekeeping()
            if time.time() - last_log >= 10:
                log.info("processed=%s alerts=%s dlq=%s labels=%s lag=%s", f"{self.totals['events']:,}",
                         f"{self.totals['alerts']:,}", self.totals["dlq"], self.totals["labels"],
                         sum(self.processing_lag().values()))
                last_log = time.time()
        self.checkpoint()
        for base, key in ((SCORED_DIR, "txn_id"), (LABELS_DIR, "txn_id"), (METRICS_DIR, None), (DLQ_DIR, None)):
            compact(base, key, min_files=2)
        log.info("Stopped cleanly · totals=%s", self.totals)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Streaming fraud scoring consumer")
    ap.add_argument("--exit-when-idle", type=float, default=None, metavar="SECONDS",
                    help="stop after the topic has been empty this long (batch / CI runs)")
    args = ap.parse_args(argv)
    try:
        ScoringService().run(max_idle_s=args.exit_when_idle)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("Consumer crashed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
