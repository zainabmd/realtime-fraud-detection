"""
Stateful, event-time streaming feature engine (an in-process online feature store).

One class is used by **both** offline training and online scoring, which rules
out training/serving skew by construction.

Per-card state (bounded memory):
    * sliding 1-hour window of (time, amount, merchant)     -> velocity features
    * last location + time                                  -> distance / speed (impossible travel)
    * Welford running mean/variance of log-amount, seeded with a
      profile prior (cold-start safe)                       -> amount z-score
    * known-device set (enrolled + previously seen)         -> new-device flag

Entity-graph features (2026): a device -> {card: last seen} map gives each card-not-present event
the number of *other* cards seen on the same device in the last 24 h, and each card's number of
distinct devices. Shared devices are the signature of fraud rings and mule networks; terminals
(card-present) are excluded because every shopper at a store shares them.

Order of operations is **score-then-update**: features for an event only see
the card's *past*, never the event itself (no leakage).

Late, out-of-order events are handled with event-time windows: counts use the
interval [t - w, t] relative to the event's own timestamp, so a late event is
evaluated against what had happened *before it*, not after.
"""
from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REFERENCE_DIR, STATE_DIR, get_logger
from src.reference import CATEGORY_RISK
from src.simulator import haversine_km
from src.utils import atomic_replace

log = get_logger("features")

FEATURE_COLUMNS = [
    "amount_usd_log", "amount_zscore", "amount_to_avg_ratio",
    "txn_count_1m", "txn_count_10m", "txn_count_1h", "amount_sum_1h_log",
    "distinct_merchants_10m", "small_txn_count_10m", "secs_since_last_log",
    "km_from_last_log", "speed_kmh_log", "km_from_home_log",
    "geo_mismatch", "ip_mismatch", "new_device", "card_not_present",
    "hour_sin", "hour_cos", "is_night", "mcc_risk", "is_atm",
]
# Graph features: used by the challenger model only, so the champion's feature contract is unchanged.
GRAPH_FEATURES = ["device_cards_24h", "card_devices_24h"]
WINDOW_S = 3600
GRAPH_WINDOW_S = 86400
SMALL_TXN_USD = 5.0
PRIOR_N, PRIOR_SD = 5.0, 0.6


class CardState:
    __slots__ = ("window", "last_ts", "last_lat", "last_lon", "n", "mean", "m2", "devices", "max_ts")

    def __init__(self, prior_mean: float, enrolled_device: str | None):
        self.window: deque = deque()            # (ts, amount_usd, merchant_id)
        self.last_ts: float | None = None
        self.last_lat = self.last_lon = 0.0
        # Welford accumulators seeded with a pseudo-count prior from the card profile.
        self.n, self.mean, self.m2 = PRIOR_N, prior_mean, PRIOR_SD ** 2 * PRIOR_N
        self.devices: set[str] = {enrolled_device} if enrolled_device else set()
        self.max_ts = 0.0


class FeatureEngine:
    def __init__(self, cards: pd.DataFrame | None = None):
        if cards is None:
            path = REFERENCE_DIR / "cards.parquet"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing – run the simulator (it writes reference data) first")
            cards = pd.read_parquet(path)
        self.profile = {r.card_id: (r.home_country, r.home_lat, r.home_lon, r.profile_avg_usd, r.enrolled_device)
                        for r in cards.itertuples(index=False)}
        self.state: dict[str, CardState] = {}
        self.events_processed = 0
        self.device_cards: dict[str, dict[str, float]] = {}   # device -> {card_id: last seen ts}
        self.card_devices: dict[str, dict[str, float]] = {}   # card -> {device: last seen ts}

    # ------------------------------------------------------------------ core
    def _get_state(self, card_id: str, ev: dict) -> tuple[CardState, tuple]:
        prof = self.profile.get(card_id)
        if prof is None:  # unknown card: home = first seen location, generic prior
            prof = (ev["country"], ev["lat"], ev["lon"], 50.0, None)
            self.profile[card_id] = prof
        st = self.state.get(card_id)
        if st is None:
            st = CardState(math.log1p(prof[3]), prof[4])
            self.state[card_id] = st
        return st, prof

    def compute(self, ev: dict, ts: float) -> dict:
        """Features for one validated event, then fold it into the card's state."""
        st, (home_cc, home_lat, home_lon, _, _) = self._get_state(ev["card_id"], ev)
        amt = float(ev["amount_usd"])
        log_amt = math.log1p(amt)

        # ---- velocity windows (event-time) ----
        c1 = c10 = c60 = small10 = 0
        sum60 = 0.0
        merchants10: set[str] = set()
        for wts, wamt, wmid in st.window:
            dt = ts - wts
            if dt < 0 or dt > WINDOW_S:
                continue  # newer than this (late) event, or outside the window
            c60 += 1
            sum60 += wamt
            if dt <= 600:
                c10 += 1
                merchants10.add(wmid)
                if wamt < SMALL_TXN_USD:
                    small10 += 1
                if dt <= 60:
                    c1 += 1

        # ---- geo / travel ----
        if st.last_ts is None:
            secs_since, km_last, speed = 30 * 86400.0, 0.0, 0.0
        else:
            secs_since = max(abs(ts - st.last_ts), 1.0)
            km_last = haversine_km(st.last_lat, st.last_lon, ev["lat"], ev["lon"])
            speed = km_last / max(secs_since, 60.0) * 3600  # floor at 1 min: GPS jitter ≠ teleport
        km_home = haversine_km(home_lat, home_lon, ev["lat"], ev["lon"])

        # ---- amount profile ----
        std = math.sqrt(st.m2 / st.n) if st.n > 1 else PRIOR_SD
        z = (log_amt - st.mean) / max(std, 0.15)

        cnp = ev["channel"] == "card_not_present"
        dev = ev["device_id"]
        if cnp:
            seen = self.device_cards.get(dev, {})
            dev_cards = sum(1 for c, t in seen.items() if c != ev["card_id"] and 0 <= ts - t <= GRAPH_WINDOW_S)
        else:
            dev_cards = 0
        mine = self.card_devices.get(ev["card_id"], {})
        card_devs = sum(1 for d, t in mine.items() if 0 <= ts - t <= GRAPH_WINDOW_S and d != dev) + (1 if cnp else 0)
        local_hour = ((ts % 86400) / 3600 + ev["lon"] / 15.0) % 24  # solar-time approximation
        feats = {
            "amount_usd_log": log_amt,
            "amount_zscore": z,
            "amount_to_avg_ratio": amt / max(math.expm1(st.mean), 1.0),
            "txn_count_1m": c1, "txn_count_10m": c10, "txn_count_1h": c60,
            "amount_sum_1h_log": math.log1p(sum60),
            "distinct_merchants_10m": len(merchants10),
            "small_txn_count_10m": small10,
            "secs_since_last_log": math.log1p(secs_since),
            "km_from_last_log": math.log1p(km_last),
            "speed_kmh_log": math.log1p(speed),
            "km_from_home_log": math.log1p(km_home),
            "geo_mismatch": int(ev["country"] != home_cc),
            "ip_mismatch": int(cnp and ev["ip_country"] != home_cc),
            "new_device": int(cnp and ev["device_id"] not in st.devices),
            "card_not_present": int(cnp),
            "hour_sin": math.sin(2 * math.pi * local_hour / 24),
            "hour_cos": math.cos(2 * math.pi * local_hour / 24),
            "is_night": int(local_hour < 5),
            "mcc_risk": CATEGORY_RISK.get(ev["merchant_category"], 0.3),
            "is_atm": int(ev["txn_type"] == "ATM"),
            "device_cards_24h": dev_cards, "card_devices_24h": card_devs,
            # Raw, human-readable companions (not model inputs) for rules and analysts:
            "speed_kmh": speed, "km_from_last": km_last, "km_from_home": km_home,
        }

        # ---- update state (after scoring features are frozen) ----
        st.window.append((ts, amt, ev["merchant_id"]))
        st.max_ts = max(st.max_ts, ts)
        while st.window and st.window[0][0] < st.max_ts - WINDOW_S:
            st.window.popleft()
        if st.last_ts is None or ts >= st.last_ts:  # a late event must not rewind "last location"
            st.last_ts, st.last_lat, st.last_lon = ts, ev["lat"], ev["lon"]
        st.n += 1
        delta = log_amt - st.mean
        st.mean += delta / st.n
        st.m2 += delta * (log_amt - st.mean)
        if cnp:
            self._link(ev["card_id"], dev, ts)
            st.devices.add(ev["device_id"])
            if len(st.devices) > 20:  # bound memory
                st.devices.pop()
        self.events_processed += 1
        return feats

    def _link(self, card_id: str, device: str, ts: float) -> None:
        """Record a card<->device edge; prune edges older than the graph window (bounded memory)."""
        for a, b, index in ((device, card_id, self.device_cards), (card_id, device, self.card_devices)):
            edges = index.setdefault(a, {})
            edges[b] = max(edges.get(b, 0.0), ts)
            if len(edges) > 64:
                for k in [k for k, t in edges.items() if ts - t > GRAPH_WINDOW_S]:
                    del edges[k]

    def compute_batch(self, events: list[dict], timestamps: list[float]) -> pd.DataFrame:
        """Vector of feature dicts for a micro-batch, in arrival order (order matters for state)."""
        return pd.DataFrame([self.compute(ev, ts) for ev, ts in zip(events, timestamps)])

    # ------------------------------------------------------------------ persistence
    def checkpoint(self, path: Path = STATE_DIR / "feature_state.pkl") -> None:
        """Atomic snapshot (write temp, then rename) so a crash never leaves a half-written file."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"state": self.state, "events_processed": self.events_processed,
                         "device_cards": self.device_cards, "card_devices": self.card_devices}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        atomic_replace(tmp, path)

    def restore(self, path: Path = STATE_DIR / "feature_state.pkl") -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "rb") as fh:
                snap = pickle.load(fh)
            self.state, self.events_processed = snap["state"], snap["events_processed"]
            self.device_cards = snap.get("device_cards", {})      # absent in pre-2026 checkpoints
            self.card_devices = snap.get("card_devices", {})
            log.info("Restored feature state for %s cards (%s events)", f"{len(self.state):,}",
                     f"{self.events_processed:,}")
            return True
        except Exception as exc:  # noqa: BLE001 – corrupt checkpoint => start cold, don't crash
            log.warning("Could not restore feature state (%s); starting cold", exc)
            return False


def as_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
