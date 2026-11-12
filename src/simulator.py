"""
Payment-switch event simulator (the producer).

Generates a realistic authorisation stream and publishes it to the broker:

* **Legit traffic** – 20k cardholders with a home city, spend level, favourite
  merchant categories and an enrolled device; diurnal arrival rate; occasional
  genuine travel and new devices (realistic false-positive pressure).
* **Injected fraud episodes** (≈2% of events), each a multi-event attack:
    - ``card_testing``        burst of tiny CNP charges, then a large cash-out
    - ``account_takeover``    new device + foreign IP + high-value risky merchants
    - ``impossible_travel``   cloned card used in-person thousands of km away minutes later
    - ``velocity_burst``      many medium charges across distinct merchants in minutes
    - ``high_amount_anomaly`` single charge 10-30x the cardholder's norm
* **Coordinated fraud rings** (2026): a ring compromises many cards over hours and drives them from a
  handful of shared devices through residential proxies in each victim's own country, paying everyday
  amounts into 1-2 collector accounts. Each payment looks normal on its own, so per-transaction rules
  miss them; the shared-device graph (``src/graph.py``) and the challenger model's graph features
  catch them.
* **Households** – some legitimate cards of the same customer share a device (realistic graph noise).
* **Optional drift** – ``--drift-after-hours`` inflates legit spend (x1.8) and new-device use (3% -> 10%),
  to exercise drift monitoring.
* **Delayed labels** – chargebacks arrive 5-90 min later on a separate topic,
  as in real life (models are always evaluated on partially-labelled data).
* **Dirty data** – ~0.1% malformed events (DLQ path) and ~1% late,
  out-of-order events (event-time handling).

Modes
-----
    python -m src.simulator backfill --hours 24 --tps 10     # history, as fast as possible
    python -m src.simulator live --tps 25                    # real-time pacing
    python -m src.simulator run --hours 24 --tps 25          # backfill, then continue live
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from src.broker import get_broker
from src.config import (REFERENCE_DIR, SIM_FRAUD_RATE, SIM_HOUSEHOLD_RATE, SIM_LABEL_DELAY_MIN, SIM_LATE_EVENT_RATE,
                        SIM_MALFORMED_RATE, SIM_N_CARDS, SIM_N_MERCHANTS, SIM_RING_RATE, SIM_SEED, STATE_DIR,
                        TOPIC_LABELS, TOPIC_TRANSACTIONS, get_logger)
from src.reference import (CITIES, EPISODE_TYPES, FX_TO_USD, HIGH_RISK_CATEGORIES, MERCHANT_CATEGORIES,
                           RING_COLLECTOR_CATEGORIES)

log = get_logger("simulator")

FRAUD_TYPE_WEIGHTS = [0.25, 0.20, 0.20, 0.15, 0.20]
EVERYDAY_CATEGORIES = [c for c in MERCHANT_CATEGORIES if c not in HIGH_RISK_CATEGORIES and c not in ("ATM Cash", "P2P Wallet")]


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(min(1.0, a)))


@dataclass
class Card:
    card_id: str
    customer_id: str
    home: tuple                      # entry from CITIES
    avg_amount_usd: float
    activity: float                  # relative transaction frequency
    prefs: list[str]
    device_id: str
    tier: str


@dataclass(order=True)
class _Pending:
    ts: float
    seq: int
    event: dict = field(compare=False)
    label: dict | None = field(compare=False, default=None)


class TransactionSimulator:
    """Stateful generator: call ``events_between(t0, t1)`` repeatedly with advancing windows."""

    def __init__(self, seed: int = SIM_SEED, n_cards: int = SIM_N_CARDS, n_merchants: int = SIM_N_MERCHANTS,
                 fraud_rate: float = SIM_FRAUD_RATE, malformed_rate: float = SIM_MALFORMED_RATE,
                 late_rate: float = SIM_LATE_EVENT_RATE, traffic_seed: int | None = None,
                 ring_rate: float = SIM_RING_RATE, household_rate: float = SIM_HOUSEHOLD_RATE,
                 drift_start: float | None = None):
        # `seed` fixes the card/merchant population (shared by producer, trainer and consumer);
        # `traffic_seed` drives the event stream, so training data never replays the live stream.
        self.rng = random.Random(seed)
        self.fraud_rate, self.malformed_rate, self.late_rate = fraud_rate, malformed_rate, late_rate
        self.ring_rate, self.household_rate, self.drift_start = ring_rate, household_rate, drift_start
        self._rings: list[dict] = []                 # active rings (devices, collector merchants, expiry)
        self._ring_seq = itertools.count(1)
        self.ring_truth: dict[str, str] = {}         # card_id -> ring_id (ground truth for evaluation)
        self._seq = itertools.count()
        self._heap: list[_Pending] = []
        self._next_arrival: float | None = None
        self._travel: dict[int, tuple[tuple, float]] = {}
        self.stats = {"legit": 0, "fraud": 0, "malformed": 0, "late": 0, "episodes": 0, "ring_events": 0, "rings": 0}

        city_w = [c[5] for c in CITIES]
        self.cards: list[Card] = []
        for i in range(n_cards):
            home = self.rng.choices(CITIES, weights=city_w)[0]
            self.cards.append(Card(
                card_id=f"card_{i:06d}", customer_id=f"cust_{i // 1.3:06.0f}", home=home,
                avg_amount_usd=round(math.exp(self.rng.gauss(math.log(48), 0.55)), 2),
                activity=math.exp(self.rng.gauss(0, 0.6)),
                prefs=self.rng.sample(EVERYDAY_CATEGORIES, 3),
                device_id=f"dev_{self.rng.getrandbits(40):010x}",
                tier=self.rng.choices(["standard", "gold", "platinum"], weights=[0.7, 0.22, 0.08])[0],
            ))
        self._card_cum = list(itertools.accumulate(c.activity for c in self.cards))

        self.merchants: dict[str, list[tuple[str, str]]] = {c: [] for c in MERCHANT_CATEGORIES}
        prefixes = ["Blue", "North", "Prime", "Urban", "Golden", "Swift", "Metro", "Nova", "Peak", "Coastal"]
        for j in range(n_merchants):
            cat = self.rng.choice(list(MERCHANT_CATEGORIES))
            name = f"{self.rng.choice(prefixes)} {cat.split(' ')[0]} {j:04d}"
            self.merchants[cat].append((f"m_{j:05d}", name))
        for cat, lst in self.merchants.items():  # guarantee every category has merchants
            if not lst:
                lst.append((f"m_{cat[:3].lower()}0", f"{cat} Co"))
        self.rng = random.Random(seed + 1 if traffic_seed is None else traffic_seed)

    # ------------------------------------------------------------------ reference
    def reference_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cards = pd.DataFrame([{
            "card_id": c.card_id, "customer_id": c.customer_id, "home_city": c.home[0], "home_country": c.home[1],
            "home_lat": c.home[2], "home_lon": c.home[3], "home_currency": c.home[4], "profile_avg_usd": c.avg_amount_usd,
            "enrolled_device": c.device_id, "tier": c.tier} for c in self.cards])
        merchants = pd.DataFrame([{"merchant_id": m, "merchant_name": n, "merchant_category": cat}
                                  for cat, lst in self.merchants.items() for m, n in lst])
        return cards, merchants

    def write_reference(self) -> None:
        cards, merchants = self.reference_frames()
        cards.to_parquet(REFERENCE_DIR / "cards.parquet", index=False)
        merchants.to_parquet(REFERENCE_DIR / "merchants.parquet", index=False)
        log.info("Reference data written: %s cards, %s merchants", f"{len(cards):,}", f"{len(merchants):,}")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def diurnal(ts: float) -> float:
        """Traffic multiplier: trough ~04:00 UTC, peak ~19:00 UTC (avg ≈ 1)."""
        hour = (ts % 86400) / 3600
        return 0.35 + 1.3 * (0.5 + 0.5 * math.sin((hour - 13) / 24 * 2 * math.pi))

    def _pick_card(self) -> int:
        u = self.rng.random() * self._card_cum[-1]
        lo, hi = 0, len(self._card_cum) - 1
        while lo < hi:  # bisect
            mid = (lo + hi) // 2
            if self._card_cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _txn_id(self) -> str:
        return f"{self.rng.getrandbits(64):016x}{self.rng.getrandbits(32):08x}"

    def _foreign_city(self, home: tuple, min_km: float = 0) -> tuple:
        options = [c for c in CITIES if c[1] != home[1] and haversine_km(home[2], home[3], c[2], c[3]) >= min_km]
        return self.rng.choice(options or [c for c in CITIES if c[1] != home[1]])

    def _make(self, ts: float, card: Card, city: tuple, category: str, amount_usd: float, *, txn_type: str | None = None,
              device: str | None = None, ip_country: str | None = None,
              merchant: tuple[str, str] | None = None) -> dict:
        spec = MERCHANT_CATEGORIES[category]
        if txn_type is None:
            types = spec["types"]
            txn_type = "ECOM" if "ECOM" in types and self.rng.random() < spec["online"] else self.rng.choice(types)
        cnp = txn_type in ("ECOM", "TRANSFER", "P2P")
        merchant_id, merchant_name = merchant or self.rng.choice(self.merchants[category])
        currency = "USD" if cnp and self.rng.random() < 0.3 else city[4]
        fx = FX_TO_USD[currency]
        amount = amount_usd / fx
        if txn_type == "ATM":  # cash comes in round notes
            amount = max(20.0, round(amount / 20) * 20)
        amount = round(amount, 0 if currency in ("JPY", "KRW", "NGN") else 2)
        return {
            "schema_version": 1, "txn_id": self._txn_id(), "event_time": iso(ts), "card_id": card.card_id,
            "merchant_id": merchant_id, "merchant_name": merchant_name, "merchant_category": category,
            "txn_type": txn_type, "channel": "card_not_present" if cnp else "card_present",
            "amount": max(amount, 0.01), "currency": currency, "amount_usd": round(max(amount * fx, 0.01), 2),
            "country": city[1], "city": city[0],
            "lat": round(city[2] + self.rng.gauss(0, 0.05), 4), "lon": round(city[3] + self.rng.gauss(0, 0.05), 4),
            "device_id": (device or card.device_id) if cnp else f"term_{merchant_id}",
            "ip_country": (ip_country or city[1]) if cnp else city[1],
        }

    def _push(self, ts: float, event: dict, label: dict | None = None) -> None:
        heapq.heappush(self._heap, _Pending(ts, next(self._seq), event, label))

    def _fraud_label(self, ev: dict, ts: float, ftype: str) -> dict:
        lo, hi = SIM_LABEL_DELAY_MIN
        return {"txn_id": ev["txn_id"], "card_id": ev["card_id"], "is_fraud": True, "fraud_type": ftype,
                "label_time": iso(ts + self.rng.uniform(lo, hi) * 60)}

    # ------------------------------------------------------------------ traffic
    def _legit(self, ts: float) -> None:
        idx = self._pick_card()
        card = self.cards[idx]
        # Genuine travel: (destination, depart, arrive, leave, back home). Travel takes real time,
        # and a card doesn't transact in the air, so honest travellers never "teleport" and
        # never trip the impossible-travel rule (an earlier version did, inflating false positives).
        trip = self._travel.get(idx)
        if trip and ts >= trip[4]:
            self._travel.pop(idx)
            trip = None
        if trip is None and self.rng.random() < 0.002:
            dest = self._foreign_city(card.home)
            flight_s = (haversine_km(card.home[2], card.home[3], dest[2], dest[3]) / 800 + 1.5) * 3600
            arrive = ts + flight_s
            leave = arrive + self.rng.uniform(1, 5) * 86400
            trip = (dest, ts, arrive, leave, leave + flight_s)
            self._travel[idx] = trip
        if trip and (ts < trip[2] or trip[3] <= ts < trip[4]):
            return  # in transit: no card activity
        city = trip[0] if trip else card.home
        category = self.rng.choice(card.prefs) if self.rng.random() < 0.75 else self.rng.choice(EVERYDAY_CATEGORIES + ["ATM Cash", "P2P Wallet"])
        amount = card.avg_amount_usd * MERCHANT_CATEGORIES[category]["amount"] * math.exp(self.rng.gauss(0, 0.5))
        drifted = self.drift_start is not None and ts >= self.drift_start
        new_dev_p = 0.10 if drifted else 0.03
        if drifted:
            amount *= 1.8                                   # e.g. a price shock / new premium product mix
        u = self.rng.random()
        if u < new_dev_p:
            device = f"dev_{self.rng.getrandbits(40):010x}"  # legit new phone/laptop
        elif u < new_dev_p + self.household_rate:
            device = f"dev_hh_{card.customer_id}"           # shared family tablet: cards of one customer
        else:
            device = None
        self._push(ts, self._make(ts, card, city, category, amount, device=device))
        self.stats["legit"] += 1
        if self.rng.random() < self.fraud_rate:
            self._episode(ts, idx)
        if self.rng.random() < self.ring_rate:
            self._ring(ts, idx)

    def _episode(self, ts: float, idx: int) -> None:
        card = self.cards[idx]
        ftype = self.rng.choices(EPISODE_TYPES, weights=FRAUD_TYPE_WEIGHTS)[0]
        self.stats["episodes"] += 1
        t = ts + self.rng.uniform(30, 600)
        events: list[tuple[float, dict]] = []
        r = self.rng

        if ftype == "card_testing":
            city = self._foreign_city(card.home)
            dev = f"dev_fr_{r.getrandbits(32):08x}"
            for _ in range(r.randint(6, 14)):
                t += r.uniform(4, 25)
                cat = r.choice(["Digital Goods", "Streaming & Apps", "Gift Cards"])
                events.append((t, self._make(t, card, city, cat, r.uniform(0.5, 3.0), txn_type="ECOM", device=dev, ip_country=city[1])))
            for _ in range(r.randint(1, 2)):
                t += r.uniform(30, 180)
                events.append((t, self._make(t, card, city, r.choice(["Electronics", "Gift Cards"]), r.uniform(250, 1500),
                                             txn_type="ECOM", device=dev, ip_country=city[1])))
        elif ftype == "account_takeover":
            city = self._foreign_city(card.home, 1500)
            dev = f"dev_fr_{r.getrandbits(32):08x}"
            for _ in range(r.randint(2, 4)):
                t += r.uniform(60, 600)
                cat = r.choice(["Electronics", "Crypto Exchange", "Gift Cards", "Travel & Airlines"])
                events.append((t, self._make(t, card, city, cat, card.avg_amount_usd * r.uniform(6, 18),
                                             txn_type="ECOM", device=dev, ip_country=city[1])))
        elif ftype == "impossible_travel":
            # Genuine purchase at home, then the cloned card is used in person far away minutes later.
            genuine = self._make(t, card, card.home, r.choice(card.prefs), card.avg_amount_usd * r.uniform(0.5, 1.5))
            self._push(t, genuine)
            self.stats["legit"] += 1
            far = self._foreign_city(card.home, 3000)
            for _ in range(r.randint(1, 3)):
                t += r.uniform(180, 2400)
                cat = r.choice(["ATM Cash", "Electronics", "Fashion", "Gift Cards"])
                tt = "ATM" if cat == "ATM Cash" else "POS"
                events.append((t, self._make(t, card, far, cat, card.avg_amount_usd * r.uniform(2, 6), txn_type=tt)))
        elif ftype == "velocity_burst":
            dev = f"dev_fr_{r.getrandbits(32):08x}" if r.random() < 0.5 else None
            for _ in range(r.randint(8, 20)):
                t += r.uniform(10, 60)
                cat = r.choice(["Fashion", "Electronics", "Gift Cards", "Digital Goods"])
                events.append((t, self._make(t, card, card.home, cat, r.uniform(50, 300), txn_type="ECOM", device=dev)))
        else:  # high_amount_anomaly
            dev = None if r.random() < 0.6 else f"dev_fr_{r.getrandbits(32):08x}"
            cat = r.choice(["Travel & Airlines", "Electronics", "Crypto Exchange", "Hotels"])
            events.append((t, self._make(t, card, card.home, cat, card.avg_amount_usd * r.uniform(10, 30),
                                         txn_type="ECOM", device=dev)))

        for et, ev in events:
            self._push(et, ev, self._fraud_label(ev, et, ftype))
            self.stats["fraud"] += 1

    def _ring(self, ts: float, idx: int) -> None:
        """A fraud ring takes over this card and cashes it out through its shared devices."""
        card, r = self.cards[idx], self.rng
        if card.card_id in self.ring_truth:
            return
        # A ring recruits until it hits its mule capacity or goes quiet, then a new crew takes over.
        self._rings = [g for g in self._rings if g["expires"] > ts and g["n"] < g["cap"]]
        if self._rings and r.random() < 0.85:
            ring = r.choice(self._rings)
        else:
            cats = r.sample(RING_COLLECTOR_CATEGORIES, 2)
            ring = {"id": f"ring_{next(self._ring_seq):04d}", "expires": ts + r.uniform(3, 12) * 3600,
                    "cap": r.randint(4, 16), "n": 0,
                    "devices": [f"dev_rg_{r.getrandbits(36):09x}" for _ in range(r.randint(2, 3))],
                    "collectors": [(c, r.choice(self.merchants[c])) for c in cats]}
            self._rings.append(ring)
            self.stats["rings"] += 1
        self.ring_truth[card.card_id] = ring["id"]
        ring["n"] += 1
        t = ts
        for _ in range(r.randint(1, 3)):
            t += r.uniform(5, 90) * 60
            cat, merchant = r.choice(ring["collectors"])
            ev = self._make(t, card, card.home, cat, card.avg_amount_usd * r.uniform(0.6, 2.0),
                            txn_type="P2P" if cat == "P2P Wallet" else "ECOM", device=r.choice(ring["devices"]),
                            ip_country=card.home[1], merchant=merchant)   # proxy inside the victim's country
            self._push(t, ev, self._fraud_label(ev, t, "fraud_ring"))
            self.stats["fraud"] += 1
            self.stats["ring_events"] += 1

    def _corrupt(self, ev: dict) -> dict:
        """Produce one of several realistic upstream defects."""
        ev = dict(ev)
        defect = self.rng.choice(["missing_card", "negative_amount", "bad_currency", "bad_geo", "bad_time"])
        if defect == "missing_card":
            ev.pop("card_id", None)
        elif defect == "negative_amount":
            ev["amount"] = -abs(ev["amount"])
        elif defect == "bad_currency":
            ev["currency"] = "X1"
        elif defect == "bad_geo":
            ev["lat"] = 123.4
        else:
            ev["event_time"] = "not-a-timestamp"
        return ev

    def events_between(self, t0: float, t1: float, base_tps: float) -> list[tuple[dict, dict | None]]:
        """All events with event_time <= t1 (in emission order). Episodes may spill into later windows."""
        if self._next_arrival is None:
            self._next_arrival = t0
        while self._next_arrival <= t1:
            self._legit(self._next_arrival)
            self._next_arrival += self.rng.expovariate(base_tps * self.diurnal(self._next_arrival))
        out = []
        while self._heap and self._heap[0].ts <= t1:
            p = heapq.heappop(self._heap)
            ev = p.event
            if self.rng.random() < self.malformed_rate:
                ev = self._corrupt(ev)
                self.stats["malformed"] += 1
                out.append((ev, None))
                continue
            if self.rng.random() < self.late_rate:  # arrives now, but happened 30-180 s ago
                ev = dict(ev, event_time=iso(p.ts - self.rng.uniform(30, 180)))
                self.stats["late"] += 1
            out.append((ev, p.label))
        return out


# ============================================================================ #
# Publishing loop
# ============================================================================ #
class Publisher:
    def __init__(self, sim: TransactionSimulator):
        self.sim = sim
        self.broker = get_broker()
        self.pending_labels: list[tuple[float, int, dict]] = []
        self._seq = itertools.count()
        self._stop = False
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, *_):
        log.info("Stop requested – finishing current batch")
        self._stop = True

    def _publish(self, items: list[tuple[dict, dict | None]], now: float) -> int:
        recs = [(ev.get("card_id"), json.dumps(ev, separators=(",", ":"))) for ev, _ in items]
        n = self.broker.produce_batch(TOPIC_TRANSACTIONS, recs)
        for _, label in items:
            if label:
                due = datetime.fromisoformat(label["label_time"]).timestamp()
                heapq.heappush(self.pending_labels, (due, next(self._seq), label))
        due_labels = []
        while self.pending_labels and self.pending_labels[0][0] <= now:
            due_labels.append(heapq.heappop(self.pending_labels)[2])
        if due_labels:
            self.broker.produce_batch(TOPIC_LABELS, [(lb["card_id"], json.dumps(lb)) for lb in due_labels])
        return n

    def backfill(self, hours: float, tps: float, chunk_s: float = 600) -> float:
        end = time.time()
        t = end - hours * 3600
        total, t_start = 0, time.perf_counter()
        log.info("Backfilling %.1f h of history at ~%.0f TPS (event time)", hours, tps)
        while t < end and not self._stop:
            t_next = min(t + chunk_s, end)
            items = self.sim.events_between(t, t_next, tps)
            # In history, a label is "due" once its label_time is in the simulated past.
            total += self._publish(items, now=t_next)
            t = t_next
        rate = total / max(time.perf_counter() - t_start, 1e-6)
        log.info("Backfill done: %s events (%.0f ev/s) · stats=%s", f"{total:,}", rate, self.sim.stats)
        return end

    def live(self, tps: float, duration: float | None, start: float | None = None, tick_s: float = 0.5) -> None:
        t = start or time.time()
        t_end = time.time() + duration if duration else float("inf")
        log.info("Live mode at ~%.0f TPS (Ctrl+C to stop)", tps)
        sent, last_log = 0, time.time()
        while not self._stop and time.time() < t_end:
            now = time.time()
            items = self.sim.events_between(t, now, tps)
            sent += self._publish(items, now)
            t = now
            if now - last_log >= 10:
                log.info("live: %s events sent · pending labels %d · stats=%s", f"{sent:,}", len(self.pending_labels), self.sim.stats)
                last_log = now
            time.sleep(max(0.0, tick_s - (time.time() - now)))
        self._save_pending()

    def _save_pending(self) -> None:
        path = STATE_DIR / "pending_labels.json"
        path.write_text(json.dumps([lb for _, _, lb in self.pending_labels]))
        log.info("Saved %d not-yet-due labels to %s", len(self.pending_labels), path.name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Payment transaction stream simulator")
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("backfill", "live", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--tps", type=float, default=25.0, help="average transactions per second (event time)")
        if name in ("backfill", "run"):
            sp.add_argument("--hours", type=float, default=24.0, help="hours of history to backfill")
        if name in ("live", "run"):
            sp.add_argument("--duration", type=float, default=None, help="seconds to run live (default: forever)")
        sp.add_argument("--drift-after-hours", type=float, default=None,
                        help="inject distribution drift this many hours after the simulated start (monitoring demo)")
    args = p.parse_args(argv)
    if args.tps <= 0:
        log.error("--tps must be positive")
        return 2

    start = time.time() - (args.hours * 3600 if args.mode in ("backfill", "run") else 0)
    drift_start = start + args.drift_after_hours * 3600 if args.drift_after_hours is not None else None
    sim = TransactionSimulator(drift_start=drift_start)
    sim.write_reference()
    if drift_start:
        log.info("Drift scenario: legit spend x1.8 and new-device use 3%% -> 10%% from %s", iso(drift_start))
    pub = Publisher(sim)
    try:
        if args.mode == "backfill":
            pub.backfill(args.hours, args.tps)
            pub._save_pending()
        elif args.mode == "live":
            pub.live(args.tps, args.duration)
        else:
            end = pub.backfill(args.hours, args.tps)
            pub.live(args.tps, args.duration, start=end)
    except Exception as exc:  # noqa: BLE001 – log and exit non-zero for the supervisor to restart us
        log.exception("Simulator crashed: %s", exc)
        return 1
    finally:
        pub.broker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
