"""
Unit + integration tests.   pytest -q

The integration test runs the real producer -> broker -> consumer -> lake path
in a temp directory with a small Isolation Forest, so it needs no trained model
or running services.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.ensemble import IsolationForest

from src.broker import SQLiteBroker, partition_for
from src.detection import RULES, ModelBundle, Scorer, evaluate_rules
from src.features import FEATURE_COLUMNS, FeatureEngine, as_matrix
from src.schemas import Transaction
from src.simulator import TransactionSimulator, haversine_km

T0 = datetime(2026, 1, 1, 12, tzinfo=timezone.utc).timestamp()


# ------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def sim() -> TransactionSimulator:
    return TransactionSimulator(n_cards=500, n_merchants=200, fraud_rate=0.02)


@pytest.fixture(scope="session")
def cards(sim) -> pd.DataFrame:
    return sim.reference_frames()[0]


@pytest.fixture(scope="session")
def tiny_bundle(sim, cards) -> ModelBundle:
    events = [e for e, _ in sim.events_between(T0 - 7200, T0, 5) if _valid(e)]
    eng = FeatureEngine(cards)
    ts = [datetime.fromisoformat(e["event_time"]).timestamp() for e in events]
    X = as_matrix(eng.compute_batch(events, ts))
    model = IsolationForest(n_estimators=50, random_state=0).fit(X)
    q = np.quantile(X, [0.25, 0.5, 0.75], axis=0)
    return ModelBundle(model=model, version="test", train_scores=np.sort(-model.score_samples(X)),
                       feature_center=q[1], feature_scale=np.where(q[2] - q[0] > 0, q[2] - q[0], 1.0))


def _ev(card="card_000001", ts=T0, amount=40.0, lat=40.71, lon=-74.01, country="USA", merchant="m1", device=None,
        channel="card_present", ip=None, category="Grocery", ttype="POS") -> dict:
    return {"txn_id": f"t{ts}{amount}{merchant}", "event_time": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
            "card_id": card, "merchant_id": merchant, "merchant_name": "M", "merchant_category": category,
            "txn_type": ttype, "channel": channel, "amount": amount, "currency": "USD", "amount_usd": amount,
            "country": country, "city": "X", "lat": lat, "lon": lon, "device_id": device or "dev_home",
            "ip_country": ip or country}


@pytest.fixture()
def engine() -> FeatureEngine:
    cards = pd.DataFrame([{"card_id": "card_000001", "home_country": "USA", "home_lat": 40.71, "home_lon": -74.01,
                           "profile_avg_usd": 40.0, "enrolled_device": "dev_home"}])
    return FeatureEngine(cards)


# --------------------------------------------------------------------- broker
def test_broker_roundtrip_commit_and_resume(tmp_path):
    b = SQLiteBroker(tmp_path / "b.sqlite", n_partitions=3)
    b.produce_batch("t", [(f"k{i % 7}", json.dumps({"i": i})) for i in range(100)])
    b.subscribe("t", "g")
    got = []
    while msgs := b.poll(30):
        got += msgs
    assert len(got) == 100
    assert sum(b.lag().values()) == 100  # nothing committed yet
    b.commit()
    assert sum(b.lag().values()) == 0
    b.produce_batch("t", [("k1", "{}")])
    b2 = SQLiteBroker(tmp_path / "b.sqlite", n_partitions=3)
    b2.subscribe("t", "g")  # a restarted consumer resumes after the committed offsets
    assert len(b2.poll(100)) == 1


def test_per_key_ordering_and_stable_routing(tmp_path):
    assert partition_for("card_1", 4) == partition_for("card_1", 4)
    b = SQLiteBroker(tmp_path / "b.sqlite", n_partitions=4)
    b.produce_batch("t", [("card_1", str(i)) for i in range(50)])
    b.subscribe("t", "g")
    vals = [int(m.value) for m in b.poll(1000) if m.key == "card_1"]
    assert vals == sorted(vals) and len(vals) == 50


def test_retention_never_resets_offsets(tmp_path):
    b = SQLiteBroker(tmp_path / "b.sqlite", n_partitions=1)
    b.produce_batch("t", [("k", "a"), ("k", "b")])
    b.subscribe("t", "g")
    b.poll(10)
    b.commit()
    b.apply_retention("t", max_age_s=-1)          # everything is "old"
    b.produce_batch("t", [("k", "c")])
    assert b.end_offsets("t")[0] == 3             # continues at 3, not 0
    assert [m.value for m in b.poll(10)] == ["c"]


# -------------------------------------------------------------------- schemas
def test_schema_accepts_valid_and_rejects_defects():
    Transaction.model_validate(_ev())
    for bad in ({"amount": -5}, {"currency": "X1"}, {"lat": 123.0}, {"event_time": "nope"}):
        with pytest.raises(ValidationError):
            Transaction.model_validate({**_ev(), **bad})
    with pytest.raises(ValidationError):
        Transaction.model_validate({k: v for k, v in _ev().items() if k != "card_id"})


# ------------------------------------------------------------------- features
def test_score_then_update_and_velocity(engine):
    f0 = engine.compute(_ev(ts=T0), T0)
    assert f0["txn_count_1m"] == 0 and f0["txn_count_1h"] == 0     # the event never sees itself
    for i in range(1, 5):
        f = engine.compute(_ev(ts=T0 + i * 10, merchant=f"m{i}"), T0 + i * 10)
    assert f["txn_count_1m"] == 4 and f["distinct_merchants_10m"] == 3  # m1 (twice), m2, m3 seen before


def test_impossible_travel_speed(engine):
    engine.compute(_ev(ts=T0), T0)
    f = engine.compute(_ev(ts=T0 + 600, lat=51.51, lon=-0.13, country="GBR"), T0 + 600)  # NYC -> London in 10 min
    assert f["km_from_last"] > 5000 and f["speed_kmh"] > 20_000 and f["geo_mismatch"] == 1


def test_late_event_uses_its_own_past(engine):
    for i in range(5):
        engine.compute(_ev(ts=T0 + 100 + i), T0 + 100 + i)
    late = engine.compute(_ev(ts=T0), T0)     # happened before all of them
    assert late["txn_count_1h"] == 0


def test_new_device_only_for_card_not_present(engine):
    f = engine.compute(_ev(channel="card_not_present", device="dev_new", ip="RUS", ttype="ECOM"), T0)
    assert f["new_device"] == 1 and f["ip_mismatch"] == 1
    f2 = engine.compute(_ev(ts=T0 + 5, channel="card_not_present", device="dev_new", ttype="ECOM"), T0 + 5)
    assert f2["new_device"] == 0  # remembered after first use


def test_checkpoint_roundtrip(engine, tmp_path):
    engine.compute(_ev(), T0)
    engine.checkpoint(tmp_path / "s.pkl")
    other = FeatureEngine(pd.DataFrame([{"card_id": "card_000001", "home_country": "USA", "home_lat": 40.71,
                                         "home_lon": -74.01, "profile_avg_usd": 40.0, "enrolled_device": "dev_home"}]))
    assert other.restore(tmp_path / "s.pkl") and other.events_processed == 1


def test_haversine_known_distance():
    assert 5500 < haversine_km(40.71, -74.01, 51.51, -0.13) < 5650  # NYC-London ≈ 5,570 km


# ------------------------------------------------------------------ detection
def test_rules_fire_on_signatures(engine):
    rows = [engine.compute(_ev(ts=T0 + i * 5, amount=1.0, merchant=f"m{i}", channel="card_not_present", ttype="ECOM",
                               device="dev_x", ip="RUS", category="Digital Goods"), T0 + i * 5) for i in range(8)]
    _, codes = evaluate_rules(pd.DataFrame(rows))
    fired = set(codes[-1])
    assert {"VELOCITY_1M", "SMALL_TXN_BURST", "MERCHANT_SPRAY"} <= fired


def test_scorer_ranges_and_explanations(tiny_bundle, cards, sim):
    eng = FeatureEngine(cards)
    events = [e for e, _ in sim.events_between(T0, T0 + 600, 5) if _valid(e)][:200]
    ts = [datetime.fromisoformat(e["event_time"]).timestamp() for e in events]
    out = Scorer(tiny_bundle).score(eng.compute_batch(events, ts))
    assert out["risk_score"].between(0, 100).all()
    assert set(out["decision"]) <= {"approve", "review", "decline"}
    assert all(len(json.loads(t)) == 3 for t in out["top_features"])


# ------------------------------------------------------------------ simulator
def test_simulator_deterministic_and_labelled():
    a = TransactionSimulator(n_cards=200, n_merchants=50, fraud_rate=0.02).events_between(T0, T0 + 3600, 3)
    b = TransactionSimulator(n_cards=200, n_merchants=50, fraud_rate=0.02).events_between(T0, T0 + 3600, 3)
    assert [e["txn_id"] for e, _ in a] == [e["txn_id"] for e, _ in b]
    labels = [lb for _, lb in a if lb]
    assert labels and all(lb["is_fraud"] for lb in labels)
    assert all(datetime.fromisoformat(lb["label_time"]) > datetime.fromtimestamp(T0, timezone.utc) for lb in labels)


# ---------------------------------------------------------------- integration
def test_end_to_end_stream_to_lake(tmp_path, monkeypatch, sim, cards, tiny_bundle):
    import src.consumer as consumer
    import src.storage as storage

    broker_path = tmp_path / "broker.sqlite"
    dirs = {n: tmp_path / "lake" / n for n in ("scored", "labels", "metrics", "dlq")}
    monkeypatch.setattr(consumer, "get_broker", lambda: SQLiteBroker(broker_path, n_partitions=2))
    monkeypatch.setattr(consumer, "FeatureEngine", lambda: FeatureEngine(cards))
    monkeypatch.setattr(consumer, "Scorer", lambda: Scorer(tiny_bundle))
    for name, attr in (("scored", "SCORED_DIR"), ("labels", "LABELS_DIR"), ("metrics", "METRICS_DIR"), ("dlq", "DLQ_DIR")):
        monkeypatch.setattr(consumer, attr, dirs[name])
    monkeypatch.setattr(consumer.FeatureEngine, "checkpoint", lambda self, *a, **k: None, raising=False)

    items = TransactionSimulator(n_cards=500, n_merchants=200, fraud_rate=0.02, malformed_rate=0.02,
                                 traffic_seed=5).events_between(T0, T0 + 1800, 3)
    prod = SQLiteBroker(broker_path, n_partitions=2)
    prod.produce_batch("transactions.v1", [(e.get("card_id"), json.dumps(e)) for e, _ in items])
    n_bad = sum(1 for e, _ in items if not _valid(e))

    svc = consumer.ScoringService()
    svc.engine.checkpoint = lambda *a, **k: None
    svc.run(max_idle_s=0.5)

    scored = pd.concat([pd.read_parquet(p) for p in dirs["scored"].rglob("*.parquet")])
    dlq = pd.concat([pd.read_parquet(p) for p in dirs["dlq"].rglob("*.parquet")]) if n_bad else pd.DataFrame()
    assert scored["txn_id"].nunique() == len(items) - n_bad
    assert len(dlq) == n_bad
    assert sum(svc.tx.lag().values()) == 0        # offsets committed on shutdown
    assert storage.write_partitioned(pd.DataFrame(), tmp_path, "x") == 0


def _valid(e: dict) -> bool:
    try:
        Transaction.model_validate(e)
        return True
    except ValidationError:
        return False


def test_rule_catalogue_is_well_formed():
    codes = [r.code for r in RULES]
    assert len(codes) == len(set(codes)) and all(r.points > 0 for r in RULES)
    assert set(FEATURE_COLUMNS).isdisjoint({"speed_kmh", "km_from_last"})  # raw companions are not model inputs
    assert math.isclose(sum(1 for _ in RULES), 8)
