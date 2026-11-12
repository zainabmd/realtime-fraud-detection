"""
Tests for the 2026 capabilities: graph features, ring detection, drift, champion/challenger, copilot.
None of these need a running stack or network access.
"""
from __future__ import annotations

import collections
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.challenger import CHALLENGER_FEATURES, ChallengerBundle
from src.config import PSI_DRIFT, PSI_WATCH
from src.copilot import TOOLS, LakeTools, OfflineAnalyst, claude_available
from src.features import FEATURE_COLUMNS, GRAPH_FEATURES, FeatureEngine
from src.graph import detect_rings
from src.monitoring import build_reference_profile, compare_models, drift_report, psi
from src.simulator import TransactionSimulator

T0 = datetime(2026, 1, 1, 12, tzinfo=timezone.utc).timestamp()


def _cnp(card: str, device: str, ts: float, merchant: str = "m1") -> dict:
    return {"txn_id": f"{card}{device}{ts}", "event_time": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
            "card_id": card, "merchant_id": merchant, "merchant_name": "M", "merchant_category": "P2P Wallet",
            "txn_type": "P2P", "channel": "card_not_present", "amount": 40.0, "currency": "USD", "amount_usd": 40.0,
            "country": "USA", "city": "X", "lat": 40.7, "lon": -74.0, "device_id": device, "ip_country": "USA"}


@pytest.fixture()
def engine() -> FeatureEngine:
    cards = pd.DataFrame({"card_id": [f"c{i}" for i in range(6)], "home_country": "USA", "home_lat": 40.7,
                          "home_lon": -74.0, "profile_avg_usd": 40.0, "enrolled_device": [f"own{i}" for i in range(6)]})
    return FeatureEngine(cards)


# ------------------------------------------------------------------ graph features
def test_device_sharing_counts_other_cards_only_from_the_past(engine):
    f1 = engine.compute(_cnp("c0", "ringdev", T0), T0)
    f2 = engine.compute(_cnp("c1", "ringdev", T0 + 60), T0 + 60)
    f3 = engine.compute(_cnp("c2", "ringdev", T0 + 120), T0 + 120)
    f4 = engine.compute(_cnp("c0", "ringdev", T0 + 180), T0 + 180)
    assert [f1["device_cards_24h"], f2["device_cards_24h"], f3["device_cards_24h"]] == [0, 1, 2]
    assert f4["device_cards_24h"] == 2          # the card itself is never counted
    assert set(GRAPH_FEATURES) <= set(f1)


def test_device_links_expire_after_24h_and_terminals_are_ignored(engine):
    engine.compute(_cnp("c0", "dev", T0), T0)
    late = engine.compute(_cnp("c1", "dev", T0 + 90_000), T0 + 90_000)
    assert late["device_cards_24h"] == 0
    pos = dict(_cnp("c3", "term_m1", T0), channel="card_present", txn_type="POS")
    engine.compute(pos, T0)
    pos2 = dict(_cnp("c4", "term_m1", T0 + 5), channel="card_present", txn_type="POS")
    assert engine.compute(pos2, T0 + 5)["device_cards_24h"] == 0


def test_checkpoint_round_trips_graph_state(engine, tmp_path):
    engine.compute(_cnp("c0", "d", T0), T0)
    path = tmp_path / "state.pkl"
    engine.checkpoint(path)
    fresh = FeatureEngine(pd.DataFrame({"card_id": ["c0"], "home_country": "USA", "home_lat": 40.7, "home_lon": -74.0,
                                        "profile_avg_usd": 40.0, "enrolled_device": ["own0"]}))
    assert fresh.restore(path)
    assert "c0" in fresh.device_cards["d"]


# ------------------------------------------------------------------ ring detection
def _payments(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"txn_id": f"t{i}", "event_time": pd.Timestamp(T0 + i, unit="s", tz="UTC"), "card_id": c,
                          "device_id": d, "merchant_id": m, "merchant_name": m, "merchant_category": "P2P Wallet",
                          "amount_usd": 50.0, "home_country": "USA", "ip_country": "USA", "risk_score": 10.0,
                          "decision": "approve", "challenger_score": 80.0, "fraud_type": None}
                         for i, (c, d, m) in enumerate(rows)])


def test_ring_found_households_and_lone_devices_ignored():
    # five cards on two ring devices; r0 used both (as ring cards hop between the ring's devices)
    ring = [(f"r{i}", f"rd{i % 2}", "collector") for i in range(5)] + [("r0", "rd1", "collector")]
    household = [("h1", "family_tablet", "grocer"), ("h2", "family_tablet", "grocer")]
    lone = [("x1", "own_phone", "shop"), ("r0", "r0_phone", "shop")]   # a victim's own phone must not join the ring
    rings, members, edges = detect_rings(_payments(ring + household + lone))
    assert len(rings) == 1
    r = rings.iloc[0]
    assert r["n_cards"] == 5 and r["n_devices"] == 2
    assert r["collector_concentration"] == 1.0
    assert set(members["card_id"]) == {f"r{i}" for i in range(5)}
    assert "r0_phone" not in set(edges["device_id"])


def test_simulated_rings_are_recovered_with_high_precision():
    sim = TransactionSimulator(n_cards=3000, n_merchants=300, fraud_rate=0.0, ring_rate=0.01, malformed_rate=0.0,
                               late_rate=0.0, household_rate=0.05)
    cards, _ = sim.reference_frames()
    home = dict(zip(cards.card_id, cards.home_country))
    rows = []
    for ev, lb in sim.events_between(T0 - 6 * 3600, T0, 3):
        if ev["channel"] == "card_not_present":
            rows.append({**ev, "event_time": pd.Timestamp(ev["event_time"]), "home_country": home[ev["card_id"]],
                         "risk_score": 0.0, "decision": "approve", "challenger_score": np.nan,
                         "fraud_type": lb["fraud_type"] if lb else None})
    rings, members, _ = detect_rings(pd.DataFrame(rows))
    assert len(rings) >= 2
    precision = members["card_id"].isin(sim.ring_truth).mean()
    assert precision >= 0.95, precision
    labelled = pd.DataFrame(rows).query("fraud_type == 'fraud_ring'")
    assert labelled["device_id"].str.startswith("dev_rg_").all()


# ------------------------------------------------------------------ drift
def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 20_000)})
    prof = build_reference_profile(ref, ["x"])
    same = drift_report(pd.DataFrame({"x": rng.normal(0, 1, 5_000)}), prof).iloc[0]
    moved = drift_report(pd.DataFrame({"x": rng.normal(1.0, 1, 5_000)}), prof).iloc[0]
    assert same["psi"] < PSI_WATCH and same["status"] == "stable"
    assert moved["psi"] > PSI_DRIFT and moved["status"] == "drift"
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


# ------------------------------------------------------------------ champion / challenger
def _labelled(n: int, n_fraud: int, challenger_good: bool, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=bool)
    y[:n_fraud] = True
    champ = np.where(y & (np.arange(n) % 2 == 0), 90, rng.uniform(0, 60, n))      # champion finds half the fraud
    chall = np.where(y, 95, rng.uniform(0, 40, n)) if challenger_good else rng.uniform(0, 100, n)
    return pd.DataFrame({"is_fraud": y, "risk_score": champ, "challenger_score": chall,
                         "fraud_type": np.where(y, "fraud_ring", None)})


def test_promotion_requires_a_win_at_equal_workload():
    assert compare_models(_labelled(5000, 200, True))["verdict"] == "promote"
    assert compare_models(_labelled(5000, 200, False))["verdict"] == "retire_challenger"
    few = compare_models(_labelled(5000, 10, True))
    assert few["verdict"] == "insufficient_data"
    r = compare_models(_labelled(5000, 200, True))
    assert r["champion"]["flagged"] == r["challenger"]["flagged"]          # same analyst workload


def test_challenger_contract():
    assert CHALLENGER_FEATURES[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert set(GRAPH_FEATURES) <= set(CHALLENGER_FEATURES)
    assert ChallengerBundle.load(__import__("pathlib").Path("does/not/exist.joblib")) is None


# ------------------------------------------------------------------ copilot
def test_tool_schemas_are_strict_and_complete():
    names = {t["name"] for t in TOOLS}
    assert names == {"get_transaction", "get_card_history", "get_ring", "list_rings", "list_alerts", "get_model_health"}
    for t in TOOLS:
        s = t["input_schema"]
        assert t["strict"] and s["additionalProperties"] is False
        assert set(s["required"]) == set(s["properties"])


class FakeTools(LakeTools):
    def get_transaction(self, txn_id):
        return {"txn_id": txn_id, "card_id": "card_1", "txn_type": "P2P", "amount_usd": 120.0, "merchant_category": "P2P Wallet",
                "country": "USA", "risk_score": 12.0, "decision": "approve", "challenger_score": 91.0, "rules_triggered": "",
                "top_deviating_features": {"device_cards_24h": 6.0}, "device_cards_24h": 6, "card_devices_24h": 2,
                "confirmed_label": None}

    def get_card_history(self, card_id, limit=20):
        return {"transactions_shown": 5, "max_risk": 30.0, "declines": 0, "confirmed_fraud_labels": 0}

    def get_ring(self, ring_id=None, card_id=None):
        return {"ring": {"ring_id": "R-ABC123", "status": "high", "ring_score": 88.0, "n_cards": 9, "n_devices": 2,
                         "collector_concentration": 1.0}}


def test_offline_brief_escalates_ring_member_the_champion_approved():
    off = OfflineAnalyst(FakeTools())
    text = off.brief_txn("t1")
    assert "R-ABC123" in text and "Block card card_1" in text
    for heading in ("Summary", "Evidence", "Assessment", "Recommended action", "Confidence and gaps"):
        assert heading in text
    assert [c["tool"] for c in off.calls] == ["get_transaction", "get_card_history", "get_ring"]


def test_tool_errors_are_returned_not_raised():
    class Boom(LakeTools):
        def list_rings(self, limit=10):
            raise RuntimeError("lake offline")
    out = Boom().call("list_rings", {"limit": 3})
    assert "error" in out and "lake offline" in out["error"]
    assert "error" in LakeTools().call("drop_table", {})


def test_claude_mode_needs_credentials(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    assert claude_available() is False
