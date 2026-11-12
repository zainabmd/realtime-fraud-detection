"""
Fraud-ring detection on the card <-> device graph.

Per-transaction scoring sees one card at a time. Coordinated rings are invisible at that level:
each payment is an everyday amount, from the victim's own country, at an ordinary merchant. What
gives them away is **structure**: many unrelated cards driven from the same few devices, paying the
same one or two collector accounts.

Job (batch, every few minutes)
------------------------------
1. Take the last ``GRAPH_WINDOW_H`` hours of card-not-present payments from the lake.
2. Build a bipartite graph card -- *shared* device (used by >= 2 cards; store terminals excluded,
   every shopper shares those).
3. Connected components with >= ``RING_MIN_CARDS`` cards are candidate rings. Households (two cards of
   one customer on a family tablet) form 2-card components and fall below the threshold.
4. Score each candidate 0-100 from size, cards per device, collector-merchant concentration and the
   challenger model's view of its payments; persist rings, members and edges as Parquet snapshots.

    python -m src.graph                # one run
    python -m src.graph --watch 120    # refresh every 2 minutes (run_local starts this)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd

from src.config import GRAPH_DIR, GRAPH_WINDOW_H, RING_MIN_CARDS, get_logger
from src.utils import atomic_replace

log = get_logger("graph")

RINGS_PATH = GRAPH_DIR / "rings.parquet"
MEMBERS_PATH = GRAPH_DIR / "ring_members.parquet"
EDGES_PATH = GRAPH_DIR / "ring_edges.parquet"


def load_payments(hours: float = GRAPH_WINDOW_H) -> pd.DataFrame:
    from src.storage import query
    labels = "LEFT JOIN labels l USING (txn_id)"
    sql = f"""
        WITH w AS (SELECT * FROM scored WHERE channel = 'card_not_present'
                   AND event_time >= (SELECT max(event_time) FROM scored) - INTERVAL {int(hours * 60)} MINUTE)
        SELECT s.txn_id, s.event_time, s.card_id, s.device_id, s.merchant_id, s.merchant_name, s.merchant_category,
               s.amount_usd, s.home_country, s.ip_country, s.risk_score, s.decision, s.challenger_score,
               {{fraud}} FROM w s {{join}}"""
    try:
        return query(sql.format(fraud="l.fraud_type", join=labels))
    except Exception:  # noqa: BLE001 – no labels table yet
        return query(sql.format(fraud="NULL::VARCHAR AS fraud_type", join=""))


def ring_id(devices: list[str]) -> str:
    """Stable id across runs while the ring keeps its first device."""
    return "R-" + hashlib.md5(min(devices).encode()).hexdigest()[:6].upper()


def score_ring(n_cards: int, n_devices: int, concentration: float, challenger_mean: float) -> float:
    size = min(n_cards / 8.0, 1.0)                       # bigger clusters are less likely to be coincidence
    fan_in = min((n_cards / max(n_devices, 1)) / 4.0, 1.0)  # many cards per device = remote control
    model = 0.0 if np.isnan(challenger_mean) else challenger_mean / 100.0
    return round(100 * (0.35 * size + 0.35 * concentration + 0.20 * fan_in + 0.10 * model), 1)


def detect_rings(pay: pd.DataFrame, min_cards: int = RING_MIN_CARDS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (rings, members, edges). Pure function of the payments frame (unit-testable)."""
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if pay.empty:
        return empty
    pay = pay[~pay["device_id"].astype(str).str.startswith("term_")].copy()
    # Only *shared* devices link cards: a victim's own phone (degree 1) is not part of the ring's
    # infrastructure and would dilute its profile.
    edges = pay.groupby(["card_id", "device_id"]).size().reset_index(name="n_txns")
    shared = edges.groupby("device_id")["card_id"].transform("nunique") >= 2
    edges = edges[shared]
    g = nx.Graph()
    g.add_edges_from((("c:" + r.card_id, "d:" + r.device_id) for r in edges.itertuples(index=False)))
    rings, members, edge_rows = [], [], []
    for comp in nx.connected_components(g):
        cards = sorted(n[2:] for n in comp if n.startswith("c:"))
        if len(cards) < min_cards:
            continue
        devices = sorted(n[2:] for n in comp if n.startswith("d:"))
        rid = ring_id(devices)
        p = pay[pay["card_id"].isin(cards) & pay["device_id"].isin(devices)]
        top = p.groupby(["merchant_id", "merchant_name", "merchant_category"]).size().sort_values(ascending=False)
        (top_mid, top_name, top_cat), top_n = top.index[0], int(top.iloc[0])
        concentration = top.head(2).sum() / len(p)       # share paid to the two biggest collectors
        ch_mean = float(p["challenger_score"].mean()) if p["challenger_score"].notna().any() else float("nan")
        labelled = p["fraud_type"].notna()
        rings.append({
            "ring_id": rid, "n_cards": len(cards), "n_devices": len(devices), "n_txns": int(len(p)),
            "amount_usd": round(float(p["amount_usd"].sum()), 2), "first_seen": p["event_time"].min(),
            "last_seen": p["event_time"].max(), "top_merchant": top_name, "top_category": top_cat,
            "top_merchant_share": round(top_n / len(p), 3), "collector_concentration": round(float(concentration), 3),
            "home_countries": int(p["home_country"].nunique()), "champion_mean_risk": round(float(p["risk_score"].mean()), 1),
            "champion_flag_rate": round(float((p["decision"] != "approve").mean()), 3),
            "challenger_mean": round(ch_mean, 1) if ch_mean == ch_mean else None,
            "confirmed_fraud_share": round(float(labelled.mean()), 3),
            "ring_score": score_ring(len(cards), len(devices), float(concentration), ch_mean),
        })
        for card, grp in p.groupby("card_id"):
            members.append({"ring_id": rid, "card_id": card, "n_txns": int(len(grp)),
                            "amount_usd": round(float(grp["amount_usd"].sum()), 2), "first_seen": grp["event_time"].min(),
                            "home_country": grp["home_country"].iloc[0], "max_risk": float(grp["risk_score"].max()),
                            "max_challenger": float(grp["challenger_score"].max()) if grp["challenger_score"].notna().any() else None,
                            "confirmed_fraud": bool(grp["fraud_type"].notna().any())})
        for r in edges[edges["card_id"].isin(cards)].itertuples(index=False):
            edge_rows.append({"ring_id": rid, "card_id": r.card_id, "device_id": r.device_id, "n_txns": int(r.n_txns)})
    if not rings:
        return empty
    rings_df = pd.DataFrame(rings)
    rings_df["status"] = np.select([rings_df["ring_score"] >= 70, rings_df["ring_score"] >= 45], ["high", "medium"], "low")
    return rings_df.sort_values("ring_score", ascending=False), pd.DataFrame(members), pd.DataFrame(edge_rows)


def _write(df: pd.DataFrame, path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    atomic_replace(tmp, path)


def run_once(hours: float = GRAPH_WINDOW_H) -> dict:
    t0 = time.perf_counter()
    pay = load_payments(hours)
    rings, members, edges = detect_rings(pay)
    stamp = pd.Timestamp(datetime.now(timezone.utc))
    for df, path in ((rings, RINGS_PATH), (members, MEMBERS_PATH), (edges, EDGES_PATH)):
        _write(df.assign(detected_at=stamp) if not df.empty else pd.DataFrame({"detected_at": [stamp]}).iloc[:0], path)
    summary = {"payments": len(pay), "rings": len(rings), "cards_in_rings": int(rings["n_cards"].sum()) if len(rings) else 0,
               "high": int((rings["status"] == "high").sum()) if len(rings) else 0,
               "seconds": round(time.perf_counter() - t0, 2)}
    log.info("graph run: %s", summary)
    return summary


def load_rings() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Latest snapshot (empty frames before the first run)."""
    out = []
    for path in (RINGS_PATH, MEMBERS_PATH, EDGES_PATH):
        out.append(pd.read_parquet(path) if path.exists() else pd.DataFrame())
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fraud-ring detection on the card-device graph")
    ap.add_argument("--hours", type=float, default=GRAPH_WINDOW_H)
    ap.add_argument("--watch", type=float, default=None, metavar="SECONDS")
    args = ap.parse_args(argv)
    while True:
        try:
            run_once(args.hours)
        except Exception as exc:  # noqa: BLE001
            log.exception("graph run failed: %s", exc)
            if not args.watch:
                return 1
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
