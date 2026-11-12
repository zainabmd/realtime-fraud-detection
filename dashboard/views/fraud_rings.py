"""Fraud Rings – coordinated attacks found on the card <-> shared-device graph."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import rings_snapshot
from dashboard.components.layout import live_body, page_header
from dashboard.components.theme import (ACCENT, INK_2, LABEL, PALETTE, STATUS, kpi, kpi_grid, money, num, pct,
                                        section, style)

STATUS_TONE = {"high": STATUS["decline"], "medium": STATUS["review"], "low": "#64748b"}


def network(rings: pd.DataFrame, members: pd.DataFrame, edges: pd.DataFrame, ring_ids: list[str]) -> go.Figure:
    """Force-directed card (circle) <-> shared device (diamond) graph for the selected rings."""
    e = edges[edges["ring_id"].isin(ring_ids)]
    g = nx.Graph()
    g.add_edges_from(("c:" + r.card_id, "d:" + r.device_id) for r in e.itertuples(index=False))
    # Lay out each ring on its own grid cell: one global spring layout pushes disconnected rings apart and
    # shrinks every ring to an unreadable dot.
    pos, cols = {}, min(3, len(ring_ids))
    for i, rid in enumerate(ring_ids):
        sub = g.subgraph({"c:" + c for c in e.loc[e["ring_id"] == rid, "card_id"]} |
                         {"d:" + d for d in e.loc[e["ring_id"] == rid, "device_id"]})
        local = nx.spring_layout(sub, seed=7, k=1.2 / max(1, np.sqrt(len(sub))), iterations=100)
        ox, oy = (i % cols) * 3.0, -(i // cols) * 3.0
        pos.update({n: (ox + x, oy + y) for n, (x, y) in local.items()})
    fig = go.Figure()
    xs, ys = [], []
    for a, b in g.edges():
        xs += [pos[a][0], pos[b][0], None]
        ys += [pos[a][1], pos[b][1], None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(148,163,184,0.35)", width=1), hoverinfo="skip",
                    showlegend=False)
    m = members.set_index("card_id")
    ring_of = dict(zip(e["card_id"], e["ring_id"]))
    color_of = {rid: PALETTE[i % len(PALETTE)] for i, rid in enumerate(ring_ids)}
    cards = [n for n in g if n.startswith("c:")]
    devs = [n for n in g if n.startswith("d:")]
    fig.add_scatter(
        x=[pos[n][0] for n in cards], y=[pos[n][1] for n in cards], mode="markers", name="Card",
        marker=dict(size=[9 + min(14, m.loc[n[2:], "n_txns"] * 2) if n[2:] in m.index else 10 for n in cards],
                    color=[color_of[ring_of[n[2:]]] for n in cards],
                    line=dict(width=[2.5 if n[2:] in m.index and m.loc[n[2:], "confirmed_fraud"] else 0 for n in cards],
                              color=LABEL["Confirmed fraud"])),
        customdata=[[n[2:], ring_of[n[2:]], m.loc[n[2:], "amount_usd"] if n[2:] in m.index else 0,
                     m.loc[n[2:], "max_risk"] if n[2:] in m.index else 0] for n in cards],
        hovertemplate="%{customdata[0]}<br>ring %{customdata[1]}<br>$%{customdata[2]:,.0f} paid<br>"
                      "max champion risk %{customdata[3]:.0f}<extra></extra>")
    fig.add_scatter(x=[pos[n][0] for n in devs], y=[pos[n][1] for n in devs], mode="markers", name="Shared device",
                    marker=dict(symbol="diamond", size=17, color=ACCENT, line=dict(width=1.5, color="#e6edf7")),
                    text=[n[2:] for n in devs], hovertemplate="device %{text}<extra></extra>")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig


def body() -> None:
    page_header("Fraud Rings", "Cards driven from shared devices · card-device graph over the last 24 h, refreshed every 2 min")
    rings, members, edges = rings_snapshot()
    if rings.empty:
        st.info("No rings in the latest snapshot. The graph job (`python -m src.graph --watch 120`) writes one every "
                "2 minutes once card-not-present traffic has accumulated.")
        return
    detected = pd.Timestamp(rings["detected_at"].iloc[0])
    high = rings[rings["status"] == "high"]
    champ_rate = (rings["champion_flag_rate"] * rings["n_txns"]).sum() / rings["n_txns"].sum()
    ch = rings["challenger_mean"].dropna()
    kpi_grid([
        kpi("Rings detected", num(len(rings)), f"{len(high)} high · snapshot {detected:%H:%M} UTC", "red"),
        kpi("Cards in rings", num(int(rings["n_cards"].sum())), f"on {int(rings['n_devices'].sum())} shared devices", "violet"),
        kpi("Ring payments", money(rings["amount_usd"].sum()), f"{num(int(rings['n_txns'].sum()))} transactions", "amber"),
        kpi("Champion flagged", pct(champ_rate), "of ring payments: rules see each card alone", "red" if champ_rate < .3 else "green"),
        kpi("Challenger mean", f"{ch.mean():.0f}/100" if len(ch) else "–", "shadow model with graph features", "cyan"),
        kpi("Confirmed fraud", pct((rings["confirmed_fraud_share"] * rings["n_txns"]).sum() / rings["n_txns"].sum()),
            "of ring payments (labels lag 5-90 min)", "green"),
    ])

    left, right = st.columns([3, 2])
    with left:
        options = rings["ring_id"].tolist()
        chosen = st.multiselect("Rings on the map", options, default=options[: min(6, len(options))], key="rings_map",
                                format_func=lambda r: f"{r} · {rings.set_index('ring_id').loc[r, 'n_cards']} cards")
        if chosen:
            st.plotly_chart(style(network(rings, members, edges, chosen),
                                  "Card ↔ shared-device network (red ring = confirmed fraud label)", 520), key="rings_net")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=rings["champion_flag_rate"], y=rings["ring_score"], mode="markers", name="Ring",
                        marker=dict(size=8 + rings["n_cards"] * 2, color=[STATUS_TONE[s] for s in rings["status"]],
                                    line=dict(width=1, color="#0b1424"), opacity=.85),
                        customdata=rings[["ring_id", "n_cards", "amount_usd"]],
                        hovertemplate="%{customdata[0]}: %{customdata[1]} cards, $%{customdata[2]:,.0f}<br>"
                                      "champion flagged %{x:.0%} · ring score %{y:.0f}<extra></extra>")
        fig.update_xaxes(title="share of the ring's payments the champion flagged", tickformat=".0%", range=[-0.05, 1.05])
        fig.update_yaxes(title="ring score (graph)", range=[0, 100])
        st.plotly_chart(style(fig, "Strong rings the champion barely flags (top-left)", 520, legend=False), key="rings_blind")

    section("Ring league table", "▸")
    show = rings[["ring_id", "status", "ring_score", "n_cards", "n_devices", "n_txns", "amount_usd", "top_category",
                  "collector_concentration", "home_countries", "champion_flag_rate", "challenger_mean",
                  "confirmed_fraud_share", "first_seen", "last_seen"]]
    st.dataframe(show, hide_index=True, height=330, column_config={
        "ring_score": st.column_config.ProgressColumn("Ring score", min_value=0, max_value=100, format="%.0f"),
        "amount_usd": st.column_config.NumberColumn("Paid out", format="$%.0f"),
        "collector_concentration": st.column_config.ProgressColumn("To top collectors", min_value=0, max_value=1, format="percent"),
        "champion_flag_rate": st.column_config.NumberColumn("Champion flagged", format="percent"),
        "challenger_mean": st.column_config.NumberColumn("Challenger mean", format="%.0f"),
        "confirmed_fraud_share": st.column_config.NumberColumn("Confirmed", format="percent"),
        "first_seen": st.column_config.DatetimeColumn(format="HH:mm"), "last_seen": st.column_config.DatetimeColumn(format="HH:mm")})

    section("Investigate a ring", "▸")
    a, b = st.columns([2, 3])
    with a:
        rid = st.selectbox("Ring", rings["ring_id"], key="rings_pick")
        mem = members[members["ring_id"] == rid].sort_values("amount_usd", ascending=False)
        st.dataframe(mem[["card_id", "n_txns", "amount_usd", "home_country", "max_risk", "max_challenger", "confirmed_fraud"]],
                     hide_index=True, height=300, column_config={
                         "amount_usd": st.column_config.NumberColumn("Paid", format="$%.0f"),
                         "max_risk": st.column_config.NumberColumn("Champion max", format="%.0f"),
                         "max_challenger": st.column_config.NumberColumn("Challenger max", format="%.0f")})
    with b:
        if st.button(f"🤖 Brief ring {rid} with the copilot", key="rings_brief", width="stretch"):
            from src.copilot import run
            with st.spinner("Gathering evidence…"):
                st.session_state["ring_brief"] = (rid, run("ring", rid).to_dict())
        got = st.session_state.get("ring_brief")
        if got and got[0] == rid:
            ans = got[1]
            st.caption(f"{'Claude · ' + ans['model'] if ans['mode'] == 'claude' else 'Offline analyst'} · "
                       f"{len(ans['tool_calls'])} tool calls" + (f" · {ans['note']}" if ans.get("note") else ""))
            st.markdown(ans["text"])
        else:
            st.markdown(f"<span style='color:{INK_2}'>Ask the copilot for a case brief: what the ring is, how much it "
                        "moved, which devices and collectors to block.</span>", unsafe_allow_html=True)


live_body(body)
