"""Investigations – analyst workspace: search a card or transaction, review its full timeline and trail."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import LakeEmpty, card_history, load_window
from dashboard.components.drill import feed_table, render_open_dialog
from dashboard.components.layout import controls, page_header
from dashboard.components.theme import ACCENT, STATUS, STATUS_ORDER, kpi, kpi_grid, money, num, pct, section, style

page_header("Investigations", "Analyst workspace · card timelines, travel trails and case queues")
f = controls().filters
try:
    df = load_window(f)
except LakeEmpty:
    st.info("No data yet.")
    st.stop()

# ---- Case queue: cards ranked by aggregated risk in the window ------------------------
section("Case queue · riskiest cards in window", "▸")
if df.empty:
    st.warning("No transactions match the current filters.")
    st.stop()
q = (df.assign(flag_usd=df["amount_usd"].where(df["flagged"], 0.0))
     .groupby("card_id").agg(txns=("txn_id", "size"), max_risk=("risk_score", "max"), flagged=("flagged", "sum"),
                                exposure=("flag_usd", "sum"),
                                countries=("country", "nunique"), confirmed=("is_fraud", "sum"),
                                last_seen=("event_time", "max"))
     .query("flagged > 0").sort_values(["max_risk", "exposure"], ascending=False).head(200).reset_index())
ev = st.dataframe(q, hide_index=True, height=300, on_select="rerun", selection_mode="single-row", key="inv_queue",
                  column_config={
                      "card_id": "Card", "txns": "Txns", "flagged": "Flagged",
                      "max_risk": st.column_config.ProgressColumn("Max risk", min_value=0, max_value=100, format="%.0f"),
                      "exposure": st.column_config.NumberColumn("Flagged $", format="$%,.0f"),
                      "countries": "Countries", "confirmed": "Confirmed fraud",
                      "last_seen": st.column_config.DatetimeColumn("Last seen", format="HH:mm:ss")})
picked = q.iloc[ev.selection.rows[0]]["card_id"] if ev.selection.rows else None

c1, c2 = st.columns([2, 3])
card_id = c1.text_input("Card ID", value=picked or (q.iloc[0]["card_id"] if len(q) else ""), key="inv_card",
                        help="Pick a row above or type any card id, e.g. card_000123").strip()
txn_lookup = c2.text_input("…or jump to a transaction ID", key="inv_txn").strip()
if txn_lookup:
    hit = df[df["txn_id"] == txn_lookup]
    if hit.empty:
        st.warning("Transaction not found in the current window.")
    else:
        st.session_state["open_txn"] = hit.iloc[0].to_dict()

hist = card_history(card_id, 500) if card_id else pd.DataFrame()
if hist.empty:
    st.info("Select a card from the queue or enter a card id.")
    render_open_dialog()
    st.stop()

hist = hist.sort_values("event_time")
section(f"Card {card_id}", "◆")
flag = hist["risk_score"] >= f.threshold
kpi_grid([
    kpi("Transactions", num(len(hist)), f"{hist['event_time'].min():%d %b %H:%M} → {hist['event_time'].max():%H:%M}", "cyan"),
    kpi("Flagged", num(int(flag.sum())), pct(flag.mean()), "amber"),
    kpi("Spend", money(hist["amount_usd"].sum()), f"median {money(hist['amount_usd'].median())}", "blue"),
    kpi("Countries", f"{hist['country'].nunique()}", ", ".join(map(str, hist["country"].unique()[:5])), "violet"),
    kpi("Devices", f"{hist['device_id'].nunique()}", "distinct devices / terminals", "violet"),
    kpi("Confirmed fraud", num(int(hist["fraud_type"].notna().sum())),
        ", ".join(sorted(hist["fraud_type"].dropna().unique())) or "none reported", "red" if hist["fraud_type"].notna().any() else "green"),
])

a, b = st.columns([3, 2])
with a:
    fig = go.Figure()
    for d in STATUS_ORDER:
        h = hist[hist["decision"] == d]
        fig.add_scatter(x=h["event_time"], y=h["amount_usd"], mode="markers", name=d,
                        marker=dict(color=STATUS[d], size=10, line=dict(width=1, color="#070d1a")),
                        customdata=h[["merchant_category", "city", "risk_score", "rules_triggered"]],
                        hovertemplate="%{x|%d %b %H:%M:%S} · $%{y:,.2f}<br>%{customdata[0]} · %{customdata[1]}"
                                      "<br>risk %{customdata[2]:.0f} · %{customdata[3]}<extra></extra>")
    fr = hist[hist["fraud_type"].notna()]
    fig.add_scatter(x=fr["event_time"], y=fr["amount_usd"], mode="markers", name="confirmed fraud",
                    marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=2, color="#f43f5e")))
    fig.update_yaxes(type="log", tickprefix="$")
    st.plotly_chart(style(fig, "Timeline · amount (log) by decision", 380), key="inv_tl")
with b:
    trail = hist[["lat", "lon", "city", "event_time", "decision"]]
    fig = go.Figure()
    fig.add_scattergeo(lat=trail["lat"], lon=trail["lon"], mode="lines", line=dict(width=1.5, color=ACCENT),
                       opacity=0.6, hoverinfo="skip", showlegend=False)
    for d in STATUS_ORDER:
        t = trail[trail["decision"] == d]
        fig.add_scattergeo(lat=t["lat"], lon=t["lon"], mode="markers", name=d,
                           marker=dict(size=9, color=STATUS[d], line=dict(width=1, color="#070d1a")),
                           text=t["city"] + " · " + t["event_time"].dt.strftime("%H:%M"), hoverinfo="text")
    st.plotly_chart(style(fig, "Travel trail", 380), key="inv_trail")

risk_t = px.line(hist, x="event_time", y="risk_score", markers=True)
risk_t.update_traces(line_color="#9085e9", marker=dict(size=6),
                     hovertemplate="%{x|%H:%M:%S} · risk %{y:.0f}<extra></extra>")
risk_t.add_hrect(y0=f.threshold, y1=100, fillcolor=STATUS["decline"], opacity=0.07, line_width=0)
risk_t.update_yaxes(range=[0, 102])
st.plotly_chart(style(risk_t, "Risk score per transaction", 240, legend=False), key="inv_risk")

section("All transactions on this card · select to investigate", "▸")
feed_table(hist.sort_values("event_time", ascending=False), key="inv_feed", height=360)
render_open_dialog()
