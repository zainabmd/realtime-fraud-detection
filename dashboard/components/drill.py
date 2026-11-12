"""
Transaction drill-down modal, shared by every page with a transaction table.

Flow (robust to auto-refresh):
    table row selected inside the live fragment
      -> `remember_selection()` maps the row to a txn record, stores it in
         session state and triggers ONE full-app rerun
      -> the page calls `render_open_dialog()` *outside* the fragment, so the
         modal stays open and stable while the fragment keeps refreshing
      -> closing the modal (on_dismiss) clears the stored record.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import card_history
from dashboard.components.theme import ACCENT, INK_2, STATUS, kpi, kpi_grid, money, style
from src.detection import RULE_INDEX

FEED_COLS = ["event_time", "card_id", "merchant_category", "city", "txn_type", "amount_usd", "currency", "risk_score",
             "decision", "rules_triggered"]


def feed_table(df: pd.DataFrame, key: str, height: int = 420) -> None:
    """Render a selectable transaction table and capture the clicked row."""
    view = df[FEED_COLS + ["txn_id"]].reset_index(drop=True)
    ev = st.dataframe(
        view, key=key, height=height, hide_index=True, on_select="rerun", selection_mode="single-row",
        column_order=FEED_COLS,
        column_config={
            "event_time": st.column_config.DatetimeColumn("Time (UTC)", format="HH:mm:ss"),
            "card_id": "Card", "merchant_category": "Merchant", "city": "City", "txn_type": "Type",
            "amount_usd": st.column_config.NumberColumn("Amount", format="$%,.2f"),
            "currency": "Ccy",
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
            "decision": "Decision", "rules_triggered": "Rules fired",
        })
    rows = ev.selection.rows if ev is not None else []
    if rows:
        txn_id = view.iloc[rows[0]]["txn_id"]
        if st.session_state.get(f"{key}_last") != txn_id:
            st.session_state[f"{key}_last"] = txn_id
            st.session_state["open_txn"] = df.loc[df["txn_id"] == txn_id].iloc[0].to_dict()
            st.rerun()  # full-app rerun so the modal opens outside the refreshing fragment


def _close() -> None:
    st.session_state.pop("open_txn", None)


@st.dialog("Transaction investigation", width="large", on_dismiss=_close)
def _txn_dialog(r: dict) -> None:
    decision = str(r["decision"])
    st.markdown(
        f"<span class='badge {decision}'>{decision.upper()}</span> &nbsp; "
        f"<code>{r['txn_id']}</code> · {pd.Timestamp(r['event_time']):%d %b %Y %H:%M:%S} UTC",
        unsafe_allow_html=True)
    tone = {"approve": "green", "review": "amber", "decline": "red"}[decision]
    kpi_grid([
        kpi("Risk score", f"{r['risk_score']:.0f}/100", f"IF {r['if_points']:.0f} + rules {r['rule_points']:.0f}", tone),
        kpi("Amount", money(r["amount_usd"]), f"{r['amount']:,.2f} {r['currency']}", "cyan"),
        kpi("Merchant", str(r["merchant_category"]), str(r.get("merchant_name", "")), "violet"),
        kpi("Location", f"{r['city']}", f"home {r['home_country']} · {r['km_from_home']:,.0f} km away", "blue"),
    ])
    left, right = st.columns([1.1, 1])
    with left:
        # Home -> transaction location arc
        fig = go.Figure()
        fig.add_scattergeo(lat=[r["home_lat"], r["lat"]], lon=[r["home_lon"], r["lon"]], mode="lines",
                           line=dict(width=2, color=STATUS[decision]), hoverinfo="skip", showlegend=False)
        fig.add_scattergeo(lat=[r["home_lat"]], lon=[r["home_lon"]], mode="markers+text", text=["home"],
                           textposition="bottom center", marker=dict(size=10, color=ACCENT), textfont=dict(color=INK_2),
                           showlegend=False)
        fig.add_scattergeo(lat=[r["lat"]], lon=[r["lon"]], mode="markers+text", text=[r["city"]],
                           textposition="top center", marker=dict(size=12, color=STATUS[decision], symbol="diamond"),
                           textfont=dict(color=INK_2), showlegend=False)
        # Frame the arc: zoom in for local purchases, out for intercontinental ones.
        dist = max(float(r.get("km_from_home") or 0), 1.0)
        fig.update_geos(projection_type="natural earth", fitbounds=False,
                        center=dict(lat=(r["home_lat"] + r["lat"]) / 2, lon=(r["home_lon"] + r["lon"]) / 2),
                        projection_scale=float(min(8.0, max(1.0, 6000 / (dist + 800)))))
        st.plotly_chart(style(fig, "Cardholder home → transaction", 300, legend=False), key="dlg_geo")
    with right:
        # Risk composition
        fig = go.Figure()
        fig.add_bar(y=["Risk"], x=[r["if_points"]], orientation="h", name="Isolation Forest", marker_color="#9085e9",
                    hovertemplate="IF: %{x:.1f} pts<extra></extra>")
        fig.add_bar(y=["Risk"], x=[min(r["rule_points"], 70)], orientation="h", name="Rules", marker_color="#d95926",
                    hovertemplate="Rules: %{x:.1f} pts<extra></extra>")
        fig.update_layout(barmode="stack", xaxis=dict(range=[0, 130], title="points"))
        fig.add_vline(x=50, line_dash="dot", line_color=STATUS["review"])
        fig.add_vline(x=80, line_dash="dot", line_color=STATUS["decline"])
        st.plotly_chart(style(fig, "Score composition", 150), key="dlg_comp")
        fired = [x for x in str(r.get("rules_triggered") or "").split("|") if x]
        if fired:
            for code in fired:
                rule = RULE_INDEX.get(code)
                st.markdown(f"🔺 **{code}** · +{rule.points:.0f} — {rule.description}" if rule else f"🔺 {code}")
        else:
            st.markdown("No rule fired — score driven by the anomaly model.")

    try:
        top = json.loads(r.get("top_features") or "{}")
    except (TypeError, json.JSONDecodeError):
        top = {}
    if top:
        tf = pd.DataFrame({"feature": list(top), "z": list(top.values())}).sort_values("z")
        fig = go.Figure(go.Bar(x=tf["z"], y=tf["feature"], orientation="h",
                               marker_color=[STATUS["decline"] if v > 0 else ACCENT for v in tf["z"]],
                               hovertemplate="%{y}: %{x:+.1f} robust-σ<extra></extra>"))
        fig.update_xaxes(title="deviation from typical (robust z)")
        st.plotly_chart(style(fig, "Why: most unusual features", 200, legend=False), key="dlg_feat")

    hist = card_history(r["card_id"], 60)
    if not hist.empty:
        hist = hist.sort_values("event_time")
        fig = go.Figure()
        for d in ("approve", "review", "decline"):
            h = hist[hist["decision"] == d]
            fig.add_scatter(x=h["event_time"], y=h["amount_usd"], mode="markers", name=d,
                            marker=dict(color=STATUS[d], size=9, line=dict(width=1, color="#070d1a")),
                            customdata=h[["merchant_category", "city", "risk_score"]],
                            hovertemplate="%{x|%H:%M:%S} · $%{y:,.2f}<br>%{customdata[0]} · %{customdata[1]}"
                                          "<br>risk %{customdata[2]:.0f}<extra></extra>")
        fig.add_scatter(x=[pd.Timestamp(r["event_time"])], y=[r["amount_usd"]], mode="markers", name="this txn",
                        marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=2, color=ACCENT)))
        fig.update_yaxes(type="log", title="USD")
        st.plotly_chart(style(fig, f"Card {r['card_id']} · last {len(hist)} transactions", 280), key="dlg_hist")
        confirmed = int(hist["fraud_type"].notna().sum())
        if confirmed:
            st.error(f"⚠️ {confirmed} transaction(s) on this card are confirmed fraud "
                     f"({', '.join(sorted(hist['fraud_type'].dropna().unique()))}).")


def render_open_dialog() -> None:
    r = st.session_state.get("open_txn")
    if r:
        _txn_dialog(r)
