"""Command Center – live KPIs, alert ticker, decision stream, risk distribution and the drill-down feed."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import LakeEmpty, load_metrics, load_window, time_bucket, window_bounds
from dashboard.components.drill import feed_table, render_open_dialog
from dashboard.components.layout import controls, live_body, page_header
from dashboard.components.theme import (ACCENT, LABEL, STATUS, STATUS_ORDER, kpi, kpi_grid, money, num, pct, section,
                                        style, ticker)
from src.config import DECLINE_THRESHOLD, REVIEW_THRESHOLD


def body() -> None:
    f = controls().filters
    page_header("Command Center", "Real-time card authorisation risk monitoring")
    try:
        df = load_window(f)
    except LakeEmpty:
        st.info("Waiting for the first scored transactions… start the simulator and consumer (see README › Quick start).")
        return
    if df.empty:
        st.warning("No transactions match the current filters in this window.")
        return

    start, end = window_bounds(f)
    bucket = time_bucket(start, end)
    ts = (df.assign(declined_usd=df["amount_usd"].where(df["decision"] == "decline", 0.0))
          .set_index("event_time").groupby(pd.Grouper(freq=bucket))
          .agg(txns=("txn_id", "size"), flagged=("flagged", "sum"), fraud=("is_fraud", "sum"),
               risk=("risk_score", "mean"), volume=("amount_usd", "sum"), declined_usd=("declined_usd", "sum")))
    ts["flag_rate"] = ts["flagged"] / ts["txns"].where(ts["txns"] > 0)
    ts["fraud_rate"] = ts["fraud"] / ts["txns"].where(ts["txns"] > 0)

    # ---- KPI wall ---------------------------------------------------------------
    last60 = df[df["event_time"] >= end - pd.Timedelta(seconds=60)]
    tps = len(last60) / 60
    flagged = df[df["flagged"]]
    declined = df[df["decision"] == "decline"]
    fraud = df[df["is_fraud"]]
    caught = (fraud["flagged"].mean() if len(fraud) else float("nan"))
    precision = (flagged["is_fraud"].mean() if len(flagged) else float("nan"))
    m = load_metrics(30)
    p95 = m["latency_p95_ms"].iloc[-1] if not m.empty else float("nan")
    lag = int(m["lag_total"].iloc[-1]) if not m.empty and "lag_total" in m else 0
    spark = lambda col: ts[col].fillna(0).tail(40).tolist()  # noqa: E731
    kpi_grid([
        kpi("Transactions", num(len(df)), f"{money(df['amount_usd'].sum())} volume", "cyan", spark("txns")),
        kpi("Throughput", f"{tps:,.1f} TPS", "last 60 s of event time", "blue", spark("txns")),
        kpi("Flag rate", pct(len(flagged) / len(df), 2), f"risk ≥ {f.threshold:.0f} · {num(len(flagged))} txns",
            "amber", spark("flag_rate")),
        kpi("Blocked value", money(declined["amount_usd"].sum()), f"{num(len(declined))} auto-declines", "red",
            spark("declined_usd")),
        kpi("Fraud caught", pct(caught), f"of {num(len(fraud))} confirmed (labels delayed)", "green", None),
        kpi("Alert precision", pct(precision), "confirmed so far · labels lag 5–90 min (see Model Intel)", "violet", None),
        kpi("Avg risk", f"{df['risk_score'].mean():.1f}", "0–100 blended score", "cyan", spark("risk")),
        kpi("Latency p95", f"{p95:,.0f} ms" if p95 == p95 else "–", f"consumer lag {lag:,} msgs",
            "green" if p95 == p95 and p95 < 2000 else "amber", m["latency_p95_ms"].tail(40).tolist() if not m.empty else None),
    ])

    # ---- Alert ticker -----------------------------------------------------------
    alerts = df[df["decision"] != "approve"].nlargest(25, "event_time")
    ticker([{**r, "rules": r["rules_triggered"].replace("|", " + ")} for r in
            alerts[["decision", "amount_usd", "merchant_category", "city", "risk_score", "rules_triggered"]].to_dict("records")])

    # ---- Decision stream + fraud rates ------------------------------------------
    left, right = st.columns([3, 2])
    with left:
        # Approvals dwarf everything else, so the stream shows the actionable decisions on a linear
        # axis (stacking on a log axis would misrepresent the parts); volume lives in the KPI wall.
        dec = (df[df["decision"] != "approve"].set_index("event_time")
               .groupby([pd.Grouper(freq=bucket), "decision"], observed=True).size().rename("n").reset_index())
        fig = px.bar(dec, x="event_time", y="n", color="decision", color_discrete_map=STATUS,
                     category_orders={"decision": ["review", "decline"]})
        fig.update_traces(hovertemplate="%{x|%H:%M:%S} · %{fullData.name}: %{y:,}<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.08)
        fig.update_yaxes(title="decisions")
        fig.update_xaxes(title=None)
        st.plotly_chart(style(fig, f"Review & decline stream · {bucket} buckets", 360), key="cc_stream")
    with right:
        tr = ts.reset_index()
        fig = go.Figure()
        fig.add_scatter(x=tr["event_time"], y=tr["flag_rate"], name=f"Flag rate (risk ≥ {f.threshold:.0f})",
                        mode="lines", line=dict(color=STATUS["review"], width=2), fill="tozeroy",
                        fillcolor="rgba(245,158,11,0.10)", hovertemplate="%{x|%H:%M} · flagged %{y:.2%}<extra></extra>")
        fig.add_scatter(x=tr["event_time"], y=tr["fraud_rate"], name="Confirmed fraud rate", mode="lines",
                        line=dict(color=LABEL["Confirmed fraud"], width=2, dash="dot"),
                        hovertemplate="%{x|%H:%M} · confirmed %{y:.2%}<extra></extra>")
        fig.update_yaxes(tickformat=".1%", rangemode="tozero")
        st.plotly_chart(style(fig, "Flagged vs confirmed fraud rate", 360), key="cc_rates")

    # ---- Risk distribution · type mix · category exposure -------------------------
    a, b, c = st.columns(3)
    with a:
        fig = px.histogram(df, x="risk_score", nbins=50, color="decision", color_discrete_map=STATUS,
                           category_orders={"decision": STATUS_ORDER}, log_y=True)
        for x, col, lbl in ((REVIEW_THRESHOLD, STATUS["review"], "review"), (DECLINE_THRESHOLD, STATUS["decline"], "decline")):
            fig.add_vline(x=x, line_dash="dot", line_color=col, annotation_text=lbl, annotation_font_color=col)
        if f.threshold not in (REVIEW_THRESHOLD, DECLINE_THRESHOLD):
            fig.add_vline(x=f.threshold, line_color=ACCENT, annotation_text="what-if", annotation_font_color=ACCENT)
        fig.update_traces(hovertemplate="risk %{x}<br>%{y:,} txns<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.05)
        fig.update_xaxes(title="risk score")
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, "Risk score distribution (log)", 340), key="cc_hist")
    with b:
        mix = df.groupby(["txn_type", "decision"], observed=True).size().rename("n").reset_index()
        mix["share"] = mix["n"] / mix.groupby("txn_type", observed=True)["n"].transform("sum")
        mix = mix[mix["decision"] != "approve"]
        fig = px.bar(mix, x="share", y="txn_type", color="decision", orientation="h", color_discrete_map=STATUS,
                     category_orders={"decision": STATUS_ORDER})
        fig.update_traces(hovertemplate="%{y} · %{fullData.name}: %{x:.2%}<extra></extra>")
        fig.update_xaxes(tickformat=".0%", title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, "Review / decline rate by transaction type", 340), key="cc_type")
    with c:
        cat = (flagged.groupby("merchant_category", observed=True)["amount_usd"].sum().sort_values().tail(8).reset_index())
        fig = px.bar(cat, x="amount_usd", y="merchant_category", orientation="h")
        fig.update_traces(marker_color=STATUS["decline"], marker_opacity=0.85,
                          hovertemplate="%{y}: $%{x:,.0f} at risk<extra></extra>")
        fig.update_xaxes(tickprefix="$", tickformat="~s", title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, "Value at risk by merchant category", 340, legend=False), key="cc_cat")

    # ---- Live feed ----------------------------------------------------------------
    section("Live transaction feed", "▸")
    c1, c2, c3 = st.columns([1.2, 1.2, 3])
    only_flagged = c1.toggle("Flagged only", value=True, key="cc_only_flagged")
    n_rows = c2.selectbox("Rows", [50, 100, 250, 500], index=1, key="cc_rows")
    c3.markdown("<div class='caption-note' style='padding-top:.6rem'>Select a row to open the investigation panel.</div>",
                unsafe_allow_html=True)
    feed = (df[df["flagged"]] if only_flagged else df).nlargest(n_rows, "event_time")
    feed_table(feed, key="cc_feed")


live_body(body)
render_open_dialog()
