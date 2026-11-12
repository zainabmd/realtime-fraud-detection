"""Pipeline Health – throughput, latency SLO, consumer lag, dead-letter queue, lake and API status."""
from __future__ import annotations

import json

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import dlq_counts, lake_stats, load_dlq, load_metrics
from dashboard.components.layout import controls, live_body, page_header
from dashboard.components.theme import ACCENT, PALETTE, SIGNAL_SCALE, STATUS, kpi, kpi_grid, num, section, style
from src.config import API_URL

LATENCY_SLO_MS = 2000


@st.cache_data(ttl=5, show_spinner=False)
def api_health() -> dict:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=1.5)
        return {"up": r.status_code == 200, **r.json()}
    except Exception as exc:  # noqa: BLE001 – the API is optional; report it as down
        return {"up": False, "error": type(exc).__name__}


def body() -> None:
    page_header("Pipeline Health", "Stream processing telemetry · per micro-batch")
    m = load_metrics(120)
    if m.empty:
        st.info("No pipeline metrics yet – start the consumer.")
        return
    m = m.sort_values("event_time")
    last = m.iloc[-1]
    recent = m[m["event_time"] >= last["event_time"] - pd.Timedelta(minutes=5)]
    eps = recent["valid_events"].sum() / max((recent["event_time"].max() - recent["event_time"].min()).total_seconds(), 1)
    slo = (recent["latency_p95_ms"] <= LATENCY_SLO_MS).mean()
    api = api_health()
    lake = lake_stats()
    age = (pd.Timestamp.now(tz="UTC") - last["event_time"]).total_seconds()

    kpi_grid([
        kpi("Ingest rate", f"{eps:,.1f} ev/s", "5-min average", "cyan", m["valid_events"].tail(60).tolist()),
        kpi("Latency p95", f"{last['latency_p95_ms']:,.0f} ms", f"p99 {last['latency_p99_ms']:,.0f} ms · SLO {LATENCY_SLO_MS} ms",
            "green" if last["latency_p95_ms"] <= LATENCY_SLO_MS else "red", m["latency_p95_ms"].tail(60).tolist()),
        kpi("SLO attainment", f"{slo:.1%}", "batches with p95 ≤ SLO (5 min)", "green" if slo >= 0.99 else "amber"),
        kpi("Consumer lag", f"{int(last['lag_total']):,}", "messages behind head", "green" if last["lag_total"] < 5000 else "amber",
            m["lag_total"].tail(60).tolist()),
        kpi("Dead letters", num(m["dlq_events"].sum()), "schema-invalid (window)", "red" if m["dlq_events"].sum() else "green"),
        kpi("Feature store", num(last["cards_in_state"]), "cards with live state", "violet"),
        kpi("Last batch", f"{age:,.0f}s ago" if age < 3600 else f"{age / 3600:,.1f}h ago",
            "consumer heartbeat", "green" if age < 30 else "amber"),
        kpi("Scoring API", "UP" if api.get("up") else "DOWN", api.get("model_version") or api.get("error", ""),
            "green" if api.get("up") else "red"),
    ])

    a, b = st.columns(2)
    with a:
        per = m.set_index("event_time").resample("10s").agg({"valid_events": "sum", "alerts": "sum"}).reset_index()
        per["eps"] = per["valid_events"] / 10
        fig = go.Figure()
        fig.add_scatter(x=per["event_time"], y=per["eps"], name="events / s", mode="lines", fill="tozeroy",
                        line=dict(color=ACCENT, width=2), fillcolor="rgba(34,211,238,0.12)",
                        hovertemplate="%{x|%H:%M:%S} · %{y:,.1f} ev/s<extra></extra>")
        fig.update_yaxes(title=None, rangemode="tozero")
        st.plotly_chart(style(fig, "Throughput (10 s buckets)", 320, legend=False), key="ph_tput")
    with b:
        fig = go.Figure()
        for col, color, name in (("latency_p50_ms", STATUS["approve"], "p50"), ("latency_p95_ms", STATUS["review"], "p95"),
                                 ("latency_p99_ms", STATUS["decline"], "p99")):
            fig.add_scatter(x=m["event_time"], y=m[col], name=name, mode="lines", line=dict(color=color, width=1.8),
                            hovertemplate=f"{name}: %{{y:,.0f}} ms<extra></extra>")
        fig.add_hline(y=LATENCY_SLO_MS, line_dash="dot", line_color="#e6edf7", annotation_text="SLO",
                      annotation_font_color="#e6edf7")
        fig.update_yaxes(type="log", title="ms (broker → scored)")
        st.plotly_chart(style(fig, "End-to-end pipeline latency", 320), key="ph_lat")

    a, b, c = st.columns(3)
    with a:
        fig = go.Figure(go.Scatter(x=m["event_time"], y=m["lag_total"], mode="lines", fill="tozeroy",
                                   line=dict(color="#9085e9", width=2), fillcolor="rgba(144,133,233,0.15)",
                                   hovertemplate="%{x|%H:%M:%S} · lag %{y:,}<extra></extra>"))
        st.plotly_chart(style(fig, "Consumer lag (messages)", 300, legend=False), key="ph_lag")
    with b:
        try:
            lp = json.loads(last["lag_by_partition"])
        except (TypeError, ValueError):
            lp = {}
        lpd = pd.DataFrame({"partition": [f"p{k}" for k in lp], "lag": list(lp.values())})
        fig = px.bar(lpd, x="partition", y="lag", color="partition", color_discrete_sequence=PALETTE)
        fig.update_traces(hovertemplate="%{x}: %{y:,} behind<extra></extra>")
        st.plotly_chart(style(fig, "Lag by partition (latest)", 300, legend=False), key="ph_part")
    with c:
        fig = px.histogram(m, x="processing_ms", nbins=40, color_discrete_sequence=[ACCENT])
        fig.update_traces(hovertemplate="%{x} ms · %{y} batches<extra></extra>")
        fig.update_xaxes(title="batch processing time (ms)")
        st.plotly_chart(style(fig, "Micro-batch processing time", 300, legend=False), key="ph_proc")

    section("Dead-letter queue", "▸")
    a, b = st.columns([1, 2])
    with a:
        dc = dlq_counts()
        if dc.empty:
            st.success("No dead letters. Every event passed schema validation.")
        else:
            fig = px.bar(dc, x="n", y="error_type", orientation="h", color="n", color_continuous_scale=SIGNAL_SCALE)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(hovertemplate="%{y}: %{x:,}<extra></extra>")
            fig.update_yaxes(title=None, categoryorder="total ascending")
            fig.update_xaxes(title=None)
            st.plotly_chart(style(fig, "Rejections by field", 300, legend=False), key="ph_dlq")
    with b:
        st.dataframe(load_dlq(200), hide_index=True, height=300, column_config={
            "received_at": "Received", "error": "Validation error",
            "raw": st.column_config.TextColumn("Raw payload", width="large")})

    section("Lakehouse storage", "▸")
    st.dataframe(pd.DataFrame(lake).T.reset_index(names="table"), hide_index=True, column_config={
        "files": st.column_config.NumberColumn("Parquet files"), "mb": st.column_config.NumberColumn("Size (MB)", format="%.2f"),
        "partitions": st.column_config.NumberColumn("dt/hr partitions")})
    st.caption("Micro-batches land every few seconds; the consumer compacts closed partitions every 5 minutes "
               "(merging files and de-duplicating replays by txn_id).")


live_body(body)
