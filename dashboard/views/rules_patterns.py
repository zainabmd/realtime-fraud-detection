"""Rules & Patterns – rule effectiveness, co-firing, temporal and behavioural attack signatures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import LakeEmpty, load_window, split_rules, time_bucket, window_bounds
from dashboard.components.layout import controls, live_body, page_header
from dashboard.components.theme import (FRAUD_TYPE_COLORS, LABEL, PALETTE, STATUS, STATUS_ORDER, THREAT_SCALE, kpi,
                                        kpi_grid, num, pct, style)
from src.detection import RULES

RULE_COLORS = {r.code: PALETTE[i % len(PALETTE)] for i, r in enumerate(RULES)}


def body() -> None:
    f = controls().filters
    page_header("Rules & Attack Patterns", "Which signals fire, how precise they are, and what attacks look like")
    try:
        df = load_window(f)
    except LakeEmpty:
        st.info("No data yet.")
        return
    if df.empty:
        st.warning("No transactions match the current filters.")
        return

    hits = split_rules(df)
    stats = (hits.groupby("rule").agg(hits=("txn_id", "size"), precision=("is_fraud", "mean")).reindex(
        [r.code for r in RULES]).fillna({"hits": 0}).reset_index())
    stats["hit_rate"] = stats["hits"] / len(df)
    any_rule = (df["rules_triggered"].fillna("") != "")
    best = stats.dropna(subset=["precision"]).sort_values("precision", ascending=False)
    kpi_grid([
        kpi("Rule hits", num(len(hits)), f"{pct(any_rule.mean(), 2)} of txns fire ≥ 1 rule", "amber"),
        kpi("Most precise rule", best.iloc[0]["rule"] if len(best) else "–",
            f"{pct(best.iloc[0]['precision'])} confirmed" if len(best) else "", "green"),
        kpi("Model-only flags", num(int((df["flagged"] & ~any_rule).sum())), "flagged with no rule → novel patterns", "violet"),
        kpi("Confirmed patterns", f"{df['fraud_type'].nunique()}", f"{num(int(df['is_fraud'].sum()))} labelled frauds", "red"),
    ])

    # ---- Rule volume vs precision (two charts, never a dual axis) -----------------
    a, b = st.columns(2)
    with a:
        s = stats.sort_values("hits")
        fig = px.bar(s, x="hits", y="rule", orientation="h", color="rule", color_discrete_map=RULE_COLORS)
        fig.update_traces(hovertemplate="%{y}: %{x:,} hits<extra></extra>")
        fig.update_xaxes(title=None, type="log")
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, "Rule volume (log)", 340, legend=False), key="rp_vol")
    with b:
        s = stats.sort_values("hits")
        fig = px.bar(s, x="precision", y="rule", orientation="h", color="rule", color_discrete_map=RULE_COLORS)
        fig.update_traces(hovertemplate="%{y}: %{x:.1%} of hits are confirmed fraud<extra></extra>")
        fig.update_xaxes(tickformat=".0%", range=[0, 1], title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, "Rule precision (confirmed fraud share)", 340, legend=False), key="rp_prec")

    # ---- Co-firing heatmap + fraud pattern timeline ------------------------------------
    a, b = st.columns([1, 1.3])
    with a:
        codes = [r.code for r in RULES]
        onehot = pd.DataFrame({c: df["rules_triggered"].fillna("").str.contains(c, regex=False) for c in codes}).astype(int)
        co = onehot.T @ onehot
        diag = np.diag(co).astype(float)
        jacc = co / (diag[:, None] + diag[None, :] - co).replace(0, np.nan)
        fig = px.imshow(jacc.round(3), color_continuous_scale=THREAT_SCALE, zmin=0, zmax=1, aspect="auto",
                        labels=dict(color="Jaccard"))
        fig.update_traces(xgap=2, ygap=2, hovertemplate="%{y} & %{x}<br>overlap %{z:.2f}<extra></extra>")
        st.plotly_chart(style(fig, "Rule co-firing (Jaccard overlap)", 400, legend=False), key="rp_co")
    with b:
        start, end = window_bounds(f)
        bucket = time_bucket(start, end) if (end - start) > pd.Timedelta(hours=2) else "5min"
        ft = (df[df["is_fraud"]].set_index("event_time").groupby([pd.Grouper(freq=bucket), "fraud_type"]).size()
              .rename("n").reset_index())
        fig = px.area(ft, x="event_time", y="n", color="fraud_type", color_discrete_map=FRAUD_TYPE_COLORS)
        fig.update_traces(line=dict(width=1), hovertemplate="%{x|%H:%M} · %{fullData.name}: %{y}<extra></extra>")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="confirmed fraud txns")
        st.plotly_chart(style(fig, "Confirmed attack patterns over time", 400), key="rp_types")

    # ---- Temporal + behavioural signatures ----------------------------------------------
    a, b = st.columns(2)
    with a:
        hh = df.assign(hour=df["event_time"].dt.hour)
        piv = hh.pivot_table(index="merchant_category", columns="hour", values="flagged", aggfunc="mean", observed=True)
        piv = piv.loc[piv.mean(axis=1).sort_values().index]
        fig = px.imshow(piv, color_continuous_scale=THREAT_SCALE, aspect="auto",
                        labels=dict(x="hour (UTC)", y="", color="flag rate"))
        fig.update_traces(xgap=1, ygap=1, hovertemplate="%{y} · %{x}:00<br>flag rate %{z:.2%}<extra></extra>")
        fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
        st.plotly_chart(style(fig, "Flag rate by merchant category × hour", 440, legend=False), key="rp_heat")
    with b:
        smp = pd.concat([df.sample(min(5000, len(df)), random_state=1), df[df["decision"] != "approve"]]).drop_duplicates("txn_id")
        fig = px.scatter(smp, x="txn_count_10m", y="amount_zscore", color="decision", color_discrete_map=STATUS,
                         category_orders={"decision": STATUS_ORDER}, opacity=0.6,
                         hover_data={"card_id": True, "merchant_category": True, "amount_usd": ":$,.2f"})
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig.update_xaxes(title="card transactions in last 10 min", type="log")
        fig.update_yaxes(title="amount z-score vs card history")
        st.plotly_chart(style(fig, "Velocity × amount anomaly", 440), key="rp_vel")

    # ---- Amount distributions ---------------------------------------------------------------
    lab = df.assign(label=np.where(df["is_fraud"], df["fraud_type"].fillna("fraud"), "not reported"))
    order = ["not reported"] + sorted(FRAUD_TYPE_COLORS)
    cmap = {"not reported": LABEL["Not reported"], **FRAUD_TYPE_COLORS}
    smp = pd.concat([lab[lab["label"] == "not reported"].sample(min(8000, (lab["label"] == "not reported").sum()), random_state=2),
                     lab[lab["label"] != "not reported"]])
    fig = px.box(smp, x="label", y="amount_usd", color="label", color_discrete_map=cmap, log_y=True, points=False,
                 category_orders={"label": order})
    fig.update_yaxes(tickprefix="$", title=None)
    fig.update_xaxes(title=None)
    st.plotly_chart(style(fig, "Ticket size by pattern: card testing hides in cents, ATO and anomalies in thousands", 360,
                          legend=False), key="rp_amt")


live_body(body)
