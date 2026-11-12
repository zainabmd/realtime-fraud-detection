"""Model Health – drift (PSI) against the training reference, and the champion-vs-challenger verdict."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import LakeEmpty, load_window, monitoring_report, reference_profile
from dashboard.components.layout import controls, live_body, page_header
from dashboard.components.theme import (FRAUD_TYPE_COLORS, LABEL, MODEL_COLORS, PSI_COLORS, kpi, kpi_grid, pct, section,
                                        style)
from src.config import PSI_DRIFT, PSI_WATCH

VERDICT = {"promote": ("Promote challenger", "green"), "keep_champion": ("Keep champion", "cyan"),
           "retire_challenger": ("Retire challenger", "amber"), "insufficient_data": ("Need more labels", "amber")}


def body() -> None:
    page_header("Model Health", "Population-stability drift vs training · champion (IF + rules) vs challenger "
                                "(gradient boosting + graph features) on matured labels")
    rep = monitoring_report()
    if not rep or rep.get("status") != "ok":
        st.info("No monitoring report yet. `python -m src.monitoring --watch 300` writes one every 5 minutes.")
        return
    comp, dsum = rep.get("comparison", {}), rep.get("drift_summary", {})
    v_label, v_tone = VERDICT.get(comp.get("verdict"), ("–", "cyan"))
    champ, chall = comp.get("champion", {}), comp.get("challenger", {})
    kpi_grid([
        kpi("Promotion verdict", v_label, comp.get("reason", "")[:70], v_tone),
        kpi("PR-AUC champion", f"{champ.get('pr_auc', float('nan')):.3f}", f"{comp.get('n_fraud', 0):,} matured frauds", "blue"),
        kpi("PR-AUC challenger", f"{chall.get('pr_auc', float('nan')):.3f}",
            f"{comp.get('pr_auc_gain', float('nan')):+.3f} vs champion", "violet"),
        kpi("Recall @ equal workload", f"{pct(champ.get('recall', float('nan')))} → {pct(chall.get('recall', float('nan')))}",
            f"both flag {pct(comp.get('flag_rate', float('nan')), 2)} of traffic", "green"),
        kpi("Features drifting", f"{dsum.get('drift', 0)}", f"{dsum.get('watch', 0)} on watch · PSI > {PSI_DRIFT}",
            "red" if dsum.get("drift") else "green"),
        kpi("Worst PSI", f"{dsum.get('max_psi', 0):.2f}", str(dsum.get("worst", "–")), "red" if dsum.get("max_psi", 0) > PSI_DRIFT else "cyan"),
    ])
    st.caption(f"Report generated {rep['generated_at']} over the last {rep['window_hours']:.0f} h · "
               f"{rep.get('matured_rows', 0):,} events old enough for their chargeback labels to have arrived.")

    # ---- Drift ------------------------------------------------------------------------------
    section("Drift: is live traffic still like the training data?", "▸")
    drift = pd.DataFrame(rep.get("drift", []))
    left, right = st.columns([3, 2])
    with left:
        if drift.empty:
            st.info("Not enough rows in the window for drift statistics.")
        else:
            d = drift.sort_values("psi").tail(18)
            fig = go.Figure(go.Bar(x=d["psi"], y=d["feature"], orientation="h", marker_color=[PSI_COLORS[s] for s in d["status"]],
                                   customdata=d[["ref_mean", "live_mean", "kind"]],
                                   hovertemplate="%{y} (%{customdata[2]})<br>PSI %{x:.3f}<br>mean %{customdata[0]:.3f} → "
                                                 "%{customdata[1]:.3f}<extra></extra>"))
            for x, name in ((PSI_WATCH, "watch"), (PSI_DRIFT, "drift")):
                fig.add_vline(x=x, line_dash="dot", line_color=PSI_COLORS[name], annotation_text=name,
                              annotation_font_color=PSI_COLORS[name])
            fig.update_xaxes(title="population stability index")
            fig.update_yaxes(title=None)
            st.plotly_chart(style(fig, "PSI per model input and score (top 18)", 520, legend=False), key="mh_psi")
    with right:
        prof = reference_profile()
        # worst non-seasonal feature first: time-of-day "drift" is expected inside a 3-hour window
        feats = (drift.sort_values(["status", "psi"], key=lambda c: c.eq("seasonal") if c.name == "status" else -c)
                 ["feature"].tolist() if not drift.empty else list(prof))
        feat = st.selectbox("Compare distributions", feats, key="mh_feat")
        try:
            df = load_window(controls().filters)
        except LakeEmpty:
            df = pd.DataFrame()
        if feat in prof and not df.empty and feat in df:
            ref = prof[feat]
            edges = np.round(np.asarray(ref["edges"]), 4)           # same grid as src/monitoring.py
            x = np.round(df[feat].to_numpy(dtype=float), 4)
            x = x[np.isfinite(x)]
            live = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1) / max(len(x), 1)
            labels = [f"≤{edges[0]:.2g}"] + [f"{a:.2g}–{b:.2g}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:.2g}"]
            fig = go.Figure()
            fig.add_bar(x=labels, y=ref["expected"], name="Training reference", marker_color="#475569")
            fig.add_bar(x=labels, y=live, name="Live window", marker_color=MODEL_COLORS["challenger"])
            fig.update_layout(barmode="group")
            fig.update_yaxes(tickformat=".0%", title="share of events")
            fig.update_xaxes(title=f"{feat} (training decile bins)")
            st.plotly_chart(style(fig, f"{feat}: reference vs live", 520), key="mh_dist")

    # ---- Champion vs challenger ----------------------------------------------------------------
    section("Champion vs challenger at the same analyst workload", "▸")
    a, b = st.columns(2)
    with a:
        bt = pd.DataFrame(comp.get("recall_by_type", {})).T.reset_index(names="fraud_type")
        if not bt.empty:
            fig = go.Figure()
            for model in ("champion", "challenger"):
                fig.add_bar(x=bt["fraud_type"], y=bt[model], name=model.title(), marker_color=MODEL_COLORS[model],
                            customdata=bt["n"], hovertemplate="%{x}: %{y:.1%} caught<br>%{customdata} matured frauds"
                                                              f"<extra>{model}</extra>")
            fig.update_layout(barmode="group")
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title="recall")
            st.plotly_chart(style(fig, "Recall by fraud pattern (rings are invisible to the champion)", 400), key="mh_types")
    with b:
        try:
            df = load_window(controls().filters)
        except LakeEmpty:
            df = pd.DataFrame()
        if not df.empty and "challenger_score" in df:
            smp = df.dropna(subset=["challenger_score"])
            smp = pd.concat([smp.sample(min(5000, len(smp)), random_state=1), smp[smp["is_fraud"]]]).drop_duplicates("txn_id")
            smp["label"] = np.where(smp["is_fraud"], np.where(smp["fraud_type"] == "fraud_ring", "Ring fraud", "Other fraud"),
                                    "Not reported")
            colors = {"Not reported": LABEL["Not reported"], "Other fraud": LABEL["Confirmed fraud"],
                      "Ring fraud": FRAUD_TYPE_COLORS["fraud_ring"]}
            fig = go.Figure()
            for lab in ("Not reported", "Other fraud", "Ring fraud"):
                s = smp[smp["label"] == lab]
                fig.add_scatter(x=s["risk_score"], y=s["challenger_score"], mode="markers", name=lab,
                                marker=dict(size=6 if lab == "Not reported" else 8, color=colors[lab],
                                            opacity=.35 if lab == "Not reported" else .85),
                                hovertemplate="champion %{x:.0f} · challenger %{y:.0f}<extra>" + lab + "</extra>")
            fig.update_xaxes(title="champion risk score", range=[-2, 102])
            fig.update_yaxes(title="challenger score", range=[-2, 102])
            st.plotly_chart(style(fig, "Where the two models disagree (top-left = challenger-only catches)", 400),
                            key="mh_scatter")
    st.caption("The challenger is shadow-scored on every event and never drives decisions. Promotion requires "
               "enough matured fraud labels and a PR-AUC lead with recall at least as good at the same flag rate.")


live_body(body)
