"""Model Intelligence – live precision/recall vs. delayed labels, threshold trade-offs, feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import LakeEmpty, feature_importances, load_window, model_card, threshold_curve
from dashboard.components.layout import controls, live_body, page_header
from dashboard.components.theme import (ACCENT, FRAUD_TYPE_COLORS, LABEL, SIGNAL_SCALE, STATUS, THREAT_SCALE, kpi,
                                        kpi_grid, pct, section, style)
from src.config import DECLINE_THRESHOLD, REVIEW_THRESHOLD


def body() -> None:
    f = controls().filters
    page_header("Model Intelligence", "Isolation Forest + rules · evaluated live against delayed chargeback labels")
    try:
        df = load_window(f)
    except LakeEmpty:
        st.info("No data yet.")
        return
    if df.empty:
        st.warning("No transactions match the current filters.")
        return

    # Labels lag by 5-90 minutes: exclude the immature tail from accuracy metrics.
    mature_cut = df["event_time"].max() - pd.Timedelta(minutes=90)
    mature = df[df["event_time"] <= mature_cut]
    basis = mature if len(mature) >= 500 and mature["is_fraud"].sum() >= 20 else df
    note = ("labels matured (> 90 min old)" if basis is mature else
            "window younger than label delay – metrics are lower bounds")

    curve = threshold_curve(basis)
    at = curve.set_index("threshold").loc[int(f.threshold)]
    y, s = basis["is_fraud"].to_numpy(), basis["risk_score"].to_numpy()
    tp, fp = int(((s >= f.threshold) & y).sum()), int(((s >= f.threshold) & ~y).sum())
    fn, tn = int(((s < f.threshold) & y).sum()), int(((s < f.threshold) & ~y).sum())
    card = model_card()

    kpi_grid([
        kpi("Precision", pct(at["precision"]), f"at risk ≥ {f.threshold:.0f}", "violet"),
        kpi("Recall", pct(at["recall"]), f"{tp:,} of {tp + fn:,} confirmed fraud", "green"),
        kpi("F1", f"{at['f1']:.3f}" if at["f1"] == at["f1"] else "–", note, "cyan"),
        kpi("Flag rate", pct(at["flag_rate"], 2), f"{tp + fp:,} alerts for analysts", "amber"),
        kpi("Train ROC-AUC", f"{card.get('roc_auc', {}).get('blended_risk', float('nan')):.3f}",
            f"PR-AUC {card.get('pr_auc', {}).get('blended_risk', float('nan')):.3f} · holdout", "blue"),
        kpi("Model", str(df["model_version"].iloc[0]).replace("iforest-", ""), "Isolation Forest version", "cyan"),
    ])

    # ---- Threshold trade-off + confusion matrix -------------------------------------
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for col, color, name in (("precision", "#9085e9", "Precision"), ("recall", STATUS["approve"], "Recall"),
                                 ("f1", ACCENT, "F1")):
            fig.add_scatter(x=curve["threshold"], y=curve[col], name=name, mode="lines", line=dict(color=color, width=2),
                            hovertemplate=f"threshold %{{x}}<br>{name}: %{{y:.3f}}<extra></extra>")
        fig.add_vline(x=f.threshold, line_color=ACCENT, line_dash="dash", annotation_text=f"what-if {f.threshold:.0f}",
                      annotation_font_color=ACCENT)
        fig.add_vline(x=REVIEW_THRESHOLD, line_color=STATUS["review"], line_dash="dot")
        fig.add_vline(x=DECLINE_THRESHOLD, line_color=STATUS["decline"], line_dash="dot")
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        fig.update_xaxes(title="risk score threshold")
        st.plotly_chart(style(fig, "Precision · recall · F1 by threshold (drag the sidebar slider)", 380), key="mi_curve")
    with right:
        cm = np.array([[tn, fp], [fn, tp]])
        fig = px.imshow(cm, x=["Pred: legit", "Pred: fraud"], y=["Actual: legit", "Actual: fraud"],
                        color_continuous_scale=SIGNAL_SCALE, text_auto=",", aspect="auto")
        fig.update_traces(xgap=3, ygap=3, hovertemplate="%{y} · %{x}: %{z:,}<extra></extra>", textfont=dict(size=16))
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, f"Confusion matrix @ {f.threshold:.0f}", 380, legend=False), key="mi_cm")

    # ---- Score separation + flag-rate cost curve ---------------------------------
    a, b = st.columns(2)
    with a:
        lab = basis.assign(label=np.where(basis["is_fraud"], "Confirmed fraud", "Not reported"))
        fig = px.histogram(lab, x="risk_score", color="label", nbins=50, histnorm="percent", barmode="overlay",
                           color_discrete_map=LABEL, opacity=0.65)
        fig.update_traces(hovertemplate="risk %{x}<br>%{y:.2f}% of class<extra>%{fullData.name}</extra>")
        fig.add_vline(x=f.threshold, line_color=ACCENT, line_dash="dash")
        fig.update_xaxes(title="risk score")
        fig.update_yaxes(title="% of class", type="log")
        st.plotly_chart(style(fig, "Score separation by confirmed label (log)", 360), key="mi_sep")
    with b:
        fig = go.Figure(go.Scatter(x=curve["flag_rate"], y=curve["recall"], mode="lines+markers",
                                   marker=dict(size=4, color=curve["threshold"], colorscale=THREAT_SCALE,
                                               colorbar=dict(title="threshold")),
                                   line=dict(color="rgba(148,163,184,0.4)", width=1),
                                   customdata=curve["threshold"],
                                   hovertemplate="threshold %{customdata}<br>flag %{x:.2%} → recall %{y:.1%}<extra></extra>"))
        fig.add_scatter(x=[at["flag_rate"]], y=[at["recall"]], mode="markers", name="current",
                        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2, color=ACCENT)), showlegend=False)
        fig.update_xaxes(type="log", tickformat=".1%", title="share of traffic flagged (analyst workload)")
        fig.update_yaxes(tickformat=".0%", title="fraud recall")
        st.plotly_chart(style(fig, "Workload vs. catch-rate (each point = a threshold)", 360, legend=False), key="mi_cost")

    # ---- Detection by fraud type + complementarity ---------------------------------
    a, b = st.columns(2)
    with a:
        ft = basis[basis["is_fraud"]].groupby("fraud_type").agg(n=("txn_id", "size"), caught=("flagged", "mean")).reset_index()
        ft["trained"] = ft["fraud_type"].map(card.get("recall_by_fraud_type", {}))
        fig = go.Figure()
        fig.add_bar(x=ft["fraud_type"], y=ft["caught"], name="Live (this window)",
                    marker_color=[FRAUD_TYPE_COLORS.get(t, ACCENT) for t in ft["fraud_type"]],
                    customdata=ft["n"], hovertemplate="%{x}: %{y:.1%} caught<br>%{customdata:,} confirmed<extra></extra>")
        fig.add_scatter(x=ft["fraud_type"], y=ft["trained"], name="Training holdout (@50)", mode="markers",
                        marker=dict(symbol="line-ew", size=26, line=dict(width=3, color="#e6edf7")),
                        hovertemplate="%{x}: %{y:.1%} on holdout<extra></extra>")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(style(fig, "Recall by fraud pattern", 360), key="mi_types")
    with b:
        smp = basis.sample(min(6000, len(basis)), random_state=3)
        smp = pd.concat([smp, basis[basis["is_fraud"]]]).drop_duplicates("txn_id")
        smp["label"] = np.where(smp["is_fraud"], "Confirmed fraud", "Not reported")
        fig = px.scatter(smp, x="if_percentile", y="rule_points", color="label", color_discrete_map=LABEL, opacity=0.55,
                         category_orders={"label": ["Not reported", "Confirmed fraud"]},
                         hover_data={"risk_score": ":.0f", "rules_triggered": True, "if_percentile": ":.3f"})
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig.update_xaxes(title="Isolation Forest percentile", range=[0.5, 1.001])
        fig.update_yaxes(title="rule points")
        st.plotly_chart(style(fig, "Two detectors, complementary coverage", 360), key="mi_comp")

    # ---- Feature importance ----------------------------------------------------------
    section("What the anomaly model looks at", "▸")
    imp = feature_importances().sort_values("importance").tail(15)
    a, b = st.columns([3, 2])
    with a:
        fig = px.bar(imp, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=SIGNAL_SCALE)
        fig.update_traces(hovertemplate="%{y}: %{x:.3f}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="surrogate importance (share of anomaly-score variance explained)")
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, "Global feature importance (surrogate random forest)", 460, legend=False), key="mi_imp")
    with b:
        st.markdown("**Training holdout metrics**")
        if card:
            rows = [{"Detector": k.replace("_", " ").title(), "ROC-AUC": card["roc_auc"][k], "PR-AUC": card["pr_auc"][k]}
                    for k in ("isolation_forest", "rules", "blended_risk")]
            st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
                "ROC-AUC": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                "PR-AUC": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
            thr = pd.DataFrame(card["thresholds"]).T.reset_index(names="policy")
            st.dataframe(thr, hide_index=True, column_config={
                "precision": st.column_config.NumberColumn(format="percent"),
                "recall": st.column_config.NumberColumn(format="percent"),
                "flag_rate": st.column_config.NumberColumn(format="percent")})
            st.caption(f"Trained on {card['n_train']:,} events, evaluated on a later {card['n_test']:,}-event holdout "
                       f"({card['test_fraud_rate']:.2%} fraud). The forest never sees labels.")
        else:
            st.info("Model card not found – run `python -m src.train_model`.")


live_body(body)
