"""Threat Map – geographic fraud hotspots, cross-border attack arcs and country drill-down."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import LakeEmpty, load_window
from dashboard.components.drill import feed_table, render_open_dialog
from dashboard.components.layout import controls, live_body, page_header
from dashboard.components.theme import (ACCENT, STATUS, THREAT_SCALE, kpi, kpi_grid, money, num, pct, section, style)


def body() -> None:
    f = controls().filters
    page_header("Threat Map", "Where flagged activity originates, and where it travels")
    try:
        df = load_window(f)
    except LakeEmpty:
        st.info("No data yet.")
        return
    if df.empty:
        st.warning("No transactions match the current filters.")
        return

    df = df.assign(flag_usd=df["amount_usd"].where(df["flagged"], 0.0))
    city = (df.groupby(["city", "country"], observed=True)
            .agg(lat=("lat", "median"), lon=("lon", "median"), txns=("txn_id", "size"), flagged=("flagged", "sum"),
                 fraud=("is_fraud", "sum"), at_risk=("flag_usd", "sum"))
            .reset_index())
    city = city[city["txns"] > 0]
    city["flag_rate"] = city["flagged"] / city["txns"]
    cross = df[df["flagged"] & (df["geo_mismatch"] == 1)]

    hottest = city.sort_values("flagged", ascending=False).iloc[0]
    kpi_grid([
        kpi("Active hotspots", f"{int((city['flag_rate'] > city['flag_rate'].median() * 2).sum())}",
            "cities > 2× median flag rate", "red"),
        kpi("Top hotspot", str(hottest["city"]), f"{num(hottest['flagged'])} flagged · {pct(hottest['flag_rate'])}", "amber"),
        kpi("Cross-border flagged", num(len(cross)), f"{pct(len(cross) / max(df['flagged'].sum(), 1))} of all flags", "violet"),
        kpi("Countries affected", f"{df.loc[df['flagged'], 'country'].nunique()}", f"of {df['country'].nunique()} active", "cyan"),
    ])

    view = st.segmented_control("Map layer", ["Hotspots", "Attack arcs", "Both"], default="Both", key="tm_layer") or "Both"
    fig = go.Figure()
    if view in ("Hotspots", "Both"):
        size = np.sqrt(city["flagged"].clip(lower=0)) * 3.2 + 4
        fig.add_scattergeo(
            lat=city["lat"], lon=city["lon"], mode="markers", name="Hotspots",
            marker=dict(size=size, color=city["flag_rate"], colorscale=THREAT_SCALE, cmin=0,
                        cmax=max(city["flag_rate"].quantile(0.95), 0.01), opacity=0.85,
                        line=dict(width=1, color="rgba(254,202,202,0.6)"),
                        colorbar=dict(title="Flag<br>rate", tickformat=".0%", len=0.6, x=1.0)),
            customdata=np.stack([city["city"], city["txns"], city["flagged"], city["flag_rate"], city["at_risk"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,} txns · %{customdata[2]:,} flagged"
                          "<br>flag rate %{customdata[3]:.2%}<br>$%{customdata[4]:,.0f} at risk<extra></extra>")
    if view in ("Attack arcs", "Both") and len(cross):
        arcs = cross.nlargest(250, "risk_score")
        for decision, grp in arcs.groupby("decision", observed=True):
            lats, lons = [], []
            for r in grp.itertuples():
                lats += [r.home_lat, r.lat, None]
                lons += [r.home_lon, r.lon, None]
            fig.add_scattergeo(lat=lats, lon=lons, mode="lines", name=f"{decision} arcs",
                               line=dict(width=1, color=STATUS[str(decision)]), opacity=0.45, hoverinfo="skip")
        fig.add_scattergeo(lat=arcs["lat"], lon=arcs["lon"], mode="markers", name="Attack endpoint",
                           marker=dict(size=5, color=ACCENT, symbol="diamond"),
                           customdata=np.stack([arcs["card_id"], arcs["home_country"], arcs["city"], arcs["amount_usd"],
                                                arcs["risk_score"]], axis=1),
                           hovertemplate="%{customdata[0]}: home %{customdata[1]} → %{customdata[2]}"
                                         "<br>$%{customdata[3]:,.0f} · risk %{customdata[4]:.0f}<extra></extra>")
    fig.update_geos(lataxis_range=[-50, 75], lonaxis_range=[-140, 160], projection_type="natural earth")
    st.plotly_chart(style(fig, "Global threat surface · bubble = flagged volume, colour = flag rate", 560), key="tm_map")

    # ---- Cross-border matrix + country ranking --------------------------------------
    left, right = st.columns([1.2, 1])
    with left:
        if len(cross):
            mat = cross.groupby(["home_country", "country"], observed=True).size().rename("n").reset_index()
            top_home = mat.groupby("home_country", observed=True)["n"].sum().nlargest(10).index
            top_dest = mat.groupby("country", observed=True)["n"].sum().nlargest(12).index
            piv = (mat[mat["home_country"].isin(top_home) & mat["country"].isin(top_dest)]
                   .pivot_table(index="home_country", columns="country", values="n", fill_value=0, observed=True))
            fig = px.imshow(piv, color_continuous_scale=THREAT_SCALE, aspect="auto",
                            labels=dict(x="Transaction country", y="Cardholder home", color="Flagged"))
            fig.update_traces(xgap=2, ygap=2, hovertemplate="home %{y} → %{x}: %{z:,} flagged<extra></extra>")
            st.plotly_chart(style(fig, "Cross-border flagged flows (home → spend country)", 400, legend=False), key="tm_matrix")
        else:
            st.info("No cross-border flagged transactions in this window.")
    with right:
        ctry = (df.groupby("country", observed=True)
                .agg(txns=("txn_id", "size"), flagged=("flagged", "sum"),
                     at_risk=("flag_usd", "sum")).reset_index())
        ctry["flag_rate"] = ctry["flagged"] / ctry["txns"]
        ctry = ctry[ctry["txns"] >= 20].sort_values("flag_rate").tail(14)
        fig = px.bar(ctry, x="flag_rate", y="country", orientation="h", color="flag_rate",
                     color_continuous_scale=THREAT_SCALE, custom_data=["txns", "flagged", "at_risk"])
        fig.update_traces(hovertemplate="%{y}: %{x:.2%} flagged<br>%{customdata[1]:,} of %{customdata[0]:,} txns"
                                        "<br>$%{customdata[2]:,.0f} at risk<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".1%", title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, "Flag rate by transaction country (≥ 20 txns)", 400, legend=False), key="tm_ctry")

    # ---- Country drill-down ---------------------------------------------------------
    section("Country drill-down", "▸")
    options = df.loc[df["flagged"], "country"].value_counts().index.tolist() or df["country"].value_counts().index.tolist()
    sel = st.selectbox("Country", options, key="tm_country")
    sub = df[df["country"] == sel]
    a, b = st.columns(2)
    with a:
        cat = sub.groupby("merchant_category", observed=True).agg(txns=("txn_id", "size"), flagged=("flagged", "sum")).reset_index()
        cat = cat[cat["txns"] > 0]
        cat["rate"] = cat["flagged"] / cat["txns"]
        fig = px.bar(cat.sort_values("rate").tail(10), x="rate", y="merchant_category", orientation="h")
        fig.update_traces(marker_color=STATUS["review"], hovertemplate="%{y}: %{x:.2%}<extra></extra>")
        fig.update_xaxes(tickformat=".0%", title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, f"{sel}: flag rate by merchant category", 320, legend=False), key="tm_dd_cat")
    with b:
        home = sub[sub["flagged"]].groupby("home_country", observed=True).size().nlargest(10).rename("n").reset_index()
        fig = px.bar(home, x="n", y="home_country", orientation="h")
        fig.update_traces(marker_color="#9085e9", hovertemplate="cards from %{y}: %{x:,}<extra></extra>")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, categoryorder="total ascending")
        st.plotly_chart(style(fig, f"{sel}: where flagged cards are issued", 320, legend=False), key="tm_dd_home")
    feed_table(sub[sub["flagged"]].nlargest(100, "risk_score"), key="tm_feed", height=300)


live_body(body)
render_open_dialog()
