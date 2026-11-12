"""
Sidebar controls + shared page chrome (header, live status, auto-refresh).

Auto-refresh uses ``st.fragment(run_every=…)`` so only the page body re-runs
on each tick – the sidebar, navigation and widget state are untouched, and
there is no full-page flicker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import streamlit as st

from dashboard.components.data import WINDOWS, Filters, LakeEmpty, latest_state
from dashboard.components.theme import header
from src.config import DECLINE_THRESHOLD, REVIEW_THRESHOLD
from src.reference import FX_TO_USD, MERCHANT_CATEGORIES

REFRESH_OPTIONS = {"2 s": 2, "5 s": 5, "10 s": 10, "30 s": 30}
TXN_TYPES = ["POS", "ECOM", "ATM", "TRANSFER", "P2P"]


@dataclass(frozen=True)
class Controls:
    live: bool
    refresh_s: int
    filters: Filters


def _reset() -> None:
    for k in list(st.session_state):
        if k.startswith("c_"):
            del st.session_state[k]


def render_sidebar() -> Controls:
    with st.sidebar:
        st.markdown("### ⚡ Live stream")
        live = st.toggle("Auto-refresh", value=True, key="c_live",
                         help="Re-query the lake on an interval. Only the page body refreshes.")
        refresh = st.select_slider("Refresh every", list(REFRESH_OPTIONS), value="5 s", key="c_refresh",
                                   disabled=not live)
        window = st.selectbox("Time window", list(WINDOWS), index=1, key="c_window",
                              help="Anchored on the newest event in the lake.")

        st.markdown("### 🎯 Risk policy")
        threshold = st.slider("Risk score threshold", 0, 100, int(REVIEW_THRESHOLD), key="c_threshold",
                              help=f"What-if: transactions at or above this score count as flagged. "
                                   f"Production policy: review ≥ {REVIEW_THRESHOLD:.0f}, decline ≥ {DECLINE_THRESHOLD:.0f}.")
        decisions = st.pills("Decision", ["approve", "review", "decline"], selection_mode="multi", key="c_decisions",
                             help="Leave empty for all decisions.")

        st.markdown("### 🔎 Transaction filters")
        txn_types = st.multiselect("Transaction type", TXN_TYPES, key="c_types", placeholder="All types")
        currencies = st.multiselect("Currency", sorted(FX_TO_USD), key="c_ccy", placeholder="All currencies")
        channels = st.multiselect("Channel", ["card_present", "card_not_present"], key="c_channel",
                                  format_func=lambda c: c.replace("_", " ").title(), placeholder="Both channels")
        with st.expander("Advanced filters"):
            categories = st.multiselect("Merchant category", sorted(MERCHANT_CATEGORIES), key="c_cats",
                                        placeholder="All categories")
            countries = st.multiselect("Transaction country", sorted({c for c in _countries()}), key="c_countries",
                                       placeholder="All countries")
            min_amount = st.number_input("Min amount (USD)", min_value=0.0, value=0.0, step=50.0, key="c_min_amt")
        st.button("↺ Reset all controls", on_click=_reset, width="stretch")

    return Controls(
        live=live, refresh_s=REFRESH_OPTIONS[refresh],
        filters=Filters(window=window, threshold=float(threshold), txn_types=tuple(txn_types),
                        currencies=tuple(currencies), channels=tuple(channels), categories=tuple(categories),
                        countries=tuple(countries), decisions=tuple(decisions or ()), min_amount=float(min_amount)))


def _countries() -> list[str]:
    from src.reference import CITIES
    return [c[1] for c in CITIES]


def controls() -> Controls:
    c = st.session_state.get("_controls")
    return c if c is not None else Controls(False, 5, Filters())


def page_header(title: str, subtitle: str) -> None:
    c = controls()
    try:
        s = latest_state()
    except LakeEmpty:
        header(title, subtitle, "paused", ["lake empty"])
        return
    age = (pd.Timestamp.now(tz="UTC") - s["last_write"]).total_seconds()
    status = "paused" if not c.live else ("on" if age < 30 else "stale")
    header(title, subtitle, status, [
        f"latest event {s['max_et']:%H:%M:%S} UTC",
        f"last write {age:,.0f}s ago" if age < 3600 else f"last write {age / 3600:,.1f}h ago",
        f"model {s['model']}",
        f"{c.filters.window.lower()}",
    ])


def live_body(fn: Callable[[], None]) -> None:
    """Run a page body inside an auto-refreshing fragment (or once, when paused)."""
    c = controls()
    st.fragment(fn, run_every=c.refresh_s if c.live else None)()
