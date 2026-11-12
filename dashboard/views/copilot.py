"""Analyst Copilot – case briefs and questions answered from the lake, with every tool call shown."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.data import rings_snapshot
from dashboard.components.layout import page_header
from dashboard.components.theme import kpi, kpi_grid, section
from src.config import COPILOT_MODEL, REVIEW_THRESHOLD
from src.copilot import LakeTools, claude_available, run

EXAMPLES = ["Is the model drifting, and should we promote the challenger?",
            "Which fraud rings should we block first?",
            "What is in the alert queue right now?"]


def show(ans: dict) -> None:
    mode = f"Claude · {ans['model']}" if ans["mode"] == "claude" else "Offline analyst"
    st.caption(f"{mode} · {len(ans['tool_calls'])} tool calls" + (f" · {ans['note']}" if ans.get("note") else ""))
    st.markdown(ans["text"])
    if ans["tool_calls"]:
        with st.expander("Evidence trail (tool calls)"):
            st.dataframe(pd.DataFrame(ans["tool_calls"]).astype(str), hide_index=True)


page_header("Analyst Copilot", "Evidence-backed case briefs over the live lake · read-only tools · analysts decide")
claude = claude_available()
kpi_grid([
    kpi("Mode", "Claude" if claude else "Offline analyst",
        COPILOT_MODEL if claude else "set ANTHROPIC_API_KEY to enable Claude", "violet" if claude else "amber"),
    kpi("Tools", "6 read-only", "transaction · card · ring · rings · alerts · model health", "cyan"),
    kpi("Guardrails", "No writes", "tool output treated as data, not instructions", "green"),
])

tab_txn, tab_ring, tab_ask = st.tabs(["🧾 Brief a transaction", "🕸️ Brief a ring", "💬 Ask"])

with tab_txn:
    try:
        q = LakeTools().list_alerts(min_risk=REVIEW_THRESHOLD, limit=25)
        pool = pd.DataFrame(q.get("challenger_disagreements", []) + q.get("alerts", []))
    except Exception:  # noqa: BLE001 – lake empty
        pool = pd.DataFrame()
    a, b = st.columns([3, 2])
    with a:
        picks = [] if pool.empty else pool.drop_duplicates("txn_id").apply(
            lambda r: f"{r['txn_id']} · ${r['amount_usd']:,.0f} {r['merchant_category']} · risk {r['risk_score']:.0f}"
                      + (f" · challenger {r['challenger_score']:.0f}" if pd.notna(r.get("challenger_score")) else ""), axis=1).tolist()
        choice = st.selectbox("Recent alerts and challenger disagreements", picks, key="cp_pick") if picks else None
    with b:
        manual = st.text_input("…or paste a transaction id", key="cp_txn")
    txn = (manual.strip() or (choice.split(" · ")[0] if choice else ""))
    if st.button("Generate case brief", key="cp_go_txn", disabled=not txn, type="primary"):
        with st.spinner("Investigating…"):
            st.session_state["cp_txn_ans"] = run("txn", txn).to_dict()
    if st.session_state.get("cp_txn_ans"):
        section("Case brief", "▸")
        show(st.session_state["cp_txn_ans"])

with tab_ring:
    rings, _, _ = rings_snapshot()
    if rings.empty:
        st.info("No rings in the latest graph snapshot yet.")
    else:
        rid = st.selectbox("Ring", rings["ring_id"], key="cp_ring", format_func=lambda r: (
            f"{r} · {rings.set_index('ring_id').loc[r, 'status']} · {rings.set_index('ring_id').loc[r, 'n_cards']} cards"))
        if st.button("Generate ring brief", key="cp_go_ring", type="primary"):
            with st.spinner("Investigating…"):
                st.session_state["cp_ring_ans"] = run("ring", rid).to_dict()
        if st.session_state.get("cp_ring_ans"):
            section("Ring brief", "▸")
            show(st.session_state["cp_ring_ans"])

with tab_ask:
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"cp_ex_{hash(ex)}", width="stretch"):
            st.session_state["cp_q"] = ex
    question = st.text_area("Question", key="cp_q", height=80, placeholder="e.g. Why was card_012345 declined?")
    if st.button("Ask", key="cp_go_ask", disabled=not question.strip(), type="primary"):
        with st.spinner("Thinking with the evidence…"):
            st.session_state["cp_ask_ans"] = run("ask", question.strip()).to_dict()
    if st.session_state.get("cp_ask_ans"):
        section("Answer", "▸")
        show(st.session_state["cp_ask_ans"])
