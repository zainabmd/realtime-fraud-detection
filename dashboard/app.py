"""
Real-time Fraud Operations Center – Streamlit entry point.

    streamlit run dashboard/app.py

Owns navigation, theme and the global sidebar; pages read the resulting
`Controls` from session state and render inside auto-refreshing fragments.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.components.layout import render_sidebar  # noqa: E402
from dashboard.components.theme import inject_css, register_plotly_template  # noqa: E402

st.set_page_config(page_title="Fraud Operations Center", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
register_plotly_template()

V = Path(__file__).parent / "views"
nav = st.navigation({
    "Operations": [
        st.Page(V / "command_center.py", title="Command Center", icon="🛰️", default=True),
        st.Page(V / "threat_map.py", title="Threat Map", icon="🌐"),
        st.Page(V / "investigate.py", title="Investigations", icon="🔍"),
    ],
    "Detection": [
        st.Page(V / "fraud_rings.py", title="Fraud Rings", icon="🕸️"),
        st.Page(V / "model_intel.py", title="Model Intelligence", icon="🧠"),
        st.Page(V / "model_health.py", title="Model Health", icon="🩺"),
        st.Page(V / "rules_patterns.py", title="Rules & Patterns", icon="🧩"),
    ],
    "Assist": [
        st.Page(V / "copilot.py", title="Analyst Copilot", icon="🤖"),
    ],
    "Platform": [
        st.Page(V / "pipeline_health.py", title="Pipeline Health", icon="⚙️"),
    ],
})

with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin:4px 0 10px'>"
        "<div style='font-size:1.6rem'>🛡️</div><div><div style='font-weight:700;color:#e6edf7'>FraudOps</div>"
        "<div style='font-size:.72rem;color:#6b7a93'>Streaming risk · graph · AI copilot</div></div></div>",
        unsafe_allow_html=True)
st.session_state["_controls"] = render_sidebar()

nav.run()
