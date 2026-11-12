"""
"Security Operations Center" visual system.

Fraud monitoring is an operational, high-stakes context, so the dashboard uses
a dark control-room aesthetic: deep-navy surfaces that keep analysts' eyes
rested on long shifts, a cyan "system" accent, and a reserved traffic-light
status scale (approve / review / decline) that is never reused for anything else.

Colour roles
------------
* STATUS     approve / review / decline – decisions only, always with a text label
* LABEL      confirmed fraud vs legit   – ground-truth comparisons
* CATEGORICAL fixed-order palette (dark-surface steps) – rules, fraud types, devices
* THREAT     single-hue red ramp, dark→bright = more risk (heatmaps, map colour)
* SIGNAL     single-hue cyan ramp for neutral magnitude (volume, throughput)
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---- Surfaces & ink ---------------------------------------------------------
BG = "#070d1a"
SURFACE = "#0e1729"
SURFACE_2 = "#131f36"
BORDER = "rgba(56, 189, 248, 0.16)"
INK = "#e6edf7"
INK_2 = "#a9b7cc"
INK_MUTED = "#6b7a93"
GRID = "rgba(148, 163, 184, 0.10)"
ACCENT = "#22d3ee"            # cyan "system" accent

# ---- Reserved status scale ------------------------------------------------
STATUS = {"approve": "#10b981", "review": "#f59e0b", "decline": "#ef4444"}
STATUS_ORDER = ["approve", "review", "decline"]
LABEL = {"Confirmed fraud": "#f43f5e", "Not reported": "#38bdf8"}

# ---- Categorical (dark-surface steps of the validated palette, fixed order) --
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#16a34a", "#9085e9", "#e66767"]
FRAUD_TYPE_COLORS = {"card_testing": PALETTE[0], "account_takeover": PALETTE[1], "impossible_travel": PALETTE[2],
                     "velocity_burst": PALETTE[3], "high_amount_anomaly": PALETTE[4], "fraud_ring": PALETTE[6]}
MODEL_COLORS = {"champion": "#38bdf8", "challenger": "#a78bfa"}       # champion = IF + rules, challenger = HGB + graph
PSI_COLORS = {"stable": "#10b981", "watch": "#f59e0b", "drift": "#ef4444", "seasonal": "#64748b"}

THREAT = ["#140f1f", "#2a1426", "#4a1729", "#76192b", "#a61b29", "#dc2626", "#f87171", "#fecaca"]
THREAT_SCALE = [[i / (len(THREAT) - 1), c] for i, c in enumerate(THREAT)]
SIGNAL = ["#0a1626", "#0b2a40", "#0c4560", "#0e6584", "#0891b2", "#22d3ee", "#a5f3fc"]
SIGNAL_SCALE = [[i / (len(SIGNAL) - 1), c] for i, c in enumerate(SIGNAL)]


# --------------------------------------------------------------------------- #
# Plotly template
# --------------------------------------------------------------------------- #
def register_plotly_template() -> None:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", color=INK_2, size=12),
        title=dict(font=dict(color=INK, size=14), x=0, xanchor="left", yref="container", y=1, yanchor="top", pad=dict(t=10)),
        colorway=PALETTE,
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(148,163,184,0.25)", tickcolor="rgba(148,163,184,0.25)",
                   showgrid=False),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK_2), title_text=""),
        hoverlabel=dict(bgcolor="#0b1424", bordercolor=ACCENT, font=dict(color=INK, size=12)),
        margin=dict(l=8, r=8, t=78, b=8),
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=INK_2), title=dict(font=dict(color=INK_2)))),
        geo=dict(bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#111c31", showocean=True, oceancolor="#060b16",
                 showcountries=True, countrycolor="rgba(56,189,248,0.18)", showcoastlines=False, showframe=False,
                 showlakes=False, projection_type="natural earth"),
        bargap=0.22,
    )
    pio.templates["soc"] = tpl
    pio.templates.default = "soc"


def style(fig: go.Figure, title: str | None = None, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(title_text=title, height=height, showlegend=legend,
                      margin=dict(t=(34 if title else 8) + (40 if legend else 6)))
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line=dict(width=2))
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
:root {{ --bg:{BG}; --surface:{SURFACE}; --surface2:{SURFACE_2}; --ink:{INK}; --ink2:{INK_2}; --muted:{INK_MUTED};
        --accent:{ACCENT}; --ok:{STATUS['approve']}; --warn:{STATUS['review']}; --bad:{STATUS['decline']}; }}
html, body, [data-testid="stAppViewContainer"] {{ font-family: Inter, 'Segoe UI', system-ui, sans-serif; }}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 600px at 85% -10%, rgba(34,211,238,0.08), transparent 60%),
              radial-gradient(900px 500px at -10% 110%, rgba(239,68,68,0.06), transparent 60%), var(--bg);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0b1427 0%, #08101f 100%);
  border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] h3 {{ color: var(--accent); letter-spacing: .08em; text-transform: uppercase; font-size: .78rem; }}
.block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
h1, h2, h3 {{ color: var(--ink); letter-spacing: -0.01em; }}

/* ---- Page header ---- */
.soc-header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom: 6px;
  padding: 14px 18px; border:1px solid {BORDER}; border-radius:14px;
  background: linear-gradient(90deg, rgba(34,211,238,0.10), rgba(14,23,41,0.6) 40%, rgba(239,68,68,0.06));
  box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 10px 30px rgba(0,0,0,0.35); animation: fadeIn .5s ease; }}
.soc-title {{ font-size: 1.55rem; font-weight: 700; color: var(--ink); margin:0; }}
.soc-sub {{ color: var(--muted); font-size: .82rem; margin-top: 2px; }}
.soc-meta {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }}
.pill {{ font-family: 'JetBrains Mono', monospace; font-size:.72rem; color: var(--ink2); padding:4px 10px;
  border-radius:999px; border:1px solid {BORDER}; background: rgba(8,16,31,.7); }}

/* ---- Live badge ---- */
.live {{ display:inline-flex; align-items:center; gap:8px; font-weight:700; font-size:.74rem; letter-spacing:.12em;
  padding:5px 12px; border-radius:999px; }}
.live.on {{ color:#a7f3d0; background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.45); }}
.live.stale {{ color:#fde68a; background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.45); }}
.live.paused {{ color:var(--ink2); background:rgba(148,163,184,.10); border:1px solid rgba(148,163,184,.3); }}
.dot {{ width:8px; height:8px; border-radius:50%; background:currentColor; position:relative; }}
.live.on .dot::after {{ content:''; position:absolute; inset:-4px; border-radius:50%; border:2px solid currentColor;
  animation: pulse 1.6s ease-out infinite; }}
@keyframes pulse {{ 0% {{ transform:scale(.6); opacity:.9; }} 100% {{ transform:scale(2.2); opacity:0; }} }}
@keyframes fadeIn {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform:none; }} }}

/* ---- KPI cards ---- */
.kpi-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin: 10px 0 6px; }}
.kpi {{ position:relative; overflow:hidden; padding:14px 16px 12px; border-radius:14px; border:1px solid {BORDER};
  background: linear-gradient(160deg, rgba(19,31,54,.92), rgba(10,18,33,.92));
  box-shadow: 0 8px 24px rgba(0,0,0,.35); transition: transform .18s ease, box-shadow .18s ease, border-color .18s;
  animation: fadeIn .45s ease both; }}
.kpi:hover {{ transform: translateY(-3px); border-color: var(--tone); box-shadow: 0 0 22px -6px var(--tone), 0 12px 28px rgba(0,0,0,.45); }}
.kpi::before {{ content:''; position:absolute; left:0; top:0; height:3px; width:100%;
  background: linear-gradient(90deg, var(--tone), transparent); }}
.kpi .lbl {{ color: var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; display:flex; gap:6px; align-items:center; }}
.kpi .val {{ color: var(--ink); font-size:1.45rem; font-weight:700; margin-top:4px; font-variant-numeric: tabular-nums; }}
.kpi .sub {{ color: var(--ink2); font-size:.76rem; margin-top:2px; }}
.kpi .delta.up {{ color:#fca5a5; }} .kpi .delta.down {{ color:#6ee7b7; }} .kpi .delta.flat {{ color:var(--muted); }}
.kpi svg {{ position:absolute; right:10px; bottom:8px; opacity:.85; }}
.tone-dot {{ width:7px; height:7px; border-radius:50%; background: var(--tone); box-shadow: 0 0 8px var(--tone); }}

/* ---- Section headers ---- */
.section {{ display:flex; align-items:center; gap:10px; margin: 18px 0 4px; color: var(--ink); font-weight:600; font-size:1.02rem; }}
.section::after {{ content:''; flex:1; height:1px; background: linear-gradient(90deg, {BORDER}, transparent); }}
.section .ico {{ color: var(--accent); }}

/* ---- Alert ticker ---- */
.ticker {{ overflow:hidden; white-space:nowrap; border:1px solid rgba(239,68,68,.35); border-radius:10px;
  background: linear-gradient(90deg, rgba(239,68,68,.12), rgba(14,23,41,.6)); padding:7px 0; margin: 8px 0 4px; }}
.ticker-track {{ display:inline-block; padding-left:100%; animation: scroll 60s linear infinite; }}
.ticker:hover .ticker-track {{ animation-play-state: paused; }}
.tick {{ display:inline-flex; align-items:center; gap:6px; margin-right:34px; font-family:'JetBrains Mono',monospace; font-size:.76rem; color: var(--ink2); }}
.tick b {{ color: var(--ink); }}
.badge {{ font-size:.66rem; font-weight:700; padding:2px 7px; border-radius:6px; letter-spacing:.06em; }}
.badge.decline {{ background:rgba(239,68,68,.18); color:#fecaca; border:1px solid rgba(239,68,68,.5); }}
.badge.review {{ background:rgba(245,158,11,.16); color:#fde68a; border:1px solid rgba(245,158,11,.5); }}
.badge.approve {{ background:rgba(16,185,129,.14); color:#a7f3d0; border:1px solid rgba(16,185,129,.45); }}
@keyframes scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-100%); }} }}

/* ---- Streamlit widgets in the dark shell ---- */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{
  border:1px solid {BORDER}; border-radius:14px; padding:6px; background: rgba(10,18,33,.55);
  box-shadow: 0 8px 24px rgba(0,0,0,.25); animation: fadeIn .4s ease both; }}
[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:12px; background: rgba(10,18,33,.5); }}
[data-testid="stMetric"] {{ background: rgba(14,23,41,.7); border:1px solid {BORDER}; border-radius:12px; padding:10px 14px; }}
div[data-baseweb="tab-list"] button[aria-selected="true"] {{ color: var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}
.caption-note {{ color: var(--muted); font-size:.78rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HTML components
# --------------------------------------------------------------------------- #
TONES = {"cyan": ACCENT, "green": STATUS["approve"], "amber": STATUS["review"], "red": STATUS["decline"],
         "violet": "#a78bfa", "blue": "#60a5fa"}


def _sparkline(values: list[float], color: str, w: int = 78, h: int = 26) -> str:
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - (v - lo) / rng * (h - 2) - 1:.1f}" for i, v in enumerate(vals))
    area = f"0,{h} {pts} {w},{h}"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polygon points="{area}" fill="{color}" opacity=".12"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/></svg>')


def kpi(label: str, value: str, sub: str | None = None, tone: str = "cyan", spark: list[float] | None = None,
        delta: str | None = None, delta_dir: str = "flat") -> str:
    color = TONES.get(tone, tone)
    d = f' <span class="delta {delta_dir}">{html.escape(delta)}</span>' if delta else ""
    s = f'<div class="sub">{html.escape(sub)}{d}</div>' if sub or delta else ""
    return (f'<div class="kpi" style="--tone:{color}"><div class="lbl"><span class="tone-dot"></span>{html.escape(label)}</div>'
            f'<div class="val">{html.escape(value)}</div>{s}{_sparkline(spark or [], color)}</div>')


def kpi_grid(cards: list[str]) -> None:
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def section(title: str, icon: str = "◆") -> None:
    st.markdown(f'<div class="section"><span class="ico">{icon}</span>{html.escape(title)}</div>', unsafe_allow_html=True)


def header(title: str, subtitle: str, status: str, pills: list[str]) -> None:
    label = {"on": "LIVE", "stale": "STALE", "paused": "PAUSED"}[status]
    pills_html = "".join(f'<span class="pill">{html.escape(p)}</span>' for p in pills)
    st.markdown(
        f'<div class="soc-header"><div><div class="soc-title">{html.escape(title)}</div>'
        f'<div class="soc-sub">{html.escape(subtitle)}</div></div>'
        f'<div class="soc-meta">{pills_html}<span class="live {status}"><span class="dot"></span>{label}</span></div></div>',
        unsafe_allow_html=True)


def ticker(rows: list[dict]) -> None:
    if not rows:
        return
    items = "".join(
        f'<span class="tick"><span class="badge {r["decision"]}">{r["decision"].upper()}</span>'
        f'<b>${r["amount_usd"]:,.0f}</b> {html.escape(str(r["merchant_category"]))} · {html.escape(str(r["city"]))} '
        f'· risk {r["risk_score"]:.0f} · {html.escape(str(r["rules"] or "anomaly"))}</span>' for r in rows)
    st.markdown(f'<div class="ticker"><div class="ticker-track">{items}{items}</div></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def money(x: float) -> str:
    if x != x:
        return "–"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            v = x / div
            return f"${v:,.{0 if abs(v) >= 100 else 1}f}{unit}"
    return f"${x:,.0f}"


def num(x: float) -> str:
    if x != x:
        return "–"
    for unit, div in (("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:,.1f}{unit}"
    return f"{x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return "–" if x != x else f"{x * 100:.{d}f}%"
