"""Dead Stock Detector: Streamlit app.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

from fid import Config, analyze, ml, sku_timeline
from fid.config import CLASS_COLORS, CLASS_LABELS, STATUS_INSUFFICIENT, STATUS_OUT_OF_STOCK
from fid.evaluate import behavioural_backtest
from fid.export import fmt_money, friendly_table, list_sheets, read_table, to_excel_bytes
from fid.schema import (MOVEMENT_ALIASES, SKU_ALIASES, SKU_COLUMNS, guess_kind_values, standardize_movements,
                        standardize_skus, suggest_mapping)
from fid.simulate import make_warehouse

st.set_page_config(page_title="Dead Stock Detector", page_icon="🦇", layout="wide")

ORDER = ["forgotten", "obsolete", "temporarily_inactive", "slow_moving", "seasonal", "active",
         STATUS_OUT_OF_STOCK, STATUS_INSUFFICIENT]
NONE = "(none)"
TEMPLATES = ROOT / "templates"



# --------------------------------------------------------------------------- theme
ACCENT = "#FFC400"
pio.templates["dsd"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,27,0.6)",
    font=dict(family="Rajdhani, Segoe UI, sans-serif", color="#E4E7EB", size=14),
    colorway=[ACCENT, "#2EC4B6", "#5B8DEF", "#FF8C1A", "#FF3B4E", "#A23B72"],
    xaxis=dict(gridcolor="#1E252E", zerolinecolor="#2A323D", linecolor="#2A323D"),
    yaxis=dict(gridcolor="#1E252E", zerolinecolor="#2A323D", linecolor="#2A323D"),
    title=dict(font=dict(family="Orbitron, Rajdhani, sans-serif", size=15, color=ACCENT)),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#0B0D10", bordercolor=ACCENT, font=dict(color="#E4E7EB")),
))
pio.templates.default = "dsd"

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Rajdhani', 'Segoe UI', sans-serif; font-size: 1.05rem; }
.stApp {
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(255,196,0,0.10), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(255,196,0,0.05), transparent 60%),
    repeating-linear-gradient(135deg, rgba(255,255,255,0.012) 0 2px, transparent 2px 14px),
    #07090C;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }

/* hero */
.dsd-hero {
  position: relative; padding: 1.5rem 2rem 1.3rem 2rem; margin-bottom: 1.2rem;
  background: linear-gradient(110deg, #11151B 0%, #0B0E12 70%);
  border: 1px solid #2A323D; border-left: 6px solid #FFC400;
  clip-path: polygon(0 0, calc(100% - 28px) 0, 100% 28px, 100% 100%, 28px 100%, 0 calc(100% - 28px));
  box-shadow: 0 0 40px rgba(255,196,0,0.08);
}
.dsd-hero:before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0 1px, transparent 1px 4px);
}
.dsd-tag { font-family: 'Orbitron', sans-serif; font-size: .72rem; letter-spacing: .35em; color: #FFC400; opacity: .9; }
.dsd-title { font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 2.3rem; letter-spacing: .08em;
  margin: .2rem 0 .3rem 0; color: #F3F5F7; text-transform: uppercase; line-height: 1.15; }
.dsd-title span { color: #FFC400; text-shadow: 0 0 18px rgba(255,196,0,0.55); }
.dsd-sub { color: #9AA5B1; font-size: 1.1rem; letter-spacing: .03em; max-width: 60rem; }
.dsd-status { position: absolute; top: 1rem; right: 2.6rem; font-family: 'Orbitron', sans-serif; font-size: .68rem;
  letter-spacing: .25em; color: #2EC4B6; }
.dsd-status:before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: #2EC4B6; margin-right: .6rem; box-shadow: 0 0 10px #2EC4B6; animation: dsdpulse 1.8s infinite; }
@keyframes dsdpulse { 0%,100% { opacity: 1 } 50% { opacity: .25 } }

/* headings */
h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: .05em; }
h2, h3 { color: #F3F5F7; }
h2:after, h3:after { content: ""; display: block; width: 56px; height: 3px; margin-top: .35rem; background: #FFC400; }

/* metric cards */
div[data-testid="stMetric"] {
  background: linear-gradient(160deg, #141920, #0C0F13); border: 1px solid #2A323D !important;
  border-top: 3px solid #FFC400 !important; border-radius: 2px !important; padding: .9rem 1rem !important;
  clip-path: polygon(0 0, 100% 0, 100% calc(100% - 14px), calc(100% - 14px) 100%, 0 100%);
}
div[data-testid="stMetricLabel"] p { font-family: 'Orbitron', sans-serif; font-size: .7rem !important;
  letter-spacing: .18em; text-transform: uppercase; color: #9AA5B1; }
div[data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; font-weight: 700; color: #F3F5F7; }

/* tabs */
button[data-baseweb="tab"] { font-family: 'Orbitron', sans-serif; font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; }
button[data-baseweb="tab"][aria-selected="true"] { color: #FFC400; }
div[data-baseweb="tab-highlight"] { background-color: #FFC400 !important; height: 3px !important; }

/* buttons */
.stButton > button, .stDownloadButton > button {
  font-family: 'Orbitron', sans-serif; letter-spacing: .12em; text-transform: uppercase; font-size: .75rem;
  background: #11151B; color: #FFC400; border: 1px solid #FFC400; border-radius: 0;
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px));
  transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: #FFC400; color: #07090C; box-shadow: 0 0 18px rgba(255,196,0,.45); }
.stButton > button[kind="primary"] { background: #FFC400; color: #07090C; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0B0E12; border-right: 1px solid #2A323D; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-size: 1rem; color: #FFC400; }

/* tables, inputs, alerts */
div[data-testid="stDataFrame"] { border: 1px solid #2A323D; }
div[data-testid="stAlert"] { border-radius: 0; border-left: 4px solid #FFC400; background: #11151B; }
details { border: 1px solid #2A323D !important; border-radius: 0 !important; background: #0E1217; }
::-webkit-scrollbar { width: 10px; height: 10px; } ::-webkit-scrollbar-thumb { background: #2A323D; }
::-webkit-scrollbar-thumb:hover { background: #FFC400; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- cached work
@st.cache_data(show_spinner="Building a simulated warehouse …")
def load_demo(n_skus: int, seed: int):
    wh = make_warehouse(n_skus=n_skus, seed=seed)
    return wh.movements, wh.skus, wh.truth, wh.as_of


@st.cache_data(show_spinner="Reading file …")
def cached_read(data: bytes, filename: str, sheet):
    return read_table(data, filename, sheet)


@st.cache_data(show_spinner="Cleaning movements …")
def cached_standardize(raw, mapping, mode, receipt_values, issue_values, dayfirst):
    return standardize_movements(raw, mapping, mode, receipt_values, issue_values, dayfirst)


@st.cache_data(show_spinner="Analysing every SKU …")
def run_analysis(movements, skus, cfg_kwargs: dict, as_of):
    return analyze(movements, skus, Config(**cfg_kwargs), as_of=pd.Timestamp(as_of))


@st.cache_data(show_spinner="Backtesting on your history …")
def run_backtest(movements, skus, cfg_kwargs: dict, as_of):
    return behavioural_backtest(movements, skus, Config(**cfg_kwargs), as_of=pd.Timestamp(as_of))


@st.cache_resource
def get_ml_model():
    return ml.load(ROOT / "models" / "ml_classifier.joblib")


def template_bytes(name: str) -> bytes:
    p = TEMPLATES / name
    return p.read_bytes() if p.exists() else b""


# --------------------------------------------------------------------------- header
st.markdown(
    """
<div class="dsd-hero">
  <div class="dsd-status">SYSTEM ONLINE</div>
  <div class="dsd-tag">INVENTORY INTELLIGENCE // WAREHOUSE SURVEILLANCE</div>
  <div class="dsd-title">Dead Stock <span>Detector</span></div>
  <div class="dsd-sub">Find stock that is quietly going dead, and tell it apart from stock that is only paused, seasonal or slow.</div>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- data source
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["Demo warehouse", "My own data"], help="The demo is simulated, with known planted answers.")

truth = None
rep = None
if source == "Demo warehouse":
    with st.sidebar:
        demo_n = st.select_slider("Demo size (SKUs)", [500, 1000, 1500, 3000], value=1500)
    movements, skus_master, truth, default_as_of = load_demo(demo_n, 7)
    st.info("You are looking at a **simulated** warehouse (about three years of receipts and picks). "
            "Switch to *My own data* in the sidebar to analyse your products.", icon="🧪")
else:
    with st.sidebar:
        mv_file = st.file_uploader("Stock movements (CSV or Excel)", type=["csv", "tsv", "txt", "xlsx", "xlsm"], key="mv")
        sk_file = st.file_uploader("SKU master (optional)", type=["csv", "tsv", "txt", "xlsx", "xlsm"], key="sk",
                                   help="Unit cost, shelf life, volume, location, category, on-hand quantity, successor SKU.")
        st.download_button("Template: movements.csv", template_bytes("movements_template.csv"), "movements_template.csv", "text/csv")
        st.download_button("Template: sku_master.csv", template_bytes("sku_master_template.csv"), "sku_master_template.csv", "text/csv")
    if mv_file is None:
        st.subheader("Bring your own data")
        st.markdown(
            "**What you need:** a table of stock movements, one row per receipt or pick, with a **SKU**, a **date** and a **quantity**. "
            "That is enough. Add a SKU master (unit cost, shelf life, volume, on-hand quantity, …) for money, expiry and space figures.\n\n"
            "**Three ways to describe movements** (you choose in the next step):\n"
            "- a column that says *receipt* or *issue* (in / out, GRN / dispatch …),\n"
            "- a signed quantity (positive = in, negative = out),\n"
            "- sales only. Then the SKU master must include the current on-hand quantity.\n\n"
            "Use the template buttons in the sidebar to see the layout, or try the demo first."
        )
        st.stop()

    sheet = None
    sheets = list_sheets(mv_file.getvalue(), mv_file.name)
    if len(sheets) > 1:
        sheet = st.selectbox("Sheet with the movements", sheets)
    raw = cached_read(mv_file.getvalue(), mv_file.name, sheet)
    st.subheader("Step 1 · Tell the app what your columns mean")
    st.caption(f"{len(raw):,} rows × {raw.shape[1]} columns read. Suggestions are pre-filled; correct anything that looks wrong.")
    st.dataframe(raw.head(8), hide_index=True)

    sug = suggest_mapping(raw.columns, MOVEMENT_ALIASES)
    options = [NONE] + [str(c) for c in raw.columns]

    def col_select(label: str, field: str, container):
        idx = options.index(str(sug[field])) if sug.get(field) is not None and str(sug[field]) in options else 0
        return container.selectbox(label, options, index=idx, key=f"mv_{field}")

    c1, c2, c3 = st.columns(3)
    sku_col = col_select("SKU / item code", "sku", c1)
    date_col = col_select("Date", "date", c2)
    qty_col = col_select("Quantity", "qty", c3)

    qty_num = pd.to_numeric(raw[qty_col], errors="coerce") if qty_col != NONE else pd.Series(dtype=float)
    default_mode = 0 if sug.get("kind") else (1 if (qty_num < 0).any() else 2)
    MODES = {
        "A column says whether each row is a receipt or an issue": "type_column",
        "Signed quantity (positive = in, negative = out)": "signed",
        "Sales / picks only (every row is an outbound movement)": "issues_only",
    }
    mode_label = st.radio("How does your file show stock coming in and going out?", list(MODES), index=default_mode)
    mode = MODES[mode_label]

    kind_col, receipt_values, issue_values = NONE, None, None
    if mode == "type_column":
        kind_col = col_select("Movement type column", "kind", st)
        if kind_col != NONE:
            vals = pd.unique(raw[kind_col].dropna().astype(str))
            if len(vals) > 60:
                st.error("That column has too many distinct values to be a movement type. Pick another column.")
                st.stop()
            g_rec, g_iss = guess_kind_values(vals)
            r1, r2 = st.columns(2)
            receipt_values = r1.multiselect("Values that mean stock IN (receipt)", list(vals), default=g_rec)
            issue_values = r2.multiselect("Values that mean stock OUT (pick / sale / dispatch)", list(vals), default=g_iss)
    dayfirst = st.checkbox("Dates are day-first (31/12/2025)", value=False)

    missing = [n for n, v in (("SKU", sku_col), ("Date", date_col), ("Quantity", qty_col)) if v == NONE]
    if missing or (mode == "type_column" and (kind_col == NONE or not receipt_values or not issue_values)):
        st.warning("Choose the columns above to continue." + (f" Still missing: {', '.join(missing)}." if missing else ""))
        st.stop()

    skus_master = None
    if sk_file is not None:
        sk_sheet = None
        sk_sheets = list_sheets(sk_file.getvalue(), sk_file.name)
        if len(sk_sheets) > 1:
            sk_sheet = st.selectbox("Sheet with the SKU master", sk_sheets)
        sk_raw = cached_read(sk_file.getvalue(), sk_file.name, sk_sheet)
        sk_sug = suggest_mapping(sk_raw.columns, SKU_ALIASES)
        sk_opts = [NONE] + [str(c) for c in sk_raw.columns]
        with st.expander("SKU master columns (optional fields can stay empty)", expanded=True):
            sk_map = {}
            cols = st.columns(3)
            for i, field in enumerate(["sku"] + SKU_COLUMNS):
                guess = sk_sug.get(field)
                idx = sk_opts.index(str(guess)) if guess is not None and str(guess) in sk_opts else 0
                chosen = cols[i % 3].selectbox(field, sk_opts, index=idx, key=f"sk_{field}")
                sk_map[field] = None if chosen == NONE else chosen
        if not sk_map.get("sku"):
            st.warning("Map the SKU column in the master to continue.")
            st.stop()
        skus_master = standardize_skus(sk_raw, sk_map)

    if mode == "issues_only" and (skus_master is None or skus_master["on_hand_qty"].notna().sum() == 0):
        st.error("With sales-only data the app cannot know what is on the shelf. Upload a SKU master with an "
                 "**on-hand quantity** column and map it above.")
        st.stop()

    mapping = {"sku": sku_col, "date": date_col, "qty": qty_col, "kind": None if kind_col == NONE else kind_col}
    movements, rep = cached_standardize(raw, mapping, mode, tuple(receipt_values or []) or None,
                                        tuple(issue_values or []) or None, dayfirst)
    if rep["rows_kept"] == 0:
        st.error("No usable rows after cleaning. Check the date column, the quantity column and the movement-type values.")
        st.stop()
    default_as_of = rep["date_max"]
    dropped = rep["rows_in"] - rep["rows_kept"]
    st.success(f"{rep['rows_kept']:,} movements for {rep['n_skus']:,} SKUs, {rep['date_min']:%d %b %Y} to {rep['date_max']:%d %b %Y} "
               f"({rep['n_receipts']:,} receipts, {rep['n_issues']:,} issues). "
               + (f"{dropped:,} input rows were merged or dropped." if dropped else ""))
    for n in rep["notes"]:
        st.warning(n)
    for key, label in (("rows_dropped_bad_date", "unreadable dates"), ("rows_dropped_bad_qty", "unreadable quantities"),
                       ("rows_dropped_unknown_type", "unrecognised movement type")):
        if rep.get(key):
            st.caption(f"⚠ {rep[key]:,} rows dropped for {label}.")

# --------------------------------------------------------------------------- settings
with st.sidebar:
    st.header("2 · Settings")
    has_cost = skus_master is not None and "unit_cost" in skus_master and skus_master["unit_cost"].notna().any()
    currency = st.text_input("Currency symbol", "₹") if has_cost else ""
    as_of = st.date_input("Analyse as of", value=pd.Timestamp(default_as_of).date(),
                          max_value=pd.Timestamp(movements["date"].max()).date() + pd.Timedelta(days=60))
    holding = st.number_input("Holding cost (% of stock value per year)", 0.0, 100.0, 20.0, 1.0)
    with st.expander("Detection thresholds"):
        dead_idle = st.slider("Days idle (in stock) before 'dead' can be called", 20, 180, 60,
                              help="Counted only on days the SKU was actually on the shelf.")
        p_unusual = st.select_slider("How unusual must the silence be to call it dead?", [0.01, 0.02, 0.05, 0.10], value=0.05,
                                     help="Chance, under the SKU's own history, of a quiet spell this long. Lower = fewer false alarms, later flags.")
        watch_idle = st.slider("Days idle before a SKU can be on the watch list", 7, 60, 14)
        active_days = st.slider("Pick days in 180 to count as 'active'", 4, 40, 12)
        never_days = st.slider("Never picked for this many days → forgotten", 60, 365, 120)
        min_hist = st.slider("Minimum history to judge a SKU (days)", 30, 365, 90)
        naive_days = st.slider("Compare with the naive 'no pick in N days' rule", 30, 180, 90)
    show_ml = False
    if get_ml_model() is not None:
        show_ml = st.checkbox("Show ML second opinion", value=False,
                              help="Gradient boosting trained on simulated warehouses. Secondary view only.")

cfg_kwargs = dict(
    dead_min_idle_days=dead_idle, p_unusual=p_unusual, watch_min_idle_days=watch_idle, active_pick_days_180=active_days,
    never_moved_days=never_days, min_history_days=min_hist, naive_days=naive_days,
    holding_cost_rate=holding / 100.0, currency=currency,
)
as_of_ts = pd.Timestamp(as_of)

an = run_analysis(movements, skus_master, cfg_kwargs, as_of_ts)
res, summ, cfg = an.result, an.summary, an.cfg
money = lambda x: fmt_money(x, currency)
unit_word = "value" if has_cost else "units"

ml_pred = None
if show_ml:
    ml_pred = ml.predict(get_ml_model(), res)
    res = res.join(ml_pred)

# --------------------------------------------------------------------------- tabs
tab_over, tab_act, tab_sku, tab_trust, tab_how = st.tabs(
    ["Overview", "Action list", "SKU explorer", "Can I trust this?", "How it works"])

# ---- Overview
with tab_over:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SKUs analysed", f"{summ['n_skus']:,}", f"{summ['n_judged']:,} judged", delta_color="off", delta_arrow="off", border=True)
    k2.metric("Forgotten + obsolete", f"{summ['n_forgotten'] + summ['n_obsolete']:,}",
              f"{summ['n_forgotten']:,} forgotten · {summ['n_obsolete']:,} obsolete", delta_color="off", delta_arrow="off", border=True)
    k3.metric(f"Stock {unit_word} in dead stock", money(summ["capital_dead"]), f"{summ['capital_dead_share']:.1%} of all stock",
              delta_color="off", delta_arrow="off", border=True)
    k4.metric("Holding cost of dead stock / yr", money(summ["holding_cost_dead"]),
              f"{summ['n_conflict']:,} still re-ordered", delta_color="inverse" if summ["n_conflict"] else "off", delta_arrow="off", border=True)
    if np.isfinite(summ["space_dead_m3"]) and summ["space_dead_m3"] > 0:
        st.caption(f"Dead stock occupies about **{summ['space_dead_m3']:,.0f} m³** of storage."
                   + (f" Stock past its shelf life is worth **{money(summ['expiry_exposure'])}**." if summ["expiry_exposure"] > 0 else ""))
    elif summ["expiry_exposure"] > 0:
        st.caption(f"Stock past its shelf life is worth **{money(summ['expiry_exposure'])}**.")

    present = [s for s in ORDER if s in set(res["status"])]
    counts = res["status"].value_counts().reindex(present)
    cap = res.groupby("status")["capital_tied"].sum().reindex(present)
    left, right = st.columns(2)
    for col, series, title in ((left, counts, "SKUs by behaviour"), (right, cap, f"Stock {unit_word} by behaviour")):
        fig = go.Figure(go.Bar(x=series.values, y=[CLASS_LABELS[s] for s in series.index], orientation="h",
                               marker_color=[CLASS_COLORS[s] for s in series.index],
                               hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
        fig.update_layout(title=title, height=340, margin=dict(l=0, r=10, t=40, b=0), yaxis=dict(autorange="reversed"))
        col.plotly_chart(fig)
    if res["category"].notna().any():
        dead = res[res["dead_flag"]].groupby("category")["capital_tied"].sum().sort_values(ascending=False).head(12)
        if len(dead):
            fig = go.Figure(go.Bar(x=dead.index, y=dead.values, marker_color=CLASS_COLORS["forgotten"]))
            fig.update_layout(title=f"Dead stock {unit_word} by category", height=320, margin=dict(l=0, r=10, t=40, b=0))
            st.plotly_chart(fig)
    if truth is not None:
        with st.expander("Peek at the planted truth (demo only)"):
            t = truth.reindex(res.index)["truth_state"]
            keep = ~res["status"].isin([STATUS_OUT_OF_STOCK])
            ct = pd.crosstab(t[keep].rename("Planted truth"), res.loc[keep, "status"].rename("Engine says"))
            st.dataframe(ct)
            st.caption("Temporarily-inactive SKUs that have been silent 60+ days are the hard case: at that moment they look exactly like dead ones.")

# ---- Action list
with tab_act:
    st.write("SKUs ranked by money at risk. Filter, then export the list for the warehouse team.")
    present = [s for s in ORDER if s in set(res["status"])]
    default = [s for s in ["forgotten", "obsolete", "temporarily_inactive"] if s in present]
    f1, f2, f3, f4 = st.columns([2, 2, 1.4, 1.4])
    pick = f1.multiselect("Show", present, default=default, format_func=lambda s: CLASS_LABELS[s])
    cats = sorted(res["category"].dropna().astype(str).unique())
    catsel = f2.multiselect("Category", cats) if cats else []
    min_cap = f3.number_input(f"Min. {unit_word} tied up", 0.0, value=0.0)
    sort_by = f4.selectbox("Sort by", ["Value at risk", "Capital tied up", "Risk score", "Idle days"])
    g1, g2 = st.columns([1, 3])
    conflict_only = g1.checkbox("Still being re-ordered")
    query = g2.text_input("Search SKU or description")

    m = res["status"].isin(pick)
    if catsel:
        m &= res["category"].astype(str).isin(catsel)
    m &= res["capital_tied"] >= min_cap
    if conflict_only:
        m &= res["replenishment_conflict"]
    if query.strip():
        q = query.strip().lower()
        m &= res.index.to_series().str.lower().str.contains(q, regex=False) | res["description"].astype(str).str.lower().str.contains(q, regex=False)
    key = {"Value at risk": "value_at_risk", "Capital tied up": "capital_tied", "Risk score": "risk_score", "Idle days": "idle_instock_days"}[sort_by]
    sel = res[m].sort_values(key, ascending=False)
    view = friendly_table(sel)
    if show_ml and ml_pred is not None:
        view["ML says"] = sel["ml_class"].map(CLASS_LABELS)
        view["ML agrees"] = sel["ml_class"].eq(sel["status"])
    st.caption(f"{len(sel):,} SKUs · {money(sel['capital_tied'].sum())} {unit_word} tied up · {money(sel['holding_cost_per_year'].sum())} holding cost per year")
    show_cols = [c for c in view.columns if view[c].notna().any() or c in ("Status", "Risk score")]
    st.dataframe(
        view[show_cols].head(2000),
        column_config={
            "Risk score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f"),
            "Capital tied up": st.column_config.NumberColumn(format="%.0f"),
            "Holding cost / yr": st.column_config.NumberColumn(format="%.0f"),
            "Unit cost": st.column_config.NumberColumn(format="%.2f"),
            "On hand": st.column_config.NumberColumn(format="%.0f"),
            "Cover (days)": st.column_config.NumberColumn(format="%.0f"),
            "Idle days (in stock)": st.column_config.NumberColumn(format="%.0f"),
            "Days since last pick": st.column_config.NumberColumn(format="%.0f"),
            "Oldest stock (days)": st.column_config.NumberColumn(format="%.0f"),
            "Still being replenished": st.column_config.CheckboxColumn(),
            "Why": st.column_config.TextColumn(width="large"),
            "Recommended action": st.column_config.TextColumn(width="large"),
        },
    )
    if len(sel) > 2000:
        st.caption("Showing the first 2,000 rows; the downloads contain all of them.")
    d1, d2, d3 = st.columns([1, 1, 3])
    d1.download_button("Download CSV", view.to_csv().encode("utf-8-sig"), "forgotten_inventory_action_list.csv", "text/csv")
    if d2.button("Build Excel report"):
        st.session_state["xlsx"] = to_excel_bytes(res, summ, cfg)
    if "xlsx" in st.session_state:
        d3.download_button("⬇ Download Excel report", st.session_state["xlsx"], "forgotten_inventory_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---- SKU explorer
with tab_sku:
    ordered = res.sort_values("risk_score", ascending=False).index.tolist()
    s1, s2 = st.columns([2, 2])
    typed = s2.text_input("…or type an exact SKU code", "")
    chosen = s1.selectbox("Pick a SKU (highest risk first)", ordered[:3000])
    sku = typed.strip() if typed.strip() in res.index else chosen
    if typed.strip() and typed.strip() not in res.index:
        s2.caption("SKU not found; showing the selection on the left.")
    row = res.loc[sku]
    st.markdown(f"### {sku}" + (f" · {row['description']}" if isinstance(row["description"], str) and row["description"] else ""))
    st.markdown(f"**{CLASS_LABELS[row['status']]}** · risk score **{row['risk_score']:.0f}/100**")
    st.write(row["reason"])
    st.info(row["action"], icon="🎯")
    if truth is not None and sku in truth.index:
        st.caption(f"Planted truth (demo only): **{truth.loc[sku, 'truth_state']}**")
    a, b, c, d, e = st.columns(5)
    a.metric("On hand", f"{row['on_hand']:,.0f}")
    b.metric("Capital tied up", money(row["capital_tied"]))
    c.metric("Idle days (in stock)", f"{row['idle_instock_days']:,.0f}")
    d.metric("Pick days, last 180", f"{row['pick_days_180']:,.0f}")
    e.metric("Cover (days)", "∞" if not np.isfinite(row["cover_days"]) else f"{row['cover_days']:,.0f}")

    tl = sku_timeline(an.movements, sku, an.as_of, days=540)
    if len(tl):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                            subplot_titles=("Stock on hand", "Daily picks and receipts"))
        colr = CLASS_COLORS.get(row["status"], "#4c78a8")
        fig.add_trace(go.Scatter(x=tl["date"], y=tl["stock"], mode="lines", line=dict(shape="hv", color=colr, width=2),
                                 fill="tozeroy", name="Stock", hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=tl["date"], y=tl["picked"], name="Picked", marker_color=colr), row=2, col=1)
        fig.add_trace(go.Bar(x=tl["date"], y=tl["received"], name="Received", marker_color="#9aa5b1"), row=2, col=1)
        if pd.notna(row["last_pick_date"]) and (an.as_of - row["last_pick_date"]).days >= 1:
            fig.add_shape(type="rect", xref="x", yref="paper", x0=row["last_pick_date"], x1=an.as_of, y0=0, y1=1,
                          fillcolor=CLASS_COLORS["forgotten"], opacity=0.10, line_width=0, layer="below")
            fig.add_annotation(x=row["last_pick_date"], y=0.93, xref="x", yref="paper", text="silent since last pick",
                               showarrow=False, xanchor="left", yanchor="top", font=dict(size=11))
        fig.update_layout(height=470, barmode="group", margin=dict(l=0, r=10, t=40, b=0), legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig)
    else:
        st.caption("No movements for this SKU.")
    with st.expander("All numbers behind this verdict"):
        keep_cols = ["status", "idle_instock_days", "p_idle", "p_exp", "p_lognormal", "n_gaps", "mean_gap", "max_gap",
                     "demand_pattern", "pick_days_30", "pick_days_90", "pick_days_180", "pick_days_365", "erosion_ratio",
                     "season_assessable", "season_corr", "season_top3", "ly_quiet_ratio", "stock_age_oldest", "life_used",
                     "days_to_expiry", "expired_qty", "receipts_90", "replenishment_conflict"]
        st.dataframe(row[keep_cols].astype(str).rename("value").to_frame())

# ---- Trust
with tab_trust:
    st.subheader("Data checks")
    if not an.warnings and not an.notes:
        st.success("No data-quality issues found.")
    for w in an.warnings:
        st.warning(w)
    for n in an.notes:
        st.info(n)

    st.subheader("Backtest on your own history")
    st.write(f"The app goes back {cfg.horizon_days} days, classifies every SKU as it would have looked then, and checks what "
             "really happened next. If the dead-stock list is any good, those SKUs should have stayed untouched. No labels needed.")
    horizon = st.select_slider("Look-ahead window (days)", [30, 60, 90, 120, 180], value=90)
    if st.button("Run backtest", type="primary") or st.session_state.get("bt_done"):
        st.session_state["bt_done"] = True
        bt = run_backtest(an.movements, skus_master, {**cfg_kwargs, "horizon_days": horizon}, an.as_of)
        if bt is None:
            st.info("Not enough history for a backtest yet: it needs roughly "
                    f"{cfg.min_history_days + 120 + horizon} days of movements.")
        else:
            tb = bt["table"].set_index("group")
            eng, nv = tb.loc["Engine: forgotten + obsolete"], tb.loc[f"Naive rule: no pick in {cfg.naive_days} days"]
            allj = tb.loc["All judged SKUs"]
            st.success(
                f"Classified as of {bt['t0']:%d %b %Y}. Of the **{int(eng['n_skus']):,}** SKUs flagged forgotten or obsolete, "
                f"**{eng['stayed_dormant_pct']:.0f}%** had no pick in the next {horizon} days. "
                f"The naive 'no pick in {cfg.naive_days} days' rule flagged {int(nv['n_skus']):,} SKUs, of which "
                f"{nv['stayed_dormant_pct']:.0f}% stayed dormant; across all SKUs {allj['stayed_dormant_pct']:.0f}% had no pick."
            )
            st.caption(f"The naive rule flagged {bt['naive_only_flagged']:,} SKUs the engine did not; "
                       f"{bt['naive_false_alarms_avoided']:,} of those were picked again (false alarms avoided).")
            show = tb[["n_skus", "stayed_dormant_pct", "picked_again_pct", "capital_tied"]].rename(columns={
                "n_skus": "SKUs", "stayed_dormant_pct": "% stayed dormant", "picked_again_pct": "% picked again",
                "capital_tied": f"Stock {unit_word}"})
            st.dataframe(show, column_config={
                "% stayed dormant": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                "% picked again": st.column_config.NumberColumn(format="%.0f%%"),
                f"Stock {unit_word}": st.column_config.NumberColumn(format="%.0f")})
            st.caption("Read it like this: a good detector has a high '% stayed dormant' for forgotten/obsolete, "
                       "a middling one for the watch list, and a very low one for active.")

    if show_ml and ml_pred is not None:
        st.subheader("ML second opinion vs the engine")
        jm = ~res["status"].isin([STATUS_OUT_OF_STOCK, STATUS_INSUFFICIENT])
        agree = (res.loc[jm, "ml_class"] == res.loc[jm, "status"]).mean()
        st.write(f"The two views agree on **{agree:.0%}** of judged SKUs. The model was trained on simulated warehouses only, "
                 "so treat disagreement as a prompt to look at the SKU, not as a verdict.")
        st.dataframe(pd.crosstab(res.loc[jm, "status"].map(CLASS_LABELS).rename("Engine"),
                                 res.loc[jm, "ml_class"].map(CLASS_LABELS).rename("ML")))

# ---- How it works
with tab_how:
    st.markdown(
        """
**The question.** For every product in stock: is it merely quiet, or has it been forgotten?

**1 · Count idle time on the shelf.** A SKU that was out of stock for two months wasn't neglected for two months. Quiet spells are counted only on days the SKU was actually in stock.

**2 · Judge each SKU against its own rhythm.** From a SKU's past gaps between picks, the app asks: *how likely is a quiet spell this long?* Two models (exponential and log-normal) must both agree it is unusual, so a SKU that naturally sells once a quarter is not flagged for being quiet for 40 days.

**3 · Check seasonality.** If a SKU has a strong yearly rhythm and was just as quiet at this point last year, the silence is *expected* and it is labelled seasonal. This needs about two years of history.

**4 · Look for obsolescence.** Stock that cannot be sold before its shelf life ends, a successor SKU that has taken over, or demand that was already fading before it stopped.

| Class | Meaning |
|---|---|
| **Forgotten** | A long, statistically unusual silence, or stock that arrived long ago and was never picked. |
| **Obsolete** | Dead *and* has no future: expiring stock, a successor, or demand that faded out. |
| **Temporarily inactive (watch)** | Unusually quiet, not yet long enough to call it dead. |
| **Slow-moving** | Few picks, but the current gap is normal for that rhythm. |
| **Seasonal** | Strong yearly pattern and equally quiet this time last year. |
| **Active** | Picked regularly. |

**Risk score (0–100)** blends how unusual the silence is, how old the stock is, how many days of cover remain, and expiry. *Still being re-ordered* flags dead or watch items that purchasing keeps replenishing.

**Limits.** Thresholds are defaults, not truths; tune them in the sidebar and use the backtest to see the effect on your own history. A product that is genuinely paused for a long time looks identical to a dead one until it resumes: the app surfaces it for investigation rather than claiming certainty. Money figures use the unit costs you supply.
        """
    )
