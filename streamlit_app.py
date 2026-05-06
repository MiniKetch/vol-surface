"""Vol Surface — interactive Streamlit dashboard.

Run:
    pip install -e ".[data,viz,dashboard]"
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

import dataclasses

from vol_surface.analytics import (
    compute_realized_vol,
    compute_skew_metrics,
    render_realized_vs_implied,
)
from vol_surface.data import (
    DataFetcher,
    RiskFreeCurve,
    fetch_earnings,
    flag_event_expiries,
)
from vol_surface.scanner import MispricingReport, scan_mispricing
from vol_surface.strategy import compute_payoff, parse_strategy, render_payoff
from vol_surface.utils import attach_attrs, extract_attrs, get_attr
from vol_surface.viz import (
    render_smile,
    render_smile_evolution,
    render_surface,
    render_term_structure,
)


# ============================================================================
# Page setup
# ============================================================================

st.set_page_config(
    page_title="Vol Surface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 28px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem; padding: 0.6rem 1.2rem;
    }
    [data-testid="stMetricValue"] { font-size: 2.0rem; }
    [data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.75; }
    .vsc-tall .js-plotly-plot { min-height: 720px; }
    hr { border-color: rgba(255,255,255,0.08); }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Live options vol surface")
st.markdown(
    "<p style='font-size:1.1rem; opacity:0.8; margin-top:-0.5rem;'>"
    "Black-Scholes implied vol via a C++ kernel · SVI smile fit · "
    "skew metrics · strategy P&L · realized-vs-implied vol."
    "</p>",
    unsafe_allow_html=True,
)


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("Underlying")
    ticker = st.text_input(
        "Ticker symbol", value="SPY",
        help="Any US-listed ticker with options. Try SPY, QQQ, AAPL, NVDA, TSLA.",
    ).strip().upper()

    st.divider()
    st.header("Chain filters")
    max_expiries = st.slider("Number of expiries", 4, 20, 12,
                              help="Stratified across the term structure.")
    min_oi  = st.slider("Minimum open interest", 0, 100, 5)
    min_dte = st.slider("Minimum days to expiry", 1, 60, 7)

    st.divider()
    st.header("View")

    SIDE_LABELS = {
        "OTM": "Out-of-the-money wings (recommended)",
        "calls": "Calls only",
        "puts":  "Puts only",
        "both":  "All contracts",
    }
    side = st.radio("Which contracts", list(SIDE_LABELS),
                     format_func=SIDE_LABELS.__getitem__, index=0)
    st.caption(
        "Affects every tab. Ranking & top-N controls live inside the "
        "**Top deviations** tab itself — they only apply there."
    )

    # Defaults for the table tab; the actual widgets live in that tab.
    METRIC_LABELS = {
        "residual_iv": "Absolute IV miss (recommended)",
        "zscore":      "Spread-normalised z-score",
        "dollar":      "Dollar P&L impact",
    }

    st.divider()
    rate_curve = _rate_curve()
    st.caption(
        "Live rates from FRED" if rate_curve.source == "fred"
        else "Bundled Treasury snapshot · set FREDAPI_KEY for live"
    )


# ============================================================================
# Cached fetch + analytics
#
# Streamlit's @st.cache_data pickles return values, which loses
# DataFrame.attrs and chokes on custom dataclasses. We work around
# that by:
#   - returning (df, attrs_dict) from cached chain/fit fns and
#     re-attaching attrs at the call site;
#   - returning primitive tuples for tiny dataclasses (earnings);
#   - using @st.cache_resource for stateful objects (RiskFreeCurve).
# ============================================================================

@st.cache_resource(show_spinner=False)
def _rate_curve() -> RiskFreeCurve:
    """RiskFreeCurve holds NumPy arrays + a string `source` flag — no
    reason to re-instantiate per render. cache_resource keeps the
    object identity stable across reruns."""
    return RiskFreeCurve()


@st.cache_data(ttl=300, show_spinner="Pulling options chain…")
def _load_chain_raw(ticker: str, max_expiries: int, min_oi: int, min_dte: int):
    """Raw cached fetch. Returns (DataFrame, attrs_dict) so the
    caller can re-attach .attrs after Streamlit's pickle round-trip."""
    fetcher = DataFetcher(use_cache=False)
    df = fetcher.get(
        ticker, max_expiries=max_expiries,
        min_open_interest=min_oi, min_dte_days=min_dte,
    )
    return df, extract_attrs(df)


def load_chain(ticker: str, max_expiries: int, min_oi: int, min_dte: int):
    """Public wrapper — calls the cached fn and rehydrates .attrs."""
    df, attrs = _load_chain_raw(ticker, max_expiries, min_oi, min_dte)
    return attach_attrs(df, attrs)


@st.cache_data(ttl=300, show_spinner="Fitting SVI per expiry…")
def _fit_scanner_raw(_df: pd.DataFrame, side: str):
    """Raw cached fit. Returns (scored_df, fits_dict, attrs) so that
    the report's DataFrame can be reconstructed with metadata intact."""
    report = scan_mispricing(_df, side=side)
    return report.scored, report.fits, extract_attrs(report.scored)


def fit_scanner(df: pd.DataFrame, side: str) -> MispricingReport:
    scored, fits, attrs = _fit_scanner_raw(df, side)
    attach_attrs(scored, attrs)
    return MispricingReport(scored=scored, fits=fits)


@st.cache_data(ttl=900, show_spinner="Looking up earnings calendar…")
def get_earnings(ticker: str):
    """Return primitives — Streamlit's cache_data won't serialize
    our custom EarningsInfo dataclass."""
    info = fetch_earnings(ticker)
    return (info.next_date, info.source)


@st.cache_data(ttl=600, show_spinner="Pulling historical prices…")
def get_realized_vol(ticker: str, window: int):
    return compute_realized_vol(ticker, period="1y", window_days=window)


# ============================================================================
# Data fetch
# ============================================================================

try:
    df = load_chain(ticker, max_expiries, min_oi, min_dte)
except (RuntimeError, ValueError, KeyError) as exc:
    st.error(f"Couldn't fetch **{ticker}**: {exc}")
    st.info("Common typos: 'APPL' → 'AAPL', 'NVIDIA' → 'NVDA'.")
    st.stop()

# Use get_attr() consistently — survives any future pandas .attrs
# regression more gracefully than raw indexing.
spot       = float(get_attr(df, "spot", 0.0))
n_iv       = int(df["iv"].notna().sum())
recovery   = n_iv / max(len(df), 1) * 100
fetched_at = get_attr(df, "fetched_at")

# Single SVI fit, shared across every tab below — instead of one
# fit_scanner call per tab. Streamlit's cache makes repeated calls
# fast on hits, but the deserialization + .attrs reattachment is
# pure waste when one call answers them all.
report = fit_scanner(df, side)

# Earnings — cheap, do it once per ticker.
earnings_next_date, earnings_source = get_earnings(ticker)
earnings_flags = flag_event_expiries(
    sorted(df["expiry"].dropna().unique()),
    earnings_next_date,
)

# --- Top metric strip ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ticker", ticker)
c2.metric("Spot", f"${spot:,.2f}")
c3.metric("Contracts", f"{len(df):,}")
c4.metric("IVs solved", f"{n_iv:,}", delta=f"{recovery:.1f}% recovery")
if earnings_next_date:
    days_out = (earnings_next_date - pd.Timestamp.now().date()).days
    c5.metric("Next earnings", earnings_next_date.isoformat(),
               delta=f"in {days_out}d" if days_out >= 0 else f"{-days_out}d ago")
else:
    c5.metric("Next earnings", "—", delta="not announced", delta_color="off")

if fetched_at is not None:
    when = (fetched_at if isinstance(fetched_at, str)
            else fetched_at.strftime("%Y-%m-%d %H:%M UTC"))
    st.caption(f"Quotes fetched {when}.")

st.write("")


# ============================================================================
# Strategy default — pick smart ATM strikes from the actual chain
# ============================================================================

def _suggest_default_strategy(df: pd.DataFrame) -> str:
    """Generate a plausible default position string for the strategy
    tab. Picks a 1-strike-wide call spread on a near-front-month
    expiry, using strikes that *actually exist* in the chain. Works
    for any ticker scale: SPY → 730/740, AAPL → 285/290, NVDA → 130/135.
    """
    spot = float(get_attr(df, "spot", 0.0))
    if spot <= 0 or df.empty or "expiry" not in df.columns:
        return "long 1 100C 2026-06-18"  # safe fallback

    # Pick the front-ish expiry (skip the very first if there are several).
    expiries = sorted(df["expiry"].dropna().unique())
    if not expiries:
        return "long 1 100C 2026-06-18"
    expiry_idx = min(2, len(expiries) - 1)
    expiry = pd.Timestamp(expiries[expiry_idx]).date().isoformat()

    # Call strikes for that expiry, sorted; pick the closest to spot
    # (lower leg) and the next strike up (upper leg).
    call_strikes = sorted(
        df[(df["type"] == "C") & (pd.to_datetime(df["expiry"]).dt.date.astype(str) == expiry)]
        ["strike"].dropna().unique()
    )
    if len(call_strikes) < 2:
        return f"long 1 {round(spot)}C {expiry}"

    lower_idx = min(range(len(call_strikes)),
                     key=lambda i: abs(call_strikes[i] - spot))
    upper_idx = min(lower_idx + 1, len(call_strikes) - 1)
    K_low  = call_strikes[lower_idx]
    K_high = call_strikes[upper_idx] if upper_idx > lower_idx else K_low

    fmt = lambda k: f"{k:g}"  # drop trailing zeros — "740" not "740.0"
    return f"long 10 {fmt(K_low)}C {expiry}, short 10 {fmt(K_high)}C {expiry}"


# ============================================================================
# Strategy preset builder — pick template + dropdowns, get a position string
# ============================================================================

def _build_strategy_from_preset(preset: str, df: pd.DataFrame) -> str:
    """Render the per-preset parameter widgets and assemble the
    canonical position string. Each preset shows whichever subset of
    inputs it needs (qty, expiry, strike picker(s)) — strikes are
    pulled from the actual chain so they're always real."""
    spot = float(get_attr(df, "spot", 100.0))
    expiries = sorted(df["expiry"].dropna().unique())
    if not expiries:
        return "long 1 100C 2026-06-18"

    # Common controls across every preset.
    c_qty, c_exp = st.columns([1, 2])
    with c_qty:
        qty = st.number_input("Quantity (per leg)",
                               min_value=1, value=10, step=1)
    with c_exp:
        labels = [pd.Timestamp(e).date().isoformat() for e in expiries]
        idx = min(2, len(labels) - 1)
        chosen = st.selectbox("Expiry", labels, index=idx,
                               key="strat_expiry")
    expiry = chosen
    expiry_ts = expiries[labels.index(chosen)]

    def _strikes_for(opt_type: str) -> list[float]:
        sel = df[(df["type"] == opt_type) & (df["expiry"] == expiry_ts)]
        return sorted(sel["strike"].dropna().unique().tolist())

    def _closest_idx(strikes: list[float], target: float) -> int:
        if not strikes: return 0
        return min(range(len(strikes)),
                    key=lambda i: abs(strikes[i] - target))

    fmt = lambda k: f"{float(k):g}"  # drop trailing zeros

    # ----------------- Single-leg presets -----------------
    if preset == "Long call":
        Ks = _strikes_for("C")
        if not Ks:
            return "long 1 100C 2026-06-18"
        K = st.selectbox("Strike", Ks, index=_closest_idx(Ks, spot),
                          key="strat_K_call",
                          help="At-the-money picked by default.")
        return f"long {qty} {fmt(K)}C {expiry}"

    if preset == "Long put":
        Ks = _strikes_for("P")
        if not Ks:
            return "long 1 100P 2026-06-18"
        K = st.selectbox("Strike", Ks, index=_closest_idx(Ks, spot),
                          key="strat_K_put")
        return f"long {qty} {fmt(K)}P {expiry}"

    # ----------------- Spread presets ---------------------
    if preset == "Call spread (bullish)":
        Ks = _strikes_for("C")
        if len(Ks) < 2:
            return "long 1 100C 2026-06-18"
        c1, c2 = st.columns(2)
        with c1:
            i_long = st.selectbox(
                "Long call strike (lower, you buy)",
                Ks, index=_closest_idx(Ks, spot),
                key="strat_call_long",
            )
        with c2:
            # Default short strike: one above the long.
            default_short = min(_closest_idx(Ks, spot) + 1, len(Ks) - 1)
            i_short = st.selectbox(
                "Short call strike (higher, you sell)",
                Ks, index=default_short,
                key="strat_call_short",
            )
        return (f"long {qty} {fmt(i_long)}C {expiry}, "
                f"short {qty} {fmt(i_short)}C {expiry}")

    if preset == "Put spread (bearish)":
        Ks = _strikes_for("P")
        if len(Ks) < 2:
            return "long 1 100P 2026-06-18"
        c1, c2 = st.columns(2)
        with c1:
            default_long = _closest_idx(Ks, spot)
            i_long = st.selectbox(
                "Long put strike (higher, you buy)",
                Ks, index=default_long,
                key="strat_put_long",
            )
        with c2:
            default_short = max(default_long - 1, 0)
            i_short = st.selectbox(
                "Short put strike (lower, you sell)",
                Ks, index=default_short,
                key="strat_put_short",
            )
        return (f"long {qty} {fmt(i_long)}P {expiry}, "
                f"short {qty} {fmt(i_short)}P {expiry}")

    # ----------------- Volatility presets -----------------
    if preset == "Long straddle":
        # Same strike on both call + put, ATM by default.
        Ks_c = _strikes_for("C")
        Ks_p = _strikes_for("P")
        Ks = sorted(set(Ks_c) & set(Ks_p))
        if not Ks:
            return "long 1 100C 2026-06-18, long 1 100P 2026-06-18"
        K = st.selectbox("Strike (same for both)", Ks,
                          index=_closest_idx(Ks, spot),
                          key="strat_strad",
                          help="Both legs at ATM by default — bet on volatility.")
        return (f"long {qty} {fmt(K)}C {expiry}, "
                f"long {qty} {fmt(K)}P {expiry}")

    if preset == "Long strangle":
        Ks_c = _strikes_for("C")
        Ks_p = _strikes_for("P")
        if not Ks_c or not Ks_p:
            return "long 1 110C 2026-06-18, long 1 90P 2026-06-18"
        c1, c2 = st.columns(2)
        with c1:
            i_call = _closest_idx(Ks_c, spot * 1.05)
            K_call = st.selectbox("OTM call strike", Ks_c, index=i_call,
                                    key="strat_strg_call")
        with c2:
            i_put = _closest_idx(Ks_p, spot * 0.95)
            K_put = st.selectbox("OTM put strike", Ks_p, index=i_put,
                                   key="strat_strg_put")
        return (f"long {qty} {fmt(K_call)}C {expiry}, "
                f"long {qty} {fmt(K_put)}P {expiry}")

    # ----------------- Iron condor (4 legs) ---------------
    if preset == "Iron condor":
        Ks_c = _strikes_for("C")
        Ks_p = _strikes_for("P")
        if len(Ks_c) < 2 or len(Ks_p) < 2:
            return "long 1 110C 2026-06-18, short 1 105C 2026-06-18, " \
                   "long 1 90P 2026-06-18, short 1 95P 2026-06-18"
        st.caption("Sell premium close to ATM, buy wings further out.")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        with c1:
            i = _closest_idx(Ks_p, spot * 0.97)
            K_short_put = st.selectbox(
                "Short put (sold, near ATM)", Ks_p, index=i,
                key="ic_short_put",
            )
        with c2:
            i = _closest_idx(Ks_p, spot * 0.92)
            K_long_put = st.selectbox(
                "Long put (bought, lower)", Ks_p, index=i,
                key="ic_long_put",
            )
        with c3:
            i = _closest_idx(Ks_c, spot * 1.03)
            K_short_call = st.selectbox(
                "Short call (sold, near ATM)", Ks_c, index=i,
                key="ic_short_call",
            )
        with c4:
            i = _closest_idx(Ks_c, spot * 1.08)
            K_long_call = st.selectbox(
                "Long call (bought, higher)", Ks_c, index=i,
                key="ic_long_call",
            )
        return (
            f"short {qty} {fmt(K_short_put)}P {expiry}, "
            f"long {qty} {fmt(K_long_put)}P {expiry}, "
            f"short {qty} {fmt(K_short_call)}C {expiry}, "
            f"long {qty} {fmt(K_long_call)}C {expiry}"
        )

    # Fallback — shouldn't reach here.
    return _suggest_default_strategy(df)


# ============================================================================
# Snapshot helper
# ============================================================================

def _snapshot_button(fig, *, key: str, default_name: str):
    """Render a download button that exports the current Plotly figure
    as PNG via kaleido. Quietly disabled if kaleido isn't installed."""
    try:
        png_bytes = fig.to_image(format="png", width=1600, height=1000, scale=2)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"📷 PNG snapshot unavailable ({exc.__class__.__name__}). "
                   "Install kaleido: `pip install kaleido`.")
        return
    st.download_button(
        "📷  Snapshot for socials (PNG)",
        data=png_bytes,
        file_name=default_name,
        mime="image/png",
        key=key,
    )


# ============================================================================
# Tabs
# ============================================================================

tabs = st.tabs([
    "🌐  3D surface",
    "🌙  Smile slices",
    "📈  Term structure",
    "🌈  Smile evolution",
    "⚖️  Skew metrics",
    "💼  Strategy P&L",
    "📊  Realized vs implied",
    "📋  Top deviations",
])
(tab_surface, tab_smile, tab_term, tab_evo,
 tab_skew, tab_strat, tab_rv, tab_table) = tabs


# --------------------------- 3D surface tab --------------------------------

with tab_surface:
    show_dots = st.checkbox("Show individual contracts as dots", True,
                             key="surf_dots")
    show_wire = st.checkbox("Show wireframe overlay", True, key="surf_wire")
    fig = render_surface(df, side=side,
                         show_data_points=show_dots,
                         show_wireframe=show_wire)
    st.markdown("<div class='vsc-tall'>", unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", height=720, key="surface_chart")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        "x = log moneyness ln(K / F)  ·  y = time to expiry (years)  ·  "
        "z = implied volatility. Plasma colormap: dark base = low IV, "
        "glowing peaks = high IV."
    )
    _snapshot_button(fig, key="snap_surface",
                      default_name=f"{ticker}_surface.png")


# --------------------------- Smile tab -------------------------------------

with tab_smile:
    expiries = sorted(df["expiry"].dropna().unique())
    if not expiries:
        st.warning("No expiries — relax the min OI / DTE filters.")
    else:
        # Annotate expiries in the picker with 📅 if they straddle earnings.
        def _label(e):
            ts = pd.Timestamp(e)
            base = ts.date().isoformat()
            return f"📅 {base} (earnings)" if earnings_flags.get(ts) else base

        labels = [_label(e) for e in expiries]
        col_pick, col_resid = st.columns([2, 1])
        with col_pick:
            chosen_label = st.selectbox("Expiry", labels,
                                         index=min(2, len(labels) - 1))
        with col_resid:
            show_resid_strip = st.checkbox(
                "Residuals strip", True,
                help="Bar chart below showing IV residuals from the SVI fit.",
            )

        chosen = expiries[labels.index(chosen_label)]
        params = report.fits.get(pd.Timestamp(chosen))

        smile_fig = render_smile(df, chosen, side=side, svi_params=params,
                                  show_residuals=show_resid_strip)
        st.plotly_chart(smile_fig, width="stretch", height=620,
                         key="smile_chart")

        if params is not None:
            arb_ok = "✓ arbitrage-free" if params.is_arbitrage_free() else "✗ NOT arb-free"
            cols = st.columns(6)
            cols[0].metric("a (level)",     f"{params.a:+.4f}")
            cols[1].metric("b (slope)",     f"{params.b:.3f}")
            cols[2].metric("ρ (skew)",      f"{params.rho:+.2f}")
            cols[3].metric("m (vertex)",    f"{params.m:+.3f}")
            cols[4].metric("σ (curvature)", f"{params.sigma:.3f}")
            cols[5].metric("Fit RSS",       f"{params.rss:.5f}",
                           delta=arb_ok, delta_color="off")
        _snapshot_button(smile_fig, key="snap_smile",
                          default_name=f"{ticker}_smile_{pd.Timestamp(chosen).date()}.png")


# --------------------------- Term structure --------------------------------

with tab_term:
    fig = render_term_structure(df, fits=report.fits)
    st.plotly_chart(fig, width="stretch", height=560, key="term_chart")
    st.caption(
        "ATM IV per expiry. Cyan dots = market quotes (IV at the strike "
        "closest to the forward). Gold diamonds = SVI fitted ATM "
        "(the smooth model's read on \"true\" ATM). Gaps between the "
        "two are typically explained by event-vol that smooth surfaces "
        "can't capture."
    )
    _snapshot_button(fig, key="snap_term",
                      default_name=f"{ticker}_term_structure.png")


# --------------------------- Smile evolution -------------------------------

with tab_evo:
    if not report.fits:
        st.info("No SVI fits available — relax filters or try a more liquid name.")
    else:
        fig = render_smile_evolution(df, report.fits)
        st.plotly_chart(fig, width="stretch", height=620, key="evo_chart")
        st.caption(
            "Every per-expiry SVI smile on one axes. Color goes "
            "front-month (deep purple) → back-month (bright yellow). "
            "Watch how skew flattens with maturity and ATM rises (contango)."
        )
        _snapshot_button(fig, key="snap_evo",
                          default_name=f"{ticker}_smile_evolution.png")


# --------------------------- Skew metrics ---------------------------------

with tab_skew:
    if not report.fits:
        st.info("No fits to compute skew on.")
    else:
        st.markdown(
            "**25-delta risk reversal & butterfly per expiry.**  "
            "RR = IV(25Δ call) − IV(25Δ put). Negative = put skew (equity-style).  "
            "BF = ½·(call+put) − ATM. Positive = wings richer than ATM (smile convexity)."
        )

        skew_df = compute_skew_metrics(
            report.fits, spot=spot,
            risk_free=float(df["r"].dropna().median()) if "r" in df else 0.04,
            dividend_yield=float(get_attr(df, "dividend_yield", 0.0)),
        )
        if skew_df.empty:
            st.info("Couldn't compute skew metrics for any expiry.")
        else:
            view = skew_df.copy()
            view["expiry"] = pd.to_datetime(view["expiry"]).dt.strftime("%Y-%m-%d")
            view = view.rename(columns={
                "expiry": "Expiry", "ttm": "TTM (yrs)",
                "atm_iv": "ATM IV",
                "iv_25d_call": "IV 25Δ call", "iv_25d_put": "IV 25Δ put",
                "iv_10d_call": "IV 10Δ call", "iv_10d_put": "IV 10Δ put",
                "rr_25": "RR 25Δ", "bf_25": "BF 25Δ",
                "rr_10": "RR 10Δ", "bf_10": "BF 10Δ",
            })
            st.dataframe(view, hide_index=True, width="stretch",
                          column_config={
                              "TTM (yrs)":    st.column_config.NumberColumn(format="%.3f"),
                              "ATM IV":       st.column_config.NumberColumn(format="%.4f"),
                              "IV 25Δ call":  st.column_config.NumberColumn(format="%.4f"),
                              "IV 25Δ put":   st.column_config.NumberColumn(format="%.4f"),
                              "IV 10Δ call":  st.column_config.NumberColumn(format="%.4f"),
                              "IV 10Δ put":   st.column_config.NumberColumn(format="%.4f"),
                              "RR 25Δ":       st.column_config.NumberColumn(format="%+.4f"),
                              "BF 25Δ":       st.column_config.NumberColumn(format="%+.4f"),
                              "RR 10Δ":       st.column_config.NumberColumn(format="%+.4f"),
                              "BF 10Δ":       st.column_config.NumberColumn(format="%+.4f"),
                          })


# --------------------------- Strategy P&L ----------------------------------

with tab_strat:
    st.markdown(
        "**Strategy payoff & current MTM.**  Pick a template and the "
        "strikes/expiry from the chain — we'll draw the hockey-stick "
        "payoff and the current mark-to-market as you sweep spot."
    )

    PRESETS = [
        "Long call",
        "Long put",
        "Call spread (bullish)",
        "Put spread (bearish)",
        "Long straddle",
        "Long strangle",
        "Iron condor",
        "Custom (type freely)",
    ]
    preset = st.selectbox("Strategy template", PRESETS, index=2)

    if preset == "Custom (type freely)":
        text = st.text_input(
            "Position",
            value=_suggest_default_strategy(df),
            help="Format: '<side> <qty> <strike><C/P> <expiry>'. "
                 "Comma-separate legs.",
        )
    else:
        text = _build_strategy_from_preset(preset, df)
        st.markdown(
            f"<div style='padding:0.5rem 0.8rem; "
            f"background:#0a0a0f; border:1px solid rgba(255,209,102,0.3); "
            f"border-radius:6px; font-family:monospace; "
            f"color:#ffd166;'>{text}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Switch to **Custom** above to edit the position freely.")

    n_grid = st.slider("Spot grid resolution", 50, 500, 200)

    try:
        strat = parse_strategy(text, name=f"{ticker} strategy")
        pnl   = compute_payoff(strat, df, n_grid=n_grid)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't compute P&L: {exc}")
    else:
        fig = render_payoff(pnl)
        st.plotly_chart(fig, width="stretch", height=560, key="strat_chart")

        legs = pnl.attrs.get("legs_info", [])
        if legs:
            legs_df = pd.DataFrame(legs)
            st.caption("Per-leg context (matched against the chain):")
            st.dataframe(legs_df, hide_index=True, width="stretch")

        cols = st.columns(3)
        cols[0].metric("Current MTM (per share)",
                        f"${pnl.attrs.get('mtm_now', 0.0):+,.4f}")
        cols[1].metric("Spot used", f"${pnl.attrs.get('spot_now'):,.2f}")
        cols[2].metric("# Legs", f"{len(strat.legs)}")
        _snapshot_button(fig, key="snap_strat",
                          default_name=f"{ticker}_strategy.png")


# --------------------------- Realized vs implied --------------------------

with tab_rv:
    rv_window = st.slider("Realized-vol window (trading days)", 5, 90, 30,
                           key="rv_window")
    try:
        rv_cached = get_realized_vol(ticker, rv_window)
    except (RuntimeError, ValueError, KeyError) as exc:
        st.error(f"Couldn't fetch history: {exc}")
    else:
        # Build a *fresh* RealizedVolSeries instead of mutating the
        # cached one — Streamlit's cache hands the same object back
        # on every render, and mutation contaminates future calls
        # with stale atm_iv from the previous tab visit.
        atm_iv = None
        atm_expiry = None
        if report.fits:
            target_T = rv_window / 252.0
            best = min(report.fits.items(),
                        key=lambda kv: abs(kv[1].T - target_T))
            atm_iv = float(best[1].iv(np.array([0.0]))[0])
            atm_expiry = pd.Timestamp(best[0])
        rv = dataclasses.replace(rv_cached, atm_iv=atm_iv, atm_expiry=atm_expiry)

        fig = render_realized_vs_implied(rv)
        st.plotly_chart(fig, width="stretch", height=560, key="rv_chart")
        st.caption(
            f"Rolling {rv_window}-day annualised realized vol from close-to-close "
            "log returns. Gold marker = current ATM IV at the matching expiry. "
            "The chronic gap between IV and RV is the volatility risk premium."
        )
        _snapshot_button(fig, key="snap_rv",
                          default_name=f"{ticker}_rv_iv.png")


# --------------------------- Top deviations -------------------------------

with tab_table:
    # Controls live here, not in the global sidebar — they only affect
    # this tab.
    col_rank, col_top = st.columns([2, 1])
    with col_rank:
        rank_by = st.selectbox(
            "Rank deviations by",
            list(METRIC_LABELS),
            format_func=METRIC_LABELS.__getitem__,
            help="Absolute IV miss is best for content; z-score is "
                 "statistically rigorous; dollar measures PnL impact.",
        )
    with col_top:
        top_n = st.slider("Top N", 5, 50, 15)

    top = report.top(n=top_n, side="both", by=rank_by)
    if top.empty:
        st.info("No contracts surfaced — try a different metric or relax filters.")
    else:
        st.markdown(
            f"**Top {len(top)} contracts ranked by {METRIC_LABELS[rank_by].lower()}.**  "
            "Positive residual = market IV above SVI fit (option rich).  "
            "Negative = market IV below fit (option cheap).  "
            "_Hypothesis generator, not a trade signal._"
        )

        cols = ["type", "strike", "expiry", "ttm", "iv", "svi_iv",
                "residual_iv", "spread_iv", "zscore", "vega", "open_interest"]
        view = top[cols].copy()
        view["expiry"] = pd.to_datetime(view["expiry"]).dt.strftime("%Y-%m-%d")
        view.attrs.clear()
        view = view.rename(columns={
            "type":          "Type",
            "strike":        "Strike",
            "expiry":        "Expiry",
            "ttm":           "TTM (yrs)",
            "iv":            "Market IV",
            "svi_iv":        "SVI fit IV",
            "residual_iv":   "Residual",
            "spread_iv":     "Spread (IV)",
            "zscore":        "Z-score",
            "vega":          "Vega",
            "open_interest": "Open interest",
        })
        st.dataframe(
            view, width="stretch", hide_index=True,
            column_config={
                "Strike":        st.column_config.NumberColumn(format="%.2f"),
                "TTM (yrs)":     st.column_config.NumberColumn(format="%.3f"),
                "Market IV":     st.column_config.NumberColumn(format="%.4f"),
                "SVI fit IV":    st.column_config.NumberColumn(format="%.4f"),
                "Residual":      st.column_config.NumberColumn(format="%+.4f"),
                "Spread (IV)":   st.column_config.NumberColumn(format="%.5f"),
                "Z-score":       st.column_config.NumberColumn(format="%+.2f"),
                "Vega":          st.column_config.NumberColumn(format="%.2f"),
                "Open interest": st.column_config.NumberColumn(format="%d"),
            },
        )

        buf = StringIO()
        view.to_csv(buf, index=False)
        st.download_button(
            "⬇  Download as CSV",
            data=buf.getvalue(),
            file_name=f"{ticker}_mispricing_top{len(top)}.csv",
            mime="text/csv",
        )


# ============================================================================
# Footer
# ============================================================================

st.divider()
st.caption(
    "C++ math kernel via pybind11 (≈3M IV solves/sec) · "
    "Data: yfinance (delayed quotes) · "
    "Risk-free rates: bundled Treasury snapshot or live FRED."
)
