"""Shared Plotly style constants and axis helpers for every viz module.

Lives in `viz/` (private, leading-underscore filename) so it's clearly
viz-internal but importable across viz/, analytics/, and strategy/ —
all of which need the same dark-quant palette and axis styling.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Palette — tune the dark-quant aesthetic from one place.
# ---------------------------------------------------------------------------

BG_BLACK     = "#000000"
BG_PANEL     = "#0a0a0f"
GRID_FAINT   = "rgba(120,120,180,0.15)"
AXIS_FAINT   = "rgba(180,180,220,0.4)"
TEXT_BRIGHT  = "rgba(230,230,255,0.95)"

# Trace colours
CALL_COLOR   = "#5cd8ff"   # cyan — calls / RV line
PUT_COLOR    = "#ff5c8a"   # hot pink — puts / payoff
FIT_COLOR    = "#ffd166"   # warm gold — SVI fit / current MTM
BAND_FILL    = "rgba(255,209,102,0.12)"
RESID_POS    = "rgba(255,92,138,0.85)"
RESID_NEG    = "rgba(92,216,255,0.85)"


# ---------------------------------------------------------------------------
# Lazy plotly import — keep top-level cheap for users who don't have
# plotly installed.
# ---------------------------------------------------------------------------

def require_plotly():
    try:
        import plotly.graph_objects as go        # noqa: WPS433
        from plotly.subplots import make_subplots  # noqa: WPS433
        return go, make_subplots
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "plotly not installed. `pip install vol-surface[viz]`."
        ) from exc


def require_griddata():
    try:
        from scipy.interpolate import griddata  # noqa: WPS433
        return griddata
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "scipy not installed. `pip install vol-surface[viz]`."
        ) from exc


# ---------------------------------------------------------------------------
# Axis dictionaries — one place to control every chart's gridlines.
# ---------------------------------------------------------------------------

def axis_3d(title: str) -> dict:
    """3D scene axis styling — faint grid on the panel, bright text."""
    return dict(
        title=dict(text=title, font=dict(color=TEXT_BRIGHT, size=12)),
        gridcolor=GRID_FAINT,
        zerolinecolor=AXIS_FAINT,
        showbackground=True,
        backgroundcolor=BG_PANEL,
        tickfont=dict(color=TEXT_BRIGHT, size=10),
    )


def axis_2d() -> dict:
    """2D axis styling — minimal grid, bright text."""
    return dict(
        gridcolor=GRID_FAINT,
        zerolinecolor=AXIS_FAINT,
        linecolor=AXIS_FAINT,
        tickfont=dict(color=TEXT_BRIGHT, size=11),
        title_font=dict(color=TEXT_BRIGHT, size=12),
    )
