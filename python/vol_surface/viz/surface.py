"""3D implied-volatility surface and 2D smile-slice rendering.

Dark-mode-first aesthetic: black scene background, plasma colorscale
(deep purple base → glowing yellow peaks), wireframe overlay, bright
data points. Designed for short-form social content where the surface
needs to *look* dramatic at a glance.

* **3D surface**: log-moneyness × time-to-expiry × IV. Plasma colormap
  reads "hot/cold" intuitively — high IV regions glow.
* **2D smile slice**: per-expiry IV vs log-moneyness with bid-ask
  shaded band, optional SVI overlay, and a residuals strip below.

OTM-only is the default because ITM contracts are dominated by
intrinsic and have low vega; their quoted IV is noisy. Market
convention is to fit on OTM puts (k<0) + OTM calls (k≥0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lazy imports — keep top-level lightweight for users who don't install
# the [viz] extras.
# ---------------------------------------------------------------------------

def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: WPS433
        from plotly.subplots import make_subplots  # noqa: WPS433
        return go, make_subplots
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "plotly not installed. `pip install vol-surface[viz]`."
        ) from exc

def _require_griddata():
    try:
        from scipy.interpolate import griddata  # noqa: WPS433
        return griddata
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "scipy not installed. `pip install vol-surface[viz]`."
        ) from exc


# Visual constants — tune the dark-quant aesthetic from one place.
_BG_BLACK     = "#000000"
_BG_PANEL     = "#0a0a0f"
_GRID_FAINT   = "rgba(120,120,180,0.15)"
_AXIS_FAINT   = "rgba(180,180,220,0.4)"
_TEXT_BRIGHT  = "rgba(230,230,255,0.95)"
_CALL_COLOR   = "#5cd8ff"   # cyan-ish — pops on dark
_PUT_COLOR    = "#ff5c8a"   # hot pink
_FIT_COLOR    = "#ffd166"   # warm gold
_BAND_FILL    = "rgba(255,209,102,0.12)"
_RESID_POS    = "rgba(255,92,138,0.85)"
_RESID_NEG    = "rgba(92,216,255,0.85)"


# ---------------------------------------------------------------------------
# 3D surface — "dark quant" aesthetic
# ---------------------------------------------------------------------------

def render_surface(
    df: pd.DataFrame,
    *,
    side: Literal["OTM", "calls", "puts", "both"] = "OTM",
    min_iv: float = 0.05,
    max_iv: float = 2.0,
    n_grid: int = 80,
    save_html: str | Path | None = None,
    title: str | None = None,
    show_wireframe: bool = True,
    show_data_points: bool = True,
):
    """Render a 3D implied-volatility surface in dark-quant style.

    Parameters
    ----------
    df
        Enriched chain DataFrame from ``DataFetcher.get(...)``.
    side
        ``"OTM"`` (default), ``"calls"``, ``"puts"``, or ``"both"``.
    min_iv, max_iv
        IV band — drop NaN-resolved-as-tiny and absurd outliers.
    n_grid
        Resolution of the interpolation mesh. Higher = smoother
        surface but a bigger HTML file. 80 is a sweet spot.
    show_wireframe
        Overlay a faint wireframe lattice to give the surface
        readable contour structure even when colour saturates.
    show_data_points
        Plot each contract as a small bright dot. Off for clean
        screenshots, on for analytical reading.
    save_html
        Write a self-contained interactive HTML to this path.
    title
        Optional override; otherwise auto-generated from df.attrs.
    """
    go, _ = _require_plotly()
    griddata = _require_griddata()

    pts = _filter_points(df, side, min_iv, max_iv)
    if len(pts) < 10:
        raise RuntimeError(
            f"Only {len(pts)} usable points after filtering — surface "
            "would be unreliable. Loosen filters or pick a more liquid "
            "underlying.")

    # Outlier suppression: a single mis-priced contract whose IV survived
    # the solver's vega gate can produce a 500 %+ spike that dominates
    # the surface visually. We drop points beyond an IQR-based upper
    # fence on IV, computed *per-expiry* so a legitimately high
    # front-month doesn't get clipped against a low back-month.
    pts = _drop_iv_outliers(pts)

    x = pts["log_moneyness"].to_numpy()
    y = pts["ttm"].to_numpy()
    z = pts["iv"].to_numpy()

    # Cubic interpolation inside the convex hull, nearest-neighbour to
    # fill the corners. Pure cubic leaves NaNs and looks ragged.
    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z, (XI, YI), method="cubic")
    ZI_fill = griddata((x, y), z, (XI, YI), method="nearest")
    ZI = np.where(np.isnan(ZI), ZI_fill, ZI)

    # Final hard cap on the rendered surface — even after outlier
    # removal, interpolation can occasionally bump above realistic
    # levels in sparse-data corners. Cap at the 99th percentile of
    # observed IVs so the colourscale doesn't waste range.
    z_cap = float(np.nanpercentile(z, 99)) if len(z) > 5 else float(np.nanmax(z))
    ZI = np.minimum(ZI, z_cap)

    fig = go.Figure()

    # Main surface — Plasma colorscale: dark purple base → magenta →
    # orange → bright yellow at the peaks. Reads "hot" intuitively.
    fig.add_trace(go.Surface(
        x=xi, y=yi, z=ZI,
        colorscale="Plasma",
        colorbar=dict(
            title=dict(text="IV", font=dict(color=_TEXT_BRIGHT)),
            tickfont=dict(color=_TEXT_BRIGHT),
            outlinewidth=0,
            thickness=14,
            len=0.7,
        ),
        showscale=True,
        opacity=0.96,
        # Flat shading — colour-only, no specular highlights or fresnel
        # reflection. High ambient + tiny diffuse keeps the geometry
        # readable without the glossy "mirror" look.
        lighting=dict(
            ambient=0.95,
            diffuse=0.05,
            specular=0.0,
            roughness=1.0,
            fresnel=0.0,
        ),
        lightposition=dict(x=0, y=0, z=10000),
        contours=dict(
            z=dict(
                show=True,
                start=float(np.nanmin(ZI)),
                end=float(np.nanmax(ZI)),
                size=max((np.nanmax(ZI) - np.nanmin(ZI)) / 18, 0.005),
                color="rgba(255,255,255,0.18)",
                width=1,
                project_z=True,  # also project onto base plane
            ),
        ),
        hovertemplate=(
            "ln(K/F) = <b>%{x:+.3f}</b><br>"
            "T = <b>%{y:.3f}y</b><br>"
            "IV = <b>%{z:.4f}</b><extra></extra>"
        ),
        name="IV surface",
    ))

    # Wireframe lattice — extra readable contour structure.
    if show_wireframe:
        # Sparse lattice for readability; 12 lines each direction.
        step_x = max(n_grid // 12, 1)
        step_y = max(n_grid // 12, 1)
        for i in range(0, n_grid, step_x):
            fig.add_trace(go.Scatter3d(
                x=XI[:, i], y=YI[:, i], z=ZI[:, i],
                mode="lines",
                line=dict(color="rgba(180,200,255,0.20)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))
        for j in range(0, n_grid, step_y):
            fig.add_trace(go.Scatter3d(
                x=XI[j, :], y=YI[j, :], z=ZI[j, :],
                mode="lines",
                line=dict(color="rgba(180,200,255,0.20)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Real contracts as small bright dots — readable as data points
    # without overpowering the surface.
    if show_data_points:
        # Hover annotations: include strike + expiry when present.
        custom = np.column_stack([
            pts["strike"].to_numpy(),
            pd.to_datetime(pts["expiry"]).dt.strftime("%Y-%m-%d").to_numpy(),
            pts["type"].to_numpy(),
        ])
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=2.4,
                color="rgba(255,255,255,0.85)",
                line=dict(width=0),
            ),
            customdata=custom,
            hovertemplate=(
                "%{customdata[2]} K=%{customdata[0]:.0f}  exp %{customdata[1]}<br>"
                "ln(K/F) = %{x:+.3f}<br>"
                "T = %{y:.3f}y<br>"
                "IV = <b>%{z:.4f}</b><extra></extra>"
            ),
            name="Contracts",
        ))

    # Vertical "ATM" plane at k=0 — cyan glow, thin. Visually anchors
    # the surface; eye-tracks where puts → calls.
    if x.min() < 0 < x.max():
        atm_y = np.array([y.min(), y.max()])
        atm_z = np.array([np.nanmin(ZI), np.nanmax(ZI)])
        fig.add_trace(go.Scatter3d(
            x=[0, 0, 0, 0],
            y=[atm_y[0], atm_y[1], atm_y[1], atm_y[0]],
            z=[atm_z[0], atm_z[0], atm_z[1], atm_z[1]],
            mode="lines",
            line=dict(color="rgba(120,200,255,0.45)", width=2),
            showlegend=False,
            hoverinfo="skip",
            name="ATM",
        ))

    auto_title = title or _auto_title(df)
    fig.update_layout(
        title=dict(
            text=auto_title,
            font=dict(color=_TEXT_BRIGHT, size=18),
            x=0.02, y=0.97,
        ),
        scene=dict(
            xaxis=_axis_dark("log moneyness  ln(K / F)"),
            yaxis=_axis_dark("time to expiry (years)"),
            zaxis=_axis_dark("implied volatility"),
            bgcolor=_BG_BLACK,
            camera=dict(eye=dict(x=1.55, y=-1.55, z=0.95)),
            aspectratio=dict(x=1.4, y=1.0, z=0.85),
        ),
        paper_bgcolor=_BG_BLACK,
        plot_bgcolor=_BG_BLACK,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        font=dict(color=_TEXT_BRIGHT),
    )

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)

    return fig


# ---------------------------------------------------------------------------
# 2D smile slice — bid/ask band + SVI overlay + residual strip
# ---------------------------------------------------------------------------

def render_smile(
    df: pd.DataFrame,
    expiry: pd.Timestamp | str,
    *,
    side: Literal["OTM", "calls", "puts", "both"] = "OTM",
    min_iv: float = 0.05,
    max_iv: float = 2.0,
    svi_params=None,
    save_html: str | Path | None = None,
    show_residuals: bool = True,
):
    """Plot IV vs log-moneyness for a single expiry.

    Snaps to the nearest available expiry if the requested date doesn't
    have listed contracts. Pass ``svi_params`` to overlay the fitted
    curve and (when ``show_residuals=True``) a residuals strip below.

    The bid-ask spread is shown as a faint shaded band around each
    point — visually conveys quote tightness, which is the most
    important context for interpreting any IV deviation.
    """
    go, make_subplots = _require_plotly()

    requested = pd.to_datetime(expiry)
    pts = _filter_points(df, side, min_iv, max_iv)

    available = sorted(pts["expiry"].dropna().unique())
    if not available:
        raise RuntimeError("No usable contracts after filtering.")

    if requested not in available:
        closest = min(available, key=lambda e: abs(pd.Timestamp(e) - requested))
        delta_days = abs((pd.Timestamp(closest) - requested).days)
        print(f"  ⚠ No contracts at {requested.date()}; using nearest "
              f"{pd.Timestamp(closest).date()} ({delta_days} days off).")
        requested = pd.Timestamp(closest)
    expiry_ts = requested

    pts = pts[pts["expiry"] == expiry_ts].sort_values("log_moneyness")
    if pts.empty:
        raise RuntimeError(f"No points for expiry {expiry_ts.date()}")

    # Decide whether we have a residuals strip below the main plot.
    has_resid = show_residuals and svi_params is not None
    if has_resid:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.78, 0.22],
            vertical_spacing=0.04,
            shared_xaxes=True,
            subplot_titles=("", "residual: market IV − SVI fit"),
        )
        main_row, resid_row = 1, 2
    else:
        fig = go.Figure()
        main_row = resid_row = None

    # --- Bid/ask shaded band (only if columns are present) ---
    if {"bid", "ask"}.issubset(pts.columns):
        T = float(pts["ttm"].iloc[0])
        S = df.attrs.get("spot", 1.0)
        # Approximate bid/ask IV by inverting the price spread through
        # local vega. For visual purposes this is fine — we just want
        # to show how tight the quote is at each strike.
        with np.errstate(divide="ignore", invalid="ignore"):
            half_spread_iv = (pts["ask"] - pts["bid"]) / (2.0 * np.maximum(
                _approx_vega(pts, S), 1e-3))
        half_spread_iv = half_spread_iv.fillna(0).clip(0, 0.5)
        upper = pts["iv"] + half_spread_iv
        lower = pts["iv"] - half_spread_iv

        kwargs = dict(row=main_row, col=1) if has_resid else {}
        fig.add_trace(go.Scatter(
            x=pts["log_moneyness"], y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), **kwargs)
        fig.add_trace(go.Scatter(
            x=pts["log_moneyness"], y=lower,
            mode="lines", fill="tonexty",
            fillcolor=_BAND_FILL,
            line=dict(width=0),
            name="bid-ask",
            hovertemplate="bid-ask half-spread=%{customdata:.4f}<extra></extra>",
            customdata=half_spread_iv,
        ), **kwargs)

    # --- Calls / Puts as glowing markers ---
    for typ, color, label in [("C", _CALL_COLOR, "Calls"),
                              ("P", _PUT_COLOR,  "Puts")]:
        sub = pts[pts["type"] == typ]
        if sub.empty:
            continue
        custom = np.column_stack([
            sub["strike"].to_numpy(),
            sub["open_interest"].fillna(0).astype(int).to_numpy(),
        ])
        kwargs = dict(row=main_row, col=1) if has_resid else {}
        fig.add_trace(go.Scatter(
            x=sub["log_moneyness"], y=sub["iv"],
            mode="markers",
            name=label,
            marker=dict(
                size=8, color=color,
                line=dict(width=0.5, color="rgba(255,255,255,0.6)"),
                opacity=0.95,
            ),
            customdata=custom,
            hovertemplate=(
                "<b>" + label[0] + " K=%{customdata[0]:.2f}</b><br>"
                "ln(K/F) = %{x:+.3f}<br>"
                "IV = %{y:.4f}<br>"
                "OI = %{customdata[1]}<extra></extra>"
            ),
        ), **kwargs)

    # --- SVI overlay ---
    if svi_params is not None:
        k_grid = np.linspace(pts["log_moneyness"].min(),
                             pts["log_moneyness"].max(), 240)
        kwargs = dict(row=main_row, col=1) if has_resid else {}
        fig.add_trace(go.Scatter(
            x=k_grid, y=svi_params.iv(k_grid),
            mode="lines",
            name=f"SVI fit  (n={svi_params.n_points})",
            line=dict(color=_FIT_COLOR, width=2.5),
            hovertemplate="SVI<br>ln(K/F) = %{x:+.3f}<br>IV = %{y:.4f}<extra></extra>",
        ), **kwargs)

    # --- ATM vertical guide (k = 0) ---
    if pts["log_moneyness"].min() < 0 < pts["log_moneyness"].max():
        kwargs = dict(row=main_row, col=1) if has_resid else {}
        fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.18)",
                                      width=1, dash="dot"), **kwargs)

    # --- Residuals strip ---
    if has_resid:
        resid = pts["iv"].to_numpy() - svi_params.iv(pts["log_moneyness"].to_numpy())
        bar_colors = [_RESID_POS if r > 0 else _RESID_NEG for r in resid]
        fig.add_trace(go.Bar(
            x=pts["log_moneyness"], y=resid,
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="ln(K/F) = %{x:+.3f}<br>resid = %{y:+.4f}<extra></extra>",
        ), row=resid_row, col=1)
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1),
                      row=resid_row, col=1)

    ticker = df.attrs.get("ticker", "")
    fig.update_layout(
        title=dict(
            text=f"{ticker} smile · expiry {expiry_ts.date()}",
            font=dict(color=_TEXT_BRIGHT, size=18), x=0.02, y=0.97,
        ),
        paper_bgcolor=_BG_BLACK,
        plot_bgcolor=_BG_PANEL,
        font=dict(color=_TEXT_BRIGHT, size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=50, b=40),
    )

    # Axis styling — applies to both figure types.
    if has_resid:
        fig.update_xaxes(_axis2d(), row=resid_row, col=1,
                         title_text="log moneyness  ln(K / F)")
        fig.update_yaxes(_axis2d(), row=resid_row, col=1,
                         title_text="residual")
        fig.update_xaxes(_axis2d(), row=main_row, col=1)
        fig.update_yaxes(_axis2d(), row=main_row, col=1,
                         title_text="implied volatility")
    else:
        fig.update_xaxes(title_text="log moneyness  ln(K / F)", **_axis2d())
        fig.update_yaxes(title_text="implied volatility", **_axis2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_points(df, side, min_iv, max_iv):
    pts = df.dropna(subset=["iv", "log_moneyness", "ttm"]).copy()
    pts = pts[(pts["iv"] >= min_iv) & (pts["iv"] <= max_iv)]
    if side == "calls":
        pts = pts[pts["type"] == "C"]
    elif side == "puts":
        pts = pts[pts["type"] == "P"]
    elif side == "OTM":
        otm_calls = (pts["type"] == "C") & (pts["log_moneyness"] >= 0)
        otm_puts  = (pts["type"] == "P") & (pts["log_moneyness"] <  0)
        pts = pts[otm_calls | otm_puts]
    elif side == "both":
        pass
    else:  # pragma: no cover
        raise ValueError(f"Unknown side: {side!r}")
    return pts


def _drop_iv_outliers(pts: pd.DataFrame, *, k: float = 3.0) -> pd.DataFrame:
    """Remove points whose IV exceeds the per-expiry IQR upper fence.

    For each ``ttm`` group we compute the 25th and 75th percentile of
    IV and keep points within ``q75 + k·IQR``. ``k=3`` is the
    "extreme outlier" fence in Tukey's classification — knocks out
    mispriced strikes (often deep wings) without touching the
    legitimate body of the smile.
    """
    if len(pts) < 8:
        return pts
    keep = pd.Series(True, index=pts.index)
    for _, idx in pts.groupby("ttm").groups.items():
        sub = pts.loc[idx, "iv"]
        if len(sub) < 4:
            continue
        q25, q75 = sub.quantile([0.25, 0.75])
        iqr = q75 - q25
        upper = q75 + k * iqr
        keep.loc[idx] &= (sub <= upper)
    return pts[keep]


def _approx_vega(pts: pd.DataFrame, spot: float) -> np.ndarray:
    """Quick vega approximation for the bid-ask band (we don't pay a
    full Greeks call here). Uses BS vega = S·pdf(d1)·√T with σ_iv
    plugged in. Order of magnitude is what matters for the band."""
    F = float(spot)
    K = pts["strike"].to_numpy()
    T = np.maximum(pts["ttm"].to_numpy(), 1e-9)
    sig = np.maximum(pts["iv"].to_numpy(), 1e-3)
    with np.errstate(invalid="ignore", divide="ignore"):
        d1 = (np.log(F / K) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
        return F * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * np.sqrt(T)


def _axis_dark(title: str) -> dict:
    """3D scene axis styling — faint grid, bright text."""
    return dict(
        title=dict(text=title, font=dict(color=_TEXT_BRIGHT, size=12)),
        gridcolor=_GRID_FAINT,
        zerolinecolor=_AXIS_FAINT,
        showbackground=True,
        backgroundcolor=_BG_PANEL,
        tickfont=dict(color=_TEXT_BRIGHT, size=10),
    )


def _axis2d() -> dict:
    """2D axis styling — minimal grid, bright text."""
    return dict(
        gridcolor=_GRID_FAINT,
        zerolinecolor=_AXIS_FAINT,
        linecolor=_AXIS_FAINT,
        tickfont=dict(color=_TEXT_BRIGHT, size=11),
        title_font=dict(color=_TEXT_BRIGHT, size=12),
    )


def _auto_title(df: pd.DataFrame) -> str:
    ticker = df.attrs.get("ticker", "")
    spot   = df.attrs.get("spot")
    fetched = df.attrs.get("fetched_at")
    bits = [f"{ticker} implied volatility surface"]
    if spot is not None:
        bits.append(f"spot ${spot:,.2f}")
    if fetched is not None:
        when = fetched if isinstance(fetched, str) else fetched.strftime("%Y-%m-%d %H:%M UTC")
        bits.append(when)
    return "  ·  ".join(bits)
