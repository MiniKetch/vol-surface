"""3D implied-volatility surface and 2D smile-slice rendering.

Dark-mode-first aesthetic: black scene background, plasma colorscale
(deep purple base → glowing yellow peaks), wireframe overlay, bright
data points. Designed for short-form social content where the surface
needs to *look* dramatic at a glance.

* **3D surface**: log-moneyness × time-to-expiry × IV.
* **2D smile slice**: per-expiry IV vs log-moneyness with bid-ask
  shaded band, optional SVI overlay, residuals strip below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from vol_surface.utils import (
    Side, filter_by_side, forward, get_attr, iv_spread_from_price,
)
from vol_surface.viz._style import (
    AXIS_FAINT, BAND_FILL, BG_BLACK, BG_PANEL,
    CALL_COLOR, FIT_COLOR, GRID_FAINT, PUT_COLOR,
    RESID_NEG, RESID_POS, TEXT_BRIGHT,
    axis_2d, axis_3d, require_griddata, require_plotly,
)


# ---------------------------------------------------------------------------
# 3D surface
# ---------------------------------------------------------------------------

def render_surface(
    df: pd.DataFrame,
    *,
    side: Side = "OTM",
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
        Resolution of the interpolation mesh.
    show_wireframe
        Overlay a faint lattice for readable contour structure.
    show_data_points
        Plot each contract as a small bright dot.
    save_html
        Write a self-contained interactive HTML to this path.
    title
        Override; otherwise auto-generated from df.attrs.
    """
    go, _ = require_plotly()
    griddata = require_griddata()

    pts = _filter_points(df, side, min_iv, max_iv)
    if len(pts) < 10:
        raise RuntimeError(
            f"Only {len(pts)} usable points after filtering — surface "
            "would be unreliable. Loosen filters or pick a more liquid "
            "underlying.")

    pts = _drop_iv_outliers(pts)

    x = pts["log_moneyness"].to_numpy()
    y = pts["ttm"].to_numpy()
    z = pts["iv"].to_numpy()

    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z, (XI, YI), method="cubic")
    ZI_fill = griddata((x, y), z, (XI, YI), method="nearest")
    ZI = np.where(np.isnan(ZI), ZI_fill, ZI)

    # Final hard cap on the rendered surface — even after outlier
    # removal, interpolation can occasionally bump above realistic
    # levels in sparse-data corners.
    z_cap = float(np.nanpercentile(z, 99)) if len(z) > 5 else float(np.nanmax(z))
    ZI = np.minimum(ZI, z_cap)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=xi, y=yi, z=ZI,
        colorscale="Plasma",
        colorbar=dict(
            title=dict(text="IV", font=dict(color=TEXT_BRIGHT)),
            tickfont=dict(color=TEXT_BRIGHT),
            outlinewidth=0, thickness=14, len=0.7,
        ),
        showscale=True,
        opacity=0.96,
        # Flat shading — colour-only, no specular highlights.
        lighting=dict(ambient=0.95, diffuse=0.05, specular=0.0,
                      roughness=1.0, fresnel=0.0),
        lightposition=dict(x=0, y=0, z=10000),
        contours=dict(z=dict(
            show=True,
            start=float(np.nanmin(ZI)),
            end=float(np.nanmax(ZI)),
            size=max((np.nanmax(ZI) - np.nanmin(ZI)) / 18, 0.005),
            color="rgba(255,255,255,0.18)",
            width=1, project_z=True,
        )),
        hovertemplate=(
            "ln(K/F) = <b>%{x:+.3f}</b><br>"
            "T = <b>%{y:.3f}y</b><br>"
            "IV = <b>%{z:.4f}</b><extra></extra>"
        ),
        name="IV surface",
    ))

    if show_wireframe:
        step_x = max(n_grid // 12, 1)
        step_y = max(n_grid // 12, 1)
        for i in range(0, n_grid, step_x):
            fig.add_trace(go.Scatter3d(
                x=XI[:, i], y=YI[:, i], z=ZI[:, i],
                mode="lines",
                line=dict(color="rgba(180,200,255,0.20)", width=1),
                showlegend=False, hoverinfo="skip",
            ))
        for j in range(0, n_grid, step_y):
            fig.add_trace(go.Scatter3d(
                x=XI[j, :], y=YI[j, :], z=ZI[j, :],
                mode="lines",
                line=dict(color="rgba(180,200,255,0.20)", width=1),
                showlegend=False, hoverinfo="skip",
            ))

    if show_data_points:
        custom = np.column_stack([
            pts["strike"].to_numpy(),
            pd.to_datetime(pts["expiry"]).dt.strftime("%Y-%m-%d").to_numpy(),
            pts["type"].to_numpy(),
        ])
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=2.4, color="rgba(255,255,255,0.85)",
                        line=dict(width=0)),
            customdata=custom,
            hovertemplate=(
                "%{customdata[2]} K=%{customdata[0]:.0f}  exp %{customdata[1]}<br>"
                "ln(K/F) = %{x:+.3f}<br>T = %{y:.3f}y<br>"
                "IV = <b>%{z:.4f}</b><extra></extra>"
            ),
            name="Contracts",
        ))

    # ATM vertical guide.
    if x.min() < 0 < x.max():
        atm_y = np.array([y.min(), y.max()])
        atm_z = np.array([np.nanmin(ZI), np.nanmax(ZI)])
        fig.add_trace(go.Scatter3d(
            x=[0, 0, 0, 0],
            y=[atm_y[0], atm_y[1], atm_y[1], atm_y[0]],
            z=[atm_z[0], atm_z[0], atm_z[1], atm_z[1]],
            mode="lines",
            line=dict(color="rgba(120,200,255,0.45)", width=2),
            showlegend=False, hoverinfo="skip",
            name="ATM",
        ))

    auto_title = title or _auto_title(df)
    fig.update_layout(
        title=dict(text=auto_title, font=dict(color=TEXT_BRIGHT, size=18),
                   x=0.02, y=0.97),
        scene=dict(
            xaxis=axis_3d("log moneyness  ln(K / F)"),
            yaxis=axis_3d("time to expiry (years)"),
            zaxis=axis_3d("implied volatility"),
            bgcolor=BG_BLACK,
            camera=dict(eye=dict(x=1.55, y=-1.55, z=0.95)),
            aspectratio=dict(x=1.4, y=1.0, z=0.85),
        ),
        paper_bgcolor=BG_BLACK, plot_bgcolor=BG_BLACK,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        font=dict(color=TEXT_BRIGHT),
    )

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)

    return fig


# ---------------------------------------------------------------------------
# 2D smile slice
# ---------------------------------------------------------------------------

def render_smile(
    df: pd.DataFrame,
    expiry: pd.Timestamp | str,
    *,
    side: Side = "OTM",
    min_iv: float = 0.05,
    max_iv: float = 2.0,
    svi_params=None,
    save_html: str | Path | None = None,
    show_residuals: bool = True,
):
    """Plot IV vs log-moneyness for a single expiry.

    Snaps to nearest available expiry if requested date isn't listed.
    Pass ``svi_params`` to overlay the fitted SVI curve and (when
    ``show_residuals=True``) a residuals strip below.
    """
    go, make_subplots = require_plotly()

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

    # Bid/ask shaded band — uses correct r-q-aware vega via shared util.
    if {"bid", "ask"}.issubset(pts.columns):
        S = float(get_attr(df, "spot", 1.0))
        vega_local = _approx_vega(pts, S)
        half_spread_iv = iv_spread_from_price(
            (pts["ask"] - pts["bid"]).to_numpy(), vega_local,
        )
        # Convert spread from "per 1.0 vol" units to a band in IV space.
        # iv_spread_from_price already returns half-spread in σ units.
        half_spread_iv = np.where(np.isfinite(half_spread_iv),
                                   half_spread_iv, 0.0)
        # Cap to keep the visual readable when one bad quote has a 5×IV spread.
        half_spread_iv = np.clip(half_spread_iv, 0.0, 0.5)
        upper = pts["iv"].to_numpy() + half_spread_iv
        lower = pts["iv"].to_numpy() - half_spread_iv

        kwargs = dict(row=main_row, col=1) if has_resid else {}
        fig.add_trace(go.Scatter(
            x=pts["log_moneyness"], y=upper,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), **kwargs)
        fig.add_trace(go.Scatter(
            x=pts["log_moneyness"], y=lower,
            mode="lines", fill="tonexty",
            fillcolor=BAND_FILL,
            line=dict(width=0),
            name="bid-ask",
            hovertemplate="bid-ask half-spread=%{customdata:.4f}<extra></extra>",
            customdata=half_spread_iv,
        ), **kwargs)

    for typ, color, label in [("C", CALL_COLOR, "Calls"),
                              ("P", PUT_COLOR,  "Puts")]:
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
            mode="markers", name=label,
            marker=dict(size=8, color=color,
                        line=dict(width=0.5, color="rgba(255,255,255,0.6)"),
                        opacity=0.95),
            customdata=custom,
            hovertemplate=(
                "<b>" + label[0] + " K=%{customdata[0]:.2f}</b><br>"
                "ln(K/F) = %{x:+.3f}<br>"
                "IV = %{y:.4f}<br>"
                "OI = %{customdata[1]}<extra></extra>"
            ),
        ), **kwargs)

    if svi_params is not None:
        k_grid = np.linspace(pts["log_moneyness"].min(),
                             pts["log_moneyness"].max(), 240)
        kwargs = dict(row=main_row, col=1) if has_resid else {}
        fig.add_trace(go.Scatter(
            x=k_grid, y=svi_params.iv(k_grid),
            mode="lines", name=f"SVI fit  (n={svi_params.n_points})",
            line=dict(color=FIT_COLOR, width=2.5),
            hovertemplate="SVI<br>ln(K/F) = %{x:+.3f}<br>IV = %{y:.4f}<extra></extra>",
        ), **kwargs)

    if pts["log_moneyness"].min() < 0 < pts["log_moneyness"].max():
        kwargs = dict(row=main_row, col=1) if has_resid else {}
        fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.18)",
                                      width=1, dash="dot"), **kwargs)

    if has_resid:
        resid = pts["iv"].to_numpy() - svi_params.iv(pts["log_moneyness"].to_numpy())
        bar_colors = [RESID_POS if r > 0 else RESID_NEG for r in resid]
        fig.add_trace(go.Bar(
            x=pts["log_moneyness"], y=resid,
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="ln(K/F) = %{x:+.3f}<br>resid = %{y:+.4f}<extra></extra>",
        ), row=resid_row, col=1)
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1),
                      row=resid_row, col=1)

    ticker = get_attr(df, "ticker", "")
    fig.update_layout(
        title=dict(text=f"{ticker} smile · expiry {expiry_ts.date()}",
                   font=dict(color=TEXT_BRIGHT, size=18), x=0.02, y=0.97),
        paper_bgcolor=BG_BLACK,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_BRIGHT, size=12),
        legend=dict(bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(255,255,255,0.15)",
                    borderwidth=1),
        margin=dict(l=10, r=10, t=50, b=40),
    )

    if has_resid:
        fig.update_xaxes(axis_2d(), row=resid_row, col=1,
                         title_text="log moneyness  ln(K / F)")
        fig.update_yaxes(axis_2d(), row=resid_row, col=1,
                         title_text="residual")
        fig.update_xaxes(axis_2d(), row=main_row, col=1)
        fig.update_yaxes(axis_2d(), row=main_row, col=1,
                         title_text="implied volatility")
    else:
        fig.update_xaxes(title_text="log moneyness  ln(K / F)", **axis_2d())
        fig.update_yaxes(title_text="implied volatility", **axis_2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_points(df, side, min_iv, max_iv):
    """Drop NaN IVs / IV out of band, then apply side filter.

    Thin wrapper around `utils.filter_by_side` so the IV-band logic
    has a single home and the side selection has a single home.
    """
    pts = df.dropna(subset=["iv", "log_moneyness", "ttm"]).copy()
    pts = pts[(pts["iv"] >= min_iv) & (pts["iv"] <= max_iv)]
    return filter_by_side(pts, side)


def _drop_iv_outliers(pts: pd.DataFrame, *, k: float = 3.0) -> pd.DataFrame:
    """Remove points whose IV exceeds the per-expiry IQR upper fence."""
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
    """Quick vega approximation for the bid-ask band.

    Uses the *correct* forward F = S·exp((r-q)·T) rather than
    F = S, and a d1 that includes the (r-q) drift. The earlier
    approximation in this file silently dropped the cost-of-carry
    term, which biased d1 by ~0.1 for typical SPY inputs.
    """
    K = pts["strike"].to_numpy()
    T = np.maximum(pts["ttm"].to_numpy(), 1e-9)
    sig = np.maximum(pts["iv"].to_numpy(), 1e-3)
    r = pts["r"].to_numpy() if "r" in pts.columns else np.zeros_like(K)
    q = pts["q"].to_numpy() if "q" in pts.columns else np.zeros_like(K)
    F = forward(spot, r, q, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        d1 = (np.log(F / K) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
        # vega = S·exp(-q·T)·pdf(d1)·√T (not F-anchored)
        return (
            spot * np.exp(-q * T)
            * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
            * np.sqrt(T)
        )


def _auto_title(df: pd.DataFrame) -> str:
    ticker = get_attr(df, "ticker", "")
    spot   = get_attr(df, "spot")
    fetched = get_attr(df, "fetched_at")
    bits = [f"{ticker} implied volatility surface"]
    if spot is not None:
        bits.append(f"spot ${spot:,.2f}")
    if fetched is not None:
        when = fetched if isinstance(fetched, str) else fetched.strftime("%Y-%m-%d %H:%M UTC")
        bits.append(when)
    return "  ·  ".join(bits)


# ---------------------------------------------------------------------------
# Backwards-compat re-exports — older code in this package imports
# the leading-underscore names from this module. We keep them as
# aliases so the refactor is non-breaking.
# ---------------------------------------------------------------------------

_BG_BLACK    = BG_BLACK
_BG_PANEL    = BG_PANEL
_TEXT_BRIGHT = TEXT_BRIGHT
_CALL_COLOR  = CALL_COLOR
_PUT_COLOR   = PUT_COLOR
_FIT_COLOR   = FIT_COLOR
_GRID_FAINT  = GRID_FAINT
_AXIS_FAINT  = AXIS_FAINT
_BAND_FILL   = BAND_FILL
_RESID_POS   = RESID_POS
_RESID_NEG   = RESID_NEG
_axis2d      = axis_2d
_axis_dark   = axis_3d
_require_plotly  = require_plotly
_require_griddata = require_griddata
