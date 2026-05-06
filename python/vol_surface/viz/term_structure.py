"""Term-structure curve: ATM IV vs maturity.

The single most-recognised plot in equity-options trading. A flat
curve = market expects the same vol over all horizons. An upward
slope = contango (longer-dated IV richer than short — typical regime,
"vol of vol" priced in). A kink at a specific date = the market is
pricing event vol there (earnings, FOMC, ex-dividend, election).

We pull ATM IV two ways and overlay them:
  * **From SVI fits** — the smooth model says "if you remove smile
    distortion, this is what ATM is worth." Best for the *shape* of
    the term structure.
  * **From raw market quotes** — interpolate IV at the strike closest
    to the forward, per expiry. Shows where the market actually trades.

The gap between the two surfaces tells you whether SVI is missing
something (typically near events).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vol_surface.viz.surface import (
    _BG_BLACK, _BG_PANEL, _TEXT_BRIGHT,
    _CALL_COLOR, _FIT_COLOR,
    _axis2d, _require_plotly,
)


def render_term_structure(
    df: pd.DataFrame,
    *,
    fits: dict | None = None,
    save_html: str | Path | None = None,
):
    """Render an ATM-IV-by-maturity term-structure plot.

    Parameters
    ----------
    df
        Enriched chain DataFrame (must have ``log_moneyness``, ``ttm``,
        ``iv``, ``expiry`` columns).
    fits
        Optional dict of ``{Timestamp: RawSVIParams}``. If provided,
        the SVI ATM curve (``svi_iv(k=0)``) is overlaid as a smooth
        line; otherwise we just show the raw-market curve.
    save_html
        If given, write a self-contained interactive HTML there.
    """
    go, _ = _require_plotly()

    # --- Raw market: ATM IV per expiry from data alone ---
    market_pts = []
    for expiry, sub in df.dropna(subset=["iv", "log_moneyness"]).groupby("expiry"):
        # IV at the contract whose log-moneyness is closest to zero.
        idx = sub["log_moneyness"].abs().idxmin()
        market_pts.append({
            "expiry": pd.Timestamp(expiry),
            "ttm":    float(sub.loc[idx, "ttm"]),
            "iv":     float(sub.loc[idx, "iv"]),
            "k_used": float(sub.loc[idx, "log_moneyness"]),
        })
    market_pts = pd.DataFrame(market_pts).sort_values("ttm")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=market_pts["ttm"],
        y=market_pts["iv"],
        mode="markers+lines",
        name="Market ATM",
        marker=dict(size=10, color=_CALL_COLOR,
                    line=dict(width=1, color="rgba(255,255,255,0.7)")),
        line=dict(color=_CALL_COLOR, width=1.5),
        customdata=np.column_stack([
            market_pts["expiry"].dt.strftime("%Y-%m-%d"),
            market_pts["k_used"],
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "T = %{x:.3f}y<br>"
            "ATM IV = %{y:.4f}<br>"
            "k_used = %{customdata[1]:+.3f}<extra></extra>"
        ),
    ))

    # --- SVI ATM: smooth model curve ---
    if fits:
        rows = []
        for expiry_ts, params in fits.items():
            atm_iv = float(params.iv(np.array([0.0]))[0])
            rows.append({"ttm": params.T, "iv": atm_iv,
                         "expiry": pd.Timestamp(expiry_ts)})
        svi_pts = pd.DataFrame(rows).sort_values("ttm")
        if not svi_pts.empty:
            fig.add_trace(go.Scatter(
                x=svi_pts["ttm"], y=svi_pts["iv"],
                mode="markers+lines",
                name="SVI ATM",
                marker=dict(size=10, color=_FIT_COLOR, symbol="diamond",
                            line=dict(width=1, color="rgba(0,0,0,0.6)")),
                line=dict(color=_FIT_COLOR, width=2),
                customdata=svi_pts["expiry"].dt.strftime("%Y-%m-%d"),
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "T = %{x:.3f}y<br>"
                    "SVI ATM IV = %{y:.4f}<extra></extra>"
                ),
            ))

    ticker = df.attrs.get("ticker", "")
    fig.update_layout(
        title=dict(
            text=f"{ticker} term structure  ·  ATM IV vs maturity",
            font=dict(color=_TEXT_BRIGHT, size=18), x=0.02, y=0.97,
        ),
        paper_bgcolor=_BG_BLACK,
        plot_bgcolor=_BG_PANEL,
        font=dict(color=_TEXT_BRIGHT),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    fig.update_xaxes(title_text="time to expiry (years)", **_axis2d())
    fig.update_yaxes(title_text="implied volatility (ATM)", **_axis2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig
