"""Realized-vs-implied volatility panel.

The "volatility risk premium" — the chronic gap between IV (forward-
looking, what the market is pricing) and RV (backward-looking, what
the underlying actually did) — is one of the cleanest empirical
regularities in equity options. IV typically sits above RV most of
the time, and short-vol strategies (selling premium) capture that
spread. The panel makes the gap visually obvious.

We compute close-to-close realized vol over a rolling window:
    RV_t = √(252) · stdev_t(log(S_t / S_{t−1}))

Default window is 30 trading days, which roughly matches the
front-month expiry the IV is being read from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RealizedVolSeries:
    """Daily realized vol time series with a single ATM-IV point."""
    dates: pd.DatetimeIndex
    realized: np.ndarray             # rolling annualised stdev
    spot: np.ndarray                 # close prices
    atm_iv: Optional[float]          # current ATM IV at the expiry below
    atm_expiry: Optional[pd.Timestamp]
    window_days: int


def compute_realized_vol(
    ticker: str,
    *,
    period: str = "1y",
    window_days: int = 30,
) -> RealizedVolSeries:
    """Pull historical closes, compute rolling realized vol.

    Parameters
    ----------
    ticker
        Underlying symbol (passed straight to yfinance).
    period
        How much history to fetch. yfinance period strings:
        '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'.
    window_days
        Rolling window size in trading days. 21 ≈ 1 month, 30 ≈ 6 weeks.
    """
    try:
        import yfinance as yf  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError(
            "yfinance not installed. `pip install vol-surface[data]`."
        ) from exc

    tk = yf.Ticker(ticker)
    hist = tk.history(period=period, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker!r}")

    close = hist["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(window=window_days).std() * np.sqrt(252)

    return RealizedVolSeries(
        dates=close.index,
        realized=rv.to_numpy(),
        spot=close.to_numpy(),
        atm_iv=None,
        atm_expiry=None,
        window_days=window_days,
    )


def render_realized_vs_implied(
    rv: RealizedVolSeries,
    *,
    save_html: str | Path | None = None,
):
    """Plot rolling realized vol with the current ATM IV as a marker."""
    from vol_surface.viz.surface import (
        _BG_BLACK, _BG_PANEL, _TEXT_BRIGHT,
        _CALL_COLOR, _FIT_COLOR, _PUT_COLOR, _axis2d, _require_plotly,
    )
    go, _ = _require_plotly()

    fig = go.Figure()

    # Realized vol time series — the main story.
    fig.add_trace(go.Scatter(
        x=rv.dates, y=rv.realized,
        mode="lines",
        name=f"Realized vol  ({rv.window_days}-day rolling)",
        line=dict(color=_CALL_COLOR, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>RV = %{y:.4f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(92,216,255,0.06)",
    ))

    # ATM IV marker — single point at the right edge if known.
    if rv.atm_iv is not None and rv.atm_expiry is not None:
        x_iv = pd.Timestamp(rv.atm_expiry)
        fig.add_trace(go.Scatter(
            x=[x_iv], y=[rv.atm_iv],
            mode="markers",
            name=f"Implied (ATM @ {x_iv.date()})",
            marker=dict(size=14, color=_FIT_COLOR, symbol="diamond",
                        line=dict(width=1.5, color="rgba(0,0,0,0.6)")),
            hovertemplate=f"<b>{x_iv.date()}</b><br>"
                          f"ATM IV = {rv.atm_iv:.4f}<extra></extra>",
        ))
        # Horizontal reference line at the current IV.
        fig.add_hline(
            y=rv.atm_iv,
            line=dict(color=_FIT_COLOR, width=1, dash="dot"),
            annotation=dict(
                text=f"current ATM IV {rv.atm_iv:.3f}",
                font=dict(color=_FIT_COLOR, size=11),
                xanchor="left", yanchor="bottom",
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor=_FIT_COLOR,
            ),
        )

    fig.update_layout(
        title=dict(
            text=f"Realized vol · {rv.window_days}-day rolling annualised",
            font=dict(color=_TEXT_BRIGHT, size=17), x=0.02, y=0.97,
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
    fig.update_xaxes(title_text="date", **_axis2d())
    fig.update_yaxes(title_text="annualised volatility", **_axis2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig
