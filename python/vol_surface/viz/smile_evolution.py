"""All smiles on one axes, color-graded by maturity.

A "small multiples done as one big multiple" view. Every expiry
overlaid as its own SVI curve, line color stepping through the plasma
ramp from front-month (purple) to back-month (yellow). The story it
tells in one glance:
* skew typically *flattens* with maturity (back-month curves are less
  steep than front-month) — visible as the yellow lines being shallower
  than the purple ones;
* ATM IV typically *rises* with maturity (contango) — yellow lines sit
  above purple lines at k=0;
* events show up as a kink in one specific colour line.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vol_surface.viz.surface import (
    _BG_BLACK, _BG_PANEL, _TEXT_BRIGHT,
    _axis2d, _require_plotly,
)


def render_smile_evolution(
    df: pd.DataFrame,
    fits: dict,
    *,
    k_min: float = -0.4,
    k_max: float = 0.4,
    n_points: int = 200,
    save_html: str | Path | None = None,
):
    """Overlay all per-expiry SVI smile curves on one 2D plot.

    Parameters
    ----------
    df
        Enriched chain DataFrame (only used for the title and ticker).
    fits
        Dict of ``{Timestamp: RawSVIParams}`` — output of the
        mispricing scanner's ``.fits`` attribute.
    k_min, k_max
        Plotting range in log-moneyness. ±0.4 covers the actively-
        traded smile while excluding deep-wing extrapolation.
    n_points
        Resolution of the curves.
    """
    go, _ = _require_plotly()

    if not fits:
        raise RuntimeError(
            "smile_evolution needs a `fits` dict (from "
            "MispricingReport.fits). Run scan_mispricing first.")

    # Sort by maturity so the colour ramp tracks time.
    items = sorted(fits.items(), key=lambda kv: kv[1].T)

    # Plasma-style colour ramp normalised by maturity index.
    n = len(items)
    palette = [_plasma_at(i / max(n - 1, 1)) for i in range(n)]

    k_grid = np.linspace(k_min, k_max, n_points)

    fig = go.Figure()
    for (expiry, params), color in zip(items, palette):
        iv_curve = params.iv(k_grid)
        fig.add_trace(go.Scatter(
            x=k_grid,
            y=iv_curve,
            mode="lines",
            name=f"{pd.Timestamp(expiry).date()}  ({params.T:.2f}y)",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{pd.Timestamp(expiry).date()}</b><br>"
                "ln(K/F) = %{x:+.3f}<br>"
                "IV = %{y:.4f}<extra></extra>"
            ),
            opacity=0.9,
        ))

    # ATM vertical guide.
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.18)",
                                  width=1, dash="dot"))

    ticker = df.attrs.get("ticker", "")
    fig.update_layout(
        title=dict(
            text=f"{ticker} smile evolution  ·  all expiries  ·  "
                 f"colour = maturity (front → back)",
            font=dict(color=_TEXT_BRIGHT, size=17), x=0.02, y=0.97,
        ),
        paper_bgcolor=_BG_BLACK,
        plot_bgcolor=_BG_PANEL,
        font=dict(color=_TEXT_BRIGHT),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(size=10),
            x=1.0, xanchor="right", y=1.0, yanchor="top",
        ),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    fig.update_xaxes(title_text="log moneyness  ln(K / F)", **_axis2d())
    fig.update_yaxes(title_text="implied volatility", **_axis2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig


def _plasma_at(t: float) -> str:
    """Interpolate matplotlib's Plasma colormap at t∈[0,1] → CSS rgb.

    We hard-code 8 anchor points sampled from matplotlib.cm.plasma so
    we don't need matplotlib at runtime — the [viz] extras only need
    plotly + scipy.
    """
    anchors = [
        (0.00, ( 13,   8, 135)),  # deep blue-purple
        (0.14, ( 75,   2, 161)),
        (0.29, (125,   3, 168)),
        (0.43, (168,  34, 150)),
        (0.57, (203,  70, 121)),  # magenta
        (0.71, (229, 107,  93)),
        (0.86, (248, 161,  60)),
        (1.00, (240, 249,  33)),  # bright yellow
    ]
    t = max(0.0, min(1.0, t))
    for i in range(len(anchors) - 1):
        t0, c0 = anchors[i]
        t1, c1 = anchors[i + 1]
        if t <= t1:
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = int(c0[0] + alpha * (c1[0] - c0[0]))
            g = int(c0[1] + alpha * (c1[1] - c0[1]))
            b = int(c0[2] + alpha * (c1[2] - c0[2]))
            return f"rgb({r},{g},{b})"
    r, g, b = anchors[-1][1]
    return f"rgb({r},{g},{b})"
