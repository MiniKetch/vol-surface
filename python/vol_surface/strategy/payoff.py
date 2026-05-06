"""Strategy P&L visualizer.

Parse simple human-readable position strings and produce two curves
in one chart:

* **Expiry payoff** — pure intrinsic-value P&L at the strategy's
  expiry, the textbook "hockey stick" for a single option, kinks for
  spreads and butterflies. Doesn't depend on σ or T, so it's clean.

* **Current mark-to-market** — what the strategy is worth *right now*
  if the spot were at each grid point, holding the rest of the
  surface (IVs, time, rates) constant. Uses ``bs_price_batch`` from
  the C++ kernel to vectorize across the spot grid + every leg.

The MTM curve typically sits *above* the expiry payoff outside the
breakevens (time value still in the options) and approaches it as
expiry nears or as spot moves deep ITM.

Position-string syntax:

    "long 10 SPY 740C 2026-06-18"            # 10 long calls
    "short 5 SPY 700P 2026-06-18"            # 5 short puts
    "long 1 740C 2026-06-18, short 1 750C 2026-06-18"   # call spread
    "long 1 700P 26-06-18, long 1 740C 26-06-18"        # straddle-ish

Tickers are optional (the dashboard knows which underlying you're
analysing). Strikes carry a single 'C' or 'P' suffix. Expiries accept
YYYY-MM-DD or YY-MM-DD or any pandas-parseable date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from vol_surface import OptionType, bs_price_batch


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    """One option leg of a strategy."""
    side: str               # 'long' or 'short' (sign of the position)
    quantity: float         # always non-negative; sign comes from side
    option_type: str        # 'C' or 'P'
    strike: float
    expiry: pd.Timestamp
    entry_price: Optional[float] = None  # if known, used for P&L

    @property
    def signed_qty(self) -> float:
        return self.quantity if self.side == "long" else -self.quantity


@dataclass
class Strategy:
    """A bundle of legs with a label."""
    name: str
    legs: List[Leg] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Single-leg pattern: optional side, qty, optional ticker, K + C/P, expiry.
# Examples it accepts:
#   long 10 SPY 740C 2026-06-18
#   short 5 700P 2026-06-18
#   1 740C 2026-06-18                        (defaults to long)
_LEG_RE = re.compile(
    r"""
    \s*
    (?:(?P<side>long|short|buy|sell|\-)\s+)?
    (?P<qty>\d+(?:\.\d+)?)\s+
    (?:(?P<ticker>[A-Za-z][A-Za-z0-9.\-]*)\s+)?
    (?P<strike>\d+(?:\.\d+)?)
    \s*
    (?P<typ>[CcPp])
    \s+
    (?P<expiry>\d{2,4}[-/]\d{1,2}[-/]\d{1,2})
    \s*
    """,
    re.VERBOSE,
)


def parse_strategy(text: str, *, name: str = "Strategy") -> Strategy:
    """Parse a comma-separated list of leg specifications.

    Raises ``ValueError`` with a helpful message if a leg can't be
    parsed — surfaces it in the dashboard so the user can fix typos.
    """
    legs: List[Leg] = []
    pieces = [p.strip() for p in text.split(",") if p.strip()]
    if not pieces:
        raise ValueError("Empty strategy — type at least one leg.")

    for piece in pieces:
        m = _LEG_RE.fullmatch(piece)
        if m is None:
            raise ValueError(
                f"Couldn't parse leg: {piece!r}. "
                "Use e.g. 'long 10 740C 2026-06-18'."
            )
        side_raw = (m.group("side") or "long").lower()
        side = "short" if side_raw in ("short", "sell", "-") else "long"
        legs.append(Leg(
            side=side,
            quantity=float(m.group("qty")),
            option_type=m.group("typ").upper(),
            strike=float(m.group("strike")),
            expiry=pd.to_datetime(m.group("expiry")),
        ))
    return Strategy(name=name, legs=legs)


# ---------------------------------------------------------------------------
# P&L computation
# ---------------------------------------------------------------------------

def compute_payoff(
    strategy: Strategy,
    df: pd.DataFrame,
    *,
    spot_lo: float | None = None,
    spot_hi: float | None = None,
    n_grid: int = 200,
) -> pd.DataFrame:
    """Compute expiry payoff and current MTM curves for the strategy.

    Parameters
    ----------
    strategy
        Parsed strategy with one or more legs.
    df
        Enriched chain DataFrame (used to look up current IVs, plus
        spot, r, q via .attrs).
    spot_lo, spot_hi
        Spot range to evaluate. Defaults to ±25 % of current spot.
    n_grid
        Resolution of the spot grid.

    Returns
    -------
    DataFrame with columns:
        spot         — grid point
        payoff       — total payoff at the strategy's expiry
        mtm          — current mark-to-market under the same spot
        mtm_minus_now — MTM relative to today's MTM (P&L if spot moves)
    """
    if not strategy.legs:
        raise ValueError("Strategy has no legs.")

    spot_now = float(df.attrs.get("spot", 100.0))
    if spot_lo is None: spot_lo = spot_now * 0.75
    if spot_hi is None: spot_hi = spot_now * 1.25
    spot_grid = np.linspace(spot_lo, spot_hi, n_grid)

    # We compute the contribution per leg, then sum.
    payoff = np.zeros_like(spot_grid)
    mtm    = np.zeros_like(spot_grid)
    legs_info = []

    for leg in strategy.legs:
        # --- Expiry payoff: pure intrinsic value ---
        if leg.option_type == "C":
            leg_payoff = np.maximum(spot_grid - leg.strike, 0.0)
        else:
            leg_payoff = np.maximum(leg.strike - spot_grid, 0.0)
        payoff += leg.signed_qty * leg_payoff

        # --- Current MTM: BS price under each spot ---
        leg_iv, leg_T, leg_r, leg_q, status = _lookup_leg_context(leg, df)
        if leg_iv is None or leg_T is None:
            # No matching contract in the chain — fall back to expiry
            # payoff (no time value visible). Caller will see the
            # status warning.
            mtm += leg.signed_qty * leg_payoff
            legs_info.append(_leg_info(leg, leg_iv, leg_T, status))
            continue

        n = len(spot_grid)
        types = np.full(n, 0 if leg.option_type == "C" else 1, dtype=np.int8)
        prices = bs_price_batch(
            types,
            spot_grid.astype(float),
            np.full(n, leg.strike),
            np.full(n, leg_T),
            np.full(n, leg_r),
            np.full(n, leg_q),
            np.full(n, leg_iv),
        )
        mtm += leg.signed_qty * prices
        legs_info.append(_leg_info(leg, leg_iv, leg_T, status))

    # MTM at the *current* spot — useful as a P&L baseline.
    idx_now = int(np.argmin(np.abs(spot_grid - spot_now)))
    mtm_at_now = float(mtm[idx_now])

    out = pd.DataFrame({
        "spot": spot_grid,
        "payoff": payoff,
        "mtm": mtm,
        "mtm_minus_now": mtm - mtm_at_now,
    })
    out.attrs["spot_now"]  = spot_now
    out.attrs["mtm_now"]   = mtm_at_now
    out.attrs["legs_info"] = legs_info
    out.attrs["strategy"]  = strategy.name
    return out


# ---------------------------------------------------------------------------
# Plotly rendering
# ---------------------------------------------------------------------------

def render_payoff(
    pnl: pd.DataFrame,
    *,
    save_html: str | Path | None = None,
):
    """Render the payoff & MTM curves with breakeven & current-spot
    annotations."""
    from vol_surface.viz.surface import (
        _BG_BLACK, _BG_PANEL, _TEXT_BRIGHT,
        _CALL_COLOR, _FIT_COLOR, _PUT_COLOR, _axis2d, _require_plotly,
    )
    go, _ = _require_plotly()

    spot_now = pnl.attrs.get("spot_now")
    mtm_now  = pnl.attrs.get("mtm_now", 0.0)
    name     = pnl.attrs.get("strategy", "Strategy")

    fig = go.Figure()

    # Expiry payoff — the "hockey stick"
    fig.add_trace(go.Scatter(
        x=pnl["spot"], y=pnl["payoff"],
        mode="lines",
        name="Payoff at expiry",
        line=dict(color=_PUT_COLOR, width=2.5, dash="dot"),
        hovertemplate="Spot=%{x:.2f}<br>Payoff=%{y:+.2f}<extra></extra>",
    ))

    # Current MTM
    fig.add_trace(go.Scatter(
        x=pnl["spot"], y=pnl["mtm"],
        mode="lines",
        name="Current MTM",
        line=dict(color=_FIT_COLOR, width=3),
        hovertemplate="Spot=%{x:.2f}<br>MTM=%{y:+.2f}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))

    # Current-spot vertical line
    if spot_now is not None:
        fig.add_vline(
            x=spot_now,
            line=dict(color=_CALL_COLOR, width=1.5, dash="dash"),
            annotation=dict(
                text=f"spot ${spot_now:,.2f}",
                font=dict(color=_CALL_COLOR, size=11),
                xanchor="left", yanchor="top",
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor=_CALL_COLOR,
            ),
        )

    fig.update_layout(
        title=dict(
            text=f"{name} · payoff at expiry vs current MTM",
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
    fig.update_xaxes(title_text="underlying price at evaluation", **_axis2d())
    fig.update_yaxes(title_text="$ per share (×100 for contracts)", **_axis2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_leg_context(leg: Leg, df: pd.DataFrame):
    """Find IV/T/r/q for a leg by matching to the closest chain contract."""
    expiry = pd.Timestamp(leg.expiry).normalize()
    same_expiry = df[(pd.to_datetime(df["expiry"]).dt.normalize() == expiry) &
                     (df["type"] == leg.option_type)]
    if same_expiry.empty:
        return None, None, None, None, "no matching expiry/type in chain"

    # Closest strike.
    same_expiry = same_expiry.dropna(subset=["iv"])
    if same_expiry.empty:
        return None, None, None, None, "matching expiry has no usable IVs"
    idx = (same_expiry["strike"] - leg.strike).abs().idxmin()
    row = same_expiry.loc[idx]
    return (float(row["iv"]),
            float(row["ttm"]),
            float(row["r"]),
            float(row["q"]),
            f"matched K={row['strike']:.2f}")


def _leg_info(leg: Leg, iv, T, status: str) -> dict:
    return {
        "side": leg.side, "quantity": leg.quantity,
        "type": leg.option_type, "strike": leg.strike,
        "expiry": str(leg.expiry.date()),
        "iv": None if iv is None else float(iv),
        "ttm": None if T is None else float(T),
        "status": status,
    }
