"""Small shared helpers used across the data, scanner, viz, and strategy
layers. The audit found a half-dozen places that re-implemented forward-
price arithmetic and contract-side filtering with subtle drift between
implementations; centralising them here removes those bugs by
construction.

Everything here is pure math + pandas — no I/O, no plotly, no Streamlit
imports — so importing from this module is always cheap.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Forward / log-moneyness
# ---------------------------------------------------------------------------

def forward(S, r, q, T):
    """Forward price under continuous compounding: ``F = S · exp((r-q)·T)``.

    Vectorised — accepts scalars or numpy arrays / pandas Series. For
    arrays of mixed type, the result follows numpy broadcasting.
    """
    return np.asarray(S) * np.exp((np.asarray(r) - np.asarray(q)) * np.asarray(T))


def log_moneyness(K, S, r, q, T):
    """Log-moneyness ``ln(K / F)`` with F = forward(S, r, q, T)."""
    return np.log(np.asarray(K) / forward(S, r, q, T))


# ---------------------------------------------------------------------------
# Contract-side filter (OTM / calls / puts / both)
# ---------------------------------------------------------------------------

Side = Literal["OTM", "calls", "puts", "both"]


def filter_by_side(df: pd.DataFrame, side: Side) -> pd.DataFrame:
    """Keep contracts on a given side of the smile.

    ``OTM`` = liquid wings: OTM puts (k < 0) + OTM calls (k ≥ 0).
    The dataframe must have ``type`` and ``log_moneyness`` columns.

    The function does **not** drop NaN IVs or apply IV bands — those
    concerns belong to the caller. We only do the side selection so
    the same filter logic appears in exactly one place.
    """
    if side == "calls":
        return df[df["type"] == "C"]
    if side == "puts":
        return df[df["type"] == "P"]
    if side == "OTM":
        otm_calls = (df["type"] == "C") & (df["log_moneyness"] >= 0)
        otm_puts  = (df["type"] == "P") & (df["log_moneyness"] <  0)
        return df[otm_calls | otm_puts]
    if side == "both":
        return df
    raise ValueError(f"Unknown side: {side!r}")


# ---------------------------------------------------------------------------
# Bid-ask price spread → IV-spread proxy
# ---------------------------------------------------------------------------

def iv_spread_from_price(price_spread, vega):
    """Convert a price-spread (in dollars) to an IV-spread (in vol units).

    ``half-spread_iv ≈ price_spread / (2 · vega)``. Vega in BS units is
    "dollars per 1.0 of σ". Returns NaN where vega ≤ 0 or non-finite.
    """
    spread = np.asarray(price_spread, dtype=float)
    v      = np.asarray(vega, dtype=float)
    out = np.full_like(spread, np.nan)
    mask = np.isfinite(v) & (v > 0) & np.isfinite(spread) & (spread >= 0)
    np.divide(spread, 2.0 * v, out=out, where=mask)
    return out


# ---------------------------------------------------------------------------
# DataFrame.attrs survival across pickle / cache boundaries
# ---------------------------------------------------------------------------

# Keys we treat as "context metadata" that travels with a chain. We
# explicitly enumerate them rather than relying on .attrs because:
#   1. Streamlit's @st.cache_data pickles the DataFrame and drops .attrs.
#   2. Several pandas operations (.copy(), .dropna(), groupby) silently
#      lose .attrs depending on pandas version.
# Using `extract_attrs` and `attach_attrs` at every cache boundary
# eliminates that whole class of bug.
ATTRS_KEYS = ("ticker", "spot", "dividend_yield", "fetched_at")


def extract_attrs(df: pd.DataFrame) -> dict:
    """Pull the canonical context-metadata keys off a DataFrame.

    Returns a plain dict — pickle-safe, network-safe, copy-safe.
    """
    return {k: df.attrs.get(k) for k in ATTRS_KEYS if k in df.attrs}


def attach_attrs(df: pd.DataFrame, attrs: dict) -> pd.DataFrame:
    """Re-attach context metadata to a DataFrame (mutates in place,
    returns the same df for convenience)."""
    for k, v in attrs.items():
        df.attrs[k] = v
    return df


def get_attr(df: pd.DataFrame, key: str, default=None):
    """Get a context value with a sensible default. Use this everywhere
    instead of ``df.attrs.get(...)`` — same surface, but the default
    behaviour is documented and consistent."""
    return df.attrs.get(key, default)
