"""Render a vol surface from synthetic data — no internet required.

Useful for:
* CI / smoke tests where yfinance is unavailable
* Iterating on the renderer without pulling fresh chains
* Sanity-checking the math: we generate prices from a *known* skewed
  surface, recover IVs via the C++ kernel, and the rendered picture
  should look like what we put in.

Run it:
    python examples/build_surface_synthetic.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from vol_surface import OptionType, bs_price_batch, implied_vol_batch
from vol_surface.viz import render_surface, render_smile


def synthetic_chain(*, spot: float = 450.0, n_expiries: int = 6,
                    n_strikes: int = 25, seed: int = 7) -> pd.DataFrame:
    """Generate an enriched-chain-style DataFrame from a known surface.

    The "true" surface is a stylised SPX-style smirk: vol goes up as
    you move down in strike (puts skew), and the term structure is in
    contango (longer maturities have higher vol).
    """
    rng = np.random.default_rng(seed)
    today = dt.datetime.now()

    expiries_dte = np.linspace(14, 365, n_expiries).astype(int)
    strikes = np.linspace(0.7, 1.4, n_strikes) * spot

    rows = []
    for dte in expiries_dte:
        T = dte / 365.0
        expiry = today + dt.timedelta(days=int(dte))
        for K in strikes:
            for typ in ("C", "P"):
                rows.append({
                    "ticker": "SYNTH",
                    "type": typ,
                    "strike": float(K),
                    "expiry": expiry,
                    "ttm": T,
                })
    df = pd.DataFrame(rows)

    # True vol surface: 18 % at-the-forward, +20 vol-points of put skew
    # at -30 % log-moneyness, +5 vol-points of contango at 1Y.
    r_const = 0.04
    q_const = 0.015
    forward = spot * np.exp((r_const - q_const) * df["ttm"])
    log_mny = np.log(df["strike"] / forward)
    true_iv = (
        0.18
        + 0.20 * np.maximum(-log_mny, 0)        # put skew
        + 0.05 * df["ttm"]                      # contango
        + 0.02 * rng.standard_normal(len(df))   # quote noise
    ).clip(0.05, 1.5)

    # Generate "market" prices from the true surface, then add a tiny
    # bid-ask noise so the round-trip isn't an identity check.
    types = np.where(df["type"] == "C", 0, 1).astype(np.int8)
    S = np.full(len(df), spot)
    K = df["strike"].to_numpy()
    T = df["ttm"].to_numpy()
    r = np.full(len(df), r_const)
    q = np.full(len(df), q_const)

    fair = bs_price_batch(types, S, K, T, r, q, true_iv.to_numpy())
    spread = 0.005 * fair + 0.01      # 0.5 % + 1 cent
    df["bid"] = np.maximum(fair - spread / 2, 0)
    df["ask"] = fair + spread / 2
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_pct"] = spread / df["mid"]
    df["volume"] = pd.array(rng.integers(0, 5000, len(df)), dtype="Int64")
    df["open_interest"] = pd.array(
        rng.integers(0, 20000, len(df)), dtype="Int64")

    # Now run the *same* path the real fetcher uses: enrich with rate,
    # moneyness, and IV-via-C++.
    df["r"] = r_const
    df["q"] = q_const
    df["moneyness"] = df["strike"] / spot
    df["log_moneyness"] = log_mny

    df["iv"] = implied_vol_batch(
        types=types,
        market_price=df["mid"].to_numpy(),
        S=S, K=K, T=T, r=r, q=q,
    )

    # Stash metadata the same way DataFetcher does.
    df.attrs["ticker"] = "SYNTH"
    df.attrs["spot"] = spot
    df.attrs["dividend_yield"] = q_const
    df.attrs["fetched_at"] = today
    return df


def main() -> int:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    print("  ▸ Generating synthetic chain…")
    df = synthetic_chain()
    print(f"    {len(df):,} contracts  ·  IVs solved: {df['iv'].notna().sum():,}")

    surface_path = out_dir / "synth_surface.html"
    smile_path   = out_dir / "synth_smile.html"

    print(f"  ▸ Rendering 3D surface → {surface_path}")
    render_surface(df, save_html=surface_path)

    # Pick the middle expiry for the smile slice.
    expiries = sorted(df["expiry"].unique())
    smile_expiry = expiries[len(expiries) // 2]
    print(f"  ▸ Rendering smile ({smile_expiry.date() if hasattr(smile_expiry, 'date') else smile_expiry}) → {smile_path}")
    render_smile(df, smile_expiry, save_html=smile_path)

    print("  ✓ Open either HTML in a browser. The surface should show the")
    print("    classic SPX smirk: high vol on the put side, smooth contango.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
