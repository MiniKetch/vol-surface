"""End-to-end demo: pull a chain, compute IVs, render the 3D surface.

Usage:
    python examples/build_surface.py SPY
    python examples/build_surface.py AAPL --max-expiries 6 --output spy.html

Requires the [data,viz] extras:
    pip install -e ".[data,viz]"

Set FREDAPI_KEY for live risk-free rates; otherwise the bundled
Treasury snapshot is used (perfectly fine for a demo).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from vol_surface.data import DataFetcher, RiskFreeCurve
from vol_surface.viz import render_smile, render_surface


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", help="Symbol, e.g. SPY")
    parser.add_argument("--max-expiries", type=int, default=12,
                        help="Cap on expirations to fetch (default: 12). "
                             "We stratify across the term structure: 3 front "
                             "weeklies + the rest spaced out to back-of-chain.")
    parser.add_argument("--min-dte",  type=int, default=7,
                        help="Drop contracts expiring sooner than this (days)")
    parser.add_argument("--max-dte",  type=int, default=365,
                        help="Drop contracts expiring later than this (days)")
    parser.add_argument("--min-oi",   type=int, default=5,
                        help="Min open interest per contract")
    parser.add_argument("--output",   type=Path, default=None,
                        help="Output HTML path (default: ./<ticker>_surface.html)")
    parser.add_argument("--smile-expiry", type=str, default=None,
                        help="If given, also render a 2D smile for this expiry "
                             "(YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the local Parquet cache and force a "
                             "fresh yfinance fetch.")
    args = parser.parse_args()

    out = args.output or Path(f"{args.ticker.upper()}_surface.html")

    print(f"  ▸ Fetching {args.ticker} options chain (yfinance)…")
    rates = RiskFreeCurve()
    print(f"    risk-free rate source: {rates.source}")

    fetcher = DataFetcher(rates=rates, use_cache=not args.no_cache)
    df = fetcher.get(
        args.ticker,
        max_expiries=args.max_expiries,
        min_dte_days=args.min_dte,
        max_dte_days=args.max_dte,
        min_open_interest=args.min_oi,
    )

    n_total = len(df)
    n_iv    = int(df["iv"].notna().sum())
    spot    = df.attrs["spot"]
    print(f"    spot ${spot:,.2f}  ·  {n_total:,} contracts  ·  {n_iv:,} with IV")

    # Show which expiries we ended up with — handy when picking a
    # --smile-expiry without guessing.
    expiries = sorted(df["expiry"].dropna().unique())
    if expiries:
        bullets = ", ".join(pd.Timestamp(e).date().isoformat() for e in expiries)
        print(f"    expiries pulled: {bullets}")

    print(f"  ▸ Rendering 3D surface → {out}")
    render_surface(df, save_html=out)

    if args.smile_expiry:
        smile_path = out.with_name(out.stem + f"_smile_{args.smile_expiry}.html")
        print(f"  ▸ Rendering smile → {smile_path}")
        render_smile(df, args.smile_expiry, save_html=smile_path)

    print("  ✓ Done. Open the HTML in any browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
