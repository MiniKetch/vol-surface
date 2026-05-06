"""End-to-end mispricing scan: pull a chain, fit SVI per expiry, rank deviations.

Usage:
    python examples/scan_mispricing.py SPY
    python examples/scan_mispricing.py NVDA --top 20 --no-cache

Outputs:
    * A printed table of the top-K most-deviating contracts
        (z-score = residual_iv / spread_iv).
    * One HTML smile-with-overlay per expiry that has a fit, written
        to ``output/<TICKER>_smile_<expiry>.html``.

Read the output as a *hypothesis generator*, not a trade signal.
Z > 0 means the market IV is above the model (option rich); Z < 0 is
the opposite. Around scheduled events (earnings, FOMC, ex-dividend)
expect specific expiries to load up with positive z — that's the
market pricing the event vol that SVI's smooth surface can't see.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from vol_surface.data import DataFetcher, RiskFreeCurve
from vol_surface.scanner import scan_mispricing
from vol_surface.viz import render_smile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker")
    parser.add_argument("--top", type=int, default=15,
                        help="How many ranked contracts to print (default: 15)")
    parser.add_argument("--side", choices=["both", "rich", "cheap"], default="both",
                        help="Filter the printed top-N to one side (default: both)")
    parser.add_argument("--by", choices=["residual_iv", "zscore", "dollar"],
                        default="residual_iv",
                        help="Ranking metric. residual_iv = absolute IV miss "
                             "(default, content-friendly). zscore = miss / "
                             "half-spread-in-IV (statistically rigorous but "
                             "amplifies noise on tight markets like SPY). "
                             "dollar = residual × vega ($/contract PnL).")
    parser.add_argument("--max-expiries", type=int, default=12)
    parser.add_argument("--min-oi", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  ▸ Fetching {args.ticker} options chain…")
    rates = RiskFreeCurve()
    print(f"    rate source: {rates.source}")

    fetcher = DataFetcher(rates=rates, use_cache=not args.no_cache)
    df = fetcher.get(args.ticker,
                     max_expiries=args.max_expiries,
                     min_open_interest=args.min_oi)
    spot = df.attrs["spot"]
    n_iv = int(df["iv"].notna().sum())
    print(f"    spot ${spot:,.2f}  ·  {len(df):,} contracts  ·  {n_iv:,} with IV")

    print("  ▸ Fitting SVI per expiry…")
    report = scan_mispricing(df)
    n_fits = len(report.fits)
    n_scored = report.n_contracts
    print(f"    {n_fits} expiries fit  ·  {n_scored:,} contracts scored")

    if n_fits == 0:
        print("  ✗ No expiries had enough usable points — try --max-expiries 16 "
              "or a more liquid name.")
        return 1

    # ---- Top-N table ----
    top = report.top(n=args.top, side=args.side, by=args.by)
    if top.empty:
        print("  (No contracts matched the side filter.)")
    else:
        print(f"\n  Top {len(top)} {args.side} deviations (ranked by {args.by}):")
        cols = ["type", "strike", "expiry", "ttm", "iv", "svi_iv",
                "residual_iv", "spread_iv", "zscore", "open_interest"]
        view = top[cols].copy()
        view["expiry"] = pd.to_datetime(view["expiry"]).dt.strftime("%Y-%m-%d")
        # Compact numeric formatting for terminal output.
        formatters = {
            "strike":      "{:>8.2f}".format,
            "ttm":         "{:>5.3f}".format,
            "iv":          "{:>7.4f}".format,
            "svi_iv":      "{:>7.4f}".format,
            "residual_iv": "{:>+8.4f}".format,
            # 6 decimals — typical SPY half-spread is ~5 bp of IV (0.00050).
            "spread_iv":   "{:>9.6f}".format,
            "zscore":      "{:>+7.2f}".format,
        }
        # Drop ±inf z-scores from the printed table — those are
        # contracts where vega was zero (price spread / 0). The scanner
        # still tracks them in `report.scored` if you want to inspect.
        import numpy as np
        view = view[np.isfinite(view["zscore"].to_numpy())]
        print(view.to_string(index=False, formatters=formatters))

    # ---- Smile overlays for the worst-fit expiries ----
    # Sort fits by RSS / n_points (avg residual squared) — biggest first.
    fits_ranked = sorted(
        report.fits.items(),
        key=lambda kv: (kv[1].rss or 0) / max(kv[1].n_points or 1, 1),
        reverse=True,
    )
    n_smiles = min(3, len(fits_ranked))
    print(f"\n  ▸ Writing {n_smiles} smile overlays (worst-fit expiries)…")
    for expiry, params in fits_ranked[:n_smiles]:
        path = args.output_dir / f"{args.ticker}_smile_{pd.Timestamp(expiry).date()}.html"
        render_smile(df, expiry, svi_params=params, save_html=path)
        print(f"    {path}  (a={params.a:.4f}  b={params.b:.3f}  "
              f"ρ={params.rho:+.2f}  m={params.m:+.3f}  σ={params.sigma:.3f})")

    print("\n  ✓ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
