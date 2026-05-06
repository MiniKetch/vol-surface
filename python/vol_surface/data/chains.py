"""Options-chain fetcher.

Wraps ``yfinance`` and produces a clean DataFrame ready for IV
computation. Handles the usual data-quality landmines:

* Zero / NaN bid or ask
* Wide bid-ask spreads (we use mid, but also flag anything > 50 % of mid)
* Near-expiry contracts (< 7 calendar days — IV blows up on noise)
* Zero open interest (often stale quotes)

The result is typed so downstream code (the C++ IV solver, the
3D surface renderer) can rely on a stable schema regardless of which
data provider you swap in later.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Columns we always emit, in this order. Downstream code can rely on this.
_SCHEMA: dict[str, str] = {
    "ticker":    "string",
    "type":      "string",   # 'C' or 'P'
    "strike":    "float64",
    "expiry":    "datetime64[ns]",
    "ttm":       "float64",  # years to expiry, ACT/365
    "bid":       "float64",
    "ask":       "float64",
    "mid":       "float64",
    "spread_pct": "float64", # (ask - bid) / mid
    "volume":    "Int64",
    "open_interest": "Int64",
}


@dataclass
class OptionsChain:
    """A snapshot of a ticker's full options chain."""
    ticker: str
    spot: float
    dividend_yield: float
    fetched_at: dt.datetime
    contracts: pd.DataFrame = field(repr=False)

    def __post_init__(self) -> None:
        # Defensive validation — catches schema drift early.
        missing = set(_SCHEMA) - set(self.contracts.columns)
        if missing:
            raise ValueError(f"Chain missing columns: {sorted(missing)}")

    @property
    def n_contracts(self) -> int:
        return len(self.contracts)

    def calls(self) -> pd.DataFrame:
        return self.contracts[self.contracts["type"] == "C"]

    def puts(self) -> pd.DataFrame:
        return self.contracts[self.contracts["type"] == "P"]


def fetch_chain(
    ticker: str,
    *,
    max_expiries: int = 8,
    min_dte_days: int = 7,
    max_dte_days: int = 365,
    min_open_interest: int = 1,
    max_spread_pct: float = 0.5,
    require_volume: bool = False,
) -> OptionsChain:
    """Fetch and clean a full options chain for ``ticker``.

    Parameters
    ----------
    ticker
        Symbol, e.g. ``"SPY"``.
    max_expiries
        Cap on number of expirations to fetch (closest first). 8 covers
        the front of the curve which is what surface viz needs.
    min_dte_days, max_dte_days
        Window of days-to-expiry to keep. Inside 7 days IV becomes
        unreliable; beyond a year quotes get stale.
    min_open_interest
        Drop contracts with OI below this. 1 = "anyone is in the trade".
    max_spread_pct
        Drop contracts whose ``(ask-bid)/mid`` exceeds this. 0.5 = 50 %
        spread, which is generous; tighten for liquid names.
    require_volume
        If True, also drop contracts with 0 volume on the day. Off by
        default because OI is the more honest liquidity signal.
    """
    try:
        import yfinance as yf  # noqa: WPS433  (optional dep)
    except ImportError as exc:
        raise RuntimeError(
            "yfinance not installed. `pip install vol-surface[data]`."
        ) from exc

    tk = yf.Ticker(ticker)

    # Spot — last close. yfinance's .info.regularMarketPrice is sometimes
    # stale; the most recent daily close is more reliable.
    hist = tk.history(period="5d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker!r}")
    spot = float(hist["Close"].iloc[-1])

    # Dividend yield — `.info` is a dict; the field is sometimes a
    # decimal (0.015) and sometimes a percent (1.5). Normalize.
    info = tk.info or {}
    raw_yield = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0
    div_yield = float(raw_yield)
    if div_yield > 1.0:  # quoted as percent
        div_yield /= 100.0

    all_expiries = list(tk.options)
    if not all_expiries:
        raise RuntimeError(f"{ticker!r} has no listed options")
    expiries = _stratified_expiries(all_expiries, max_expiries,
                                     min_dte_days, max_dte_days,
                                     reference=dt.datetime.now())

    fetched_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    rows: list[pd.DataFrame] = []

    for expiry_str in expiries:
        expiry = pd.to_datetime(expiry_str)
        dte = (expiry - pd.Timestamp(fetched_at)).days
        if dte < min_dte_days or dte > max_dte_days:
            continue
        try:
            chain = tk.option_chain(expiry_str)
        except Exception:
            # yfinance occasionally 404s on a single expiry — skip it
            # rather than fail the whole fetch.
            continue
        for side, df in (("C", chain.calls), ("P", chain.puts)):
            if df is None or df.empty:
                continue
            tidy = _tidy_one_side(df, ticker, side, expiry, fetched_at)
            rows.append(tidy)

    if not rows:
        raise RuntimeError(f"No usable contracts found for {ticker!r}")

    chain_df = pd.concat(rows, ignore_index=True)
    chain_df = _filter(chain_df, min_open_interest, max_spread_pct, require_volume)
    chain_df = chain_df.astype(_SCHEMA, errors="ignore")[list(_SCHEMA)]

    return OptionsChain(
        ticker=ticker,
        spot=spot,
        dividend_yield=div_yield,
        fetched_at=fetched_at,
        contracts=chain_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tidy_one_side(
    df: pd.DataFrame,
    ticker: str,
    side: str,
    expiry: pd.Timestamp,
    fetched_at: dt.datetime,
) -> pd.DataFrame:
    """Reshape one (calls or puts) DataFrame into our schema."""
    # ACT/365 — close enough for IV. Fancier daycounts (252 trading
    # days) make sense for theta but not pricing.
    ttm = max((expiry - pd.Timestamp(fetched_at)).total_seconds()
              / (365.0 * 24 * 3600), 0.0)

    bid = pd.to_numeric(df.get("bid"), errors="coerce")
    ask = pd.to_numeric(df.get("ask"), errors="coerce")
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid.where(mid > 0)

    return pd.DataFrame({
        "ticker": ticker,
        "type":   side,
        "strike": pd.to_numeric(df["strike"], errors="coerce"),
        "expiry": expiry,
        "ttm":    ttm,
        "bid":    bid,
        "ask":    ask,
        "mid":    mid,
        "spread_pct": spread_pct,
        "volume":        pd.to_numeric(df.get("volume"), errors="coerce").astype("Int64"),
        "open_interest": pd.to_numeric(df.get("openInterest"), errors="coerce").astype("Int64"),
    })


def _stratified_expiries(
    all_expiries: list[str],
    max_n: int,
    min_dte: int,
    max_dte: int,
    *,
    reference: dt.datetime,
) -> list[str]:
    """Pick a stratified sample of expiries spanning the term structure.

    Naive ``[:max_n]`` slicing on yfinance picks 8 weeklies bunched in
    the next 10 days — the term-structure axis of the surface ends up
    degenerate. We instead:

      1. Drop any expiries outside the [min_dte, max_dte] window.
      2. Take the first 3 ("front weeklies") for skew resolution.
      3. Distribute the remaining ``max_n - 3`` slots evenly (in time)
         across the rest of the available range, so the surface gets
         a real Y axis.

    With max_n=12 on SPY, you'll typically end up with 3 weeklies in
    the next 2 weeks, then ~9 monthlies/quarterlies out to the back of
    the chain.
    """
    ref_ts = pd.Timestamp(reference)
    in_window: list[str] = []
    for e in all_expiries:
        dte = (pd.to_datetime(e) - ref_ts).days
        if min_dte <= dte <= max_dte:
            in_window.append(e)
    if len(in_window) <= max_n:
        return in_window

    front_count = min(3, max_n)
    front = in_window[:front_count]
    rest  = in_window[front_count:]
    n_left = max_n - front_count
    if n_left <= 0:
        return front

    # Evenly-spaced indices into `rest` — np.linspace gives us float
    # indices that we cast to int, then dedupe to handle small lists.
    idx = sorted(set(np.linspace(0, len(rest) - 1, n_left).astype(int).tolist()))
    sampled = [rest[i] for i in idx]
    return front + sampled


def _filter(
    df: pd.DataFrame,
    min_oi: int,
    max_spread: float,
    require_volume: bool,
) -> pd.DataFrame:
    """Apply the standard liquidity / quality filters."""
    keep = (
        df["bid"].notna() & (df["bid"] > 0) &
        df["ask"].notna() & (df["ask"] > 0) &
        df["mid"].notna() & (df["mid"] > 0) &
        df["spread_pct"].notna() & (df["spread_pct"] <= max_spread) &
        df["open_interest"].fillna(0).astype("int64").ge(min_oi)
    )
    if require_volume:
        keep &= df["volume"].fillna(0).astype("int64").gt(0)
    return df[keep].copy()
