"""High-level orchestrator: pull a chain, attach rates, compute IVs.

This is the entry point most callers want. It bundles the steps:

  1. Fetch a clean options chain (yfinance + filters).
  2. Look up the risk-free rate per expiry from the Treasury curve.
  3. Call into the C++ kernel to compute implied volatilities in batch.

Output is a single DataFrame ready for the surface renderer or the
mispricing scanner.

Caching: each ticker's chain is cached to a Parquet file in
``~/.cache/vol_surface/`` keyed by ticker + date. yfinance is rate-
limited and occasionally flaky, so iterating on viz code without
re-pulling is significantly less painful.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from vol_surface import OptionType, implied_vol_batch
from vol_surface.data.chains import OptionsChain, fetch_chain
from vol_surface.data.rates import RiskFreeCurve

_CACHE_DIR = Path.home() / ".cache" / "vol_surface"


class DataFetcher:
    """End-to-end chain → IV pipeline.

    Parameters
    ----------
    rates
        A pre-built ``RiskFreeCurve`` (default: auto-detect FRED vs
        snapshot).
    cache_dir
        Where to stash Parquet caches. Set to ``None`` to disable.
    cache_ttl_minutes
        How fresh a cached chain must be to skip the live pull.
        Default 30 mins — short enough to feel "live", long enough to
        not hammer yfinance during iteration.
    """

    def __init__(
        self,
        rates: RiskFreeCurve | None = None,
        cache_dir: Path | str | None = _CACHE_DIR,
        cache_ttl_minutes: int = 30,
        use_cache: bool = True,
    ) -> None:
        self.rates = rates or RiskFreeCurve()
        self.cache_dir = Path(cache_dir) if (cache_dir and use_cache) else None
        self.cache_ttl = dt.timedelta(minutes=cache_ttl_minutes)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, ticker: str, **fetch_kwargs) -> pd.DataFrame:
        """Return a DataFrame of contracts for ``ticker`` with IVs filled in.

        Schema: every column from ``OptionsChain.contracts`` plus
        ``r`` (risk-free rate at this expiry, continuous), ``q``
        (dividend yield), ``moneyness`` (K/S), ``log_moneyness``
        (log(K/F)), and ``iv`` (implied volatility, NaN when
        unsolvable).
        """
        chain = self._load_chain(ticker, **fetch_kwargs)
        return self._enrich(chain)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_chain(self, ticker: str, **fetch_kwargs) -> OptionsChain:
        cache_path = self._cache_path(ticker, fetch_kwargs) if self.cache_dir else None
        if cache_path and cache_path.exists():
            age = dt.datetime.now() - dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
            if age < self.cache_ttl:
                return _load_cached_chain(cache_path)

        chain = fetch_chain(ticker, **fetch_kwargs)

        if cache_path:
            _save_cached_chain(chain, cache_path)
        return chain

    def _cache_path(self, ticker: str, fetch_kwargs: dict) -> Path:
        """Cache key includes the fetch params, so changing max_expiries
        or DTE bounds invalidates the cache rather than reusing a stale
        narrower fetch."""
        assert self.cache_dir is not None
        import hashlib
        date_tag = dt.date.today().isoformat()
        # Stable hash of the kwargs that affect what gets fetched.
        relevant = {k: fetch_kwargs.get(k) for k in (
            "max_expiries", "min_dte_days", "max_dte_days",
            "min_open_interest", "max_spread_pct", "require_volume",
        ) if k in fetch_kwargs}
        key = repr(sorted(relevant.items())).encode()
        param_hash = hashlib.sha1(key).hexdigest()[:8]
        return self.cache_dir / f"{ticker.upper()}_{date_tag}_{param_hash}.parquet"

    def _enrich(self, chain: OptionsChain) -> pd.DataFrame:
        """Attach rate, moneyness, and IV columns."""
        df = chain.contracts.copy()

        # Risk-free rate per row (vector lookup keyed on TTM).
        df["r"] = self.rates.rate(df["ttm"].to_numpy())
        df["q"] = chain.dividend_yield

        # Useful coordinates for the surface.
        df["moneyness"] = df["strike"] / chain.spot
        forward = chain.spot * np.exp((df["r"] - df["q"]) * df["ttm"])
        df["log_moneyness"] = np.log(df["strike"] / forward)

        # Batch IV solve in C++.
        types = np.where(df["type"].to_numpy() == "C", 0, 1).astype(np.int8)
        df["iv"] = implied_vol_batch(
            types=types,
            market_price=df["mid"].to_numpy(),
            S=np.full(len(df), chain.spot),
            K=df["strike"].to_numpy(),
            T=df["ttm"].to_numpy(),
            r=df["r"].to_numpy(),
            q=df["q"].to_numpy(),
        )

        # Tag spot + fetch metadata for downstream code.
        df.attrs["ticker"]    = chain.ticker
        df.attrs["spot"]      = chain.spot
        df.attrs["dividend_yield"] = chain.dividend_yield
        df.attrs["fetched_at"] = chain.fetched_at
        return df


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _save_cached_chain(chain: OptionsChain, path: Path) -> None:
    df = chain.contracts.copy()
    df.attrs["ticker"] = chain.ticker
    df.attrs["spot"] = chain.spot
    df.attrs["dividend_yield"] = chain.dividend_yield
    df.attrs["fetched_at"] = chain.fetched_at.isoformat()
    # Parquet doesn't preserve .attrs — stash them in a sidecar.
    df.to_parquet(path, index=False)
    sidecar = path.with_suffix(".meta.json")
    sidecar.write_text(_meta_json(chain))


def _load_cached_chain(path: Path) -> OptionsChain:
    import json
    df = pd.read_parquet(path)
    sidecar = path.with_suffix(".meta.json")
    meta = json.loads(sidecar.read_text())
    return OptionsChain(
        ticker=meta["ticker"],
        spot=float(meta["spot"]),
        dividend_yield=float(meta["dividend_yield"]),
        fetched_at=dt.datetime.fromisoformat(meta["fetched_at"]),
        contracts=df,
    )


def _meta_json(chain: OptionsChain) -> str:
    import json
    return json.dumps({
        "ticker": chain.ticker,
        "spot": chain.spot,
        "dividend_yield": chain.dividend_yield,
        "fetched_at": chain.fetched_at.isoformat(),
    })
