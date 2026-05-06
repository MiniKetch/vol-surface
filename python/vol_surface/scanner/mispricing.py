"""Find contracts whose market IV deviates from the SVI-fitted smile.

Pipeline:

  1. Group the enriched chain DataFrame by expiry.
  2. Fit a raw-SVI slice to each expiry (weighted by 1/spread).
  3. For every contract, compute the residual:
        residual_iv = market_iv − svi_iv
  4. Convert each contract's bid-ask price spread into an IV spread
     using its vega: spread_iv ≈ price_spread / (2 · vega).
  5. Z-score = residual_iv / spread_iv. A z above ~2 means the
     market IV is two half-spreads off the model — material.

How to read the output:

    z > 0   →  market IV ABOVE the model — option *expensive* (rich)
    z < 0   →  market IV BELOW the model — option *cheap*

These are *deviations from the model*, not arbitrage. They concentrate
around scheduled events (earnings, dividends, FOMC), but also flag
genuine mispricings, stale quotes, or contracts whose risk profile
the SVI smile can't capture (e.g. a kink near a heavily-traded strike).
The scanner is a hypothesis generator — eyeball the smile overlay
before drawing conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from vol_surface import OptionType, bs_greeks_batch
from vol_surface.svi import RawSVIParams, fit_raw_svi


@dataclass
class MispricingReport:
    """Per-contract scoring + per-expiry SVI fits."""
    scored: pd.DataFrame = field(repr=False)
    fits: dict[pd.Timestamp, RawSVIParams] = field(repr=False)

    @property
    def n_contracts(self) -> int:
        return int(self.scored["zscore"].notna().sum())

    def top(self, n: int = 10, side: str = "both",
            by: str = "residual_iv") -> pd.DataFrame:
        """Top-N most-deviating contracts.

        ``side`` ∈ {"rich", "cheap", "both"}.  "both" sorts by absolute value.
        ``by``   ∈ {"residual_iv", "zscore", "dollar"}.

        ``residual_iv`` (default) ranks by the absolute IV miss — the
        most "content-meaningful" metric on tight-spread markets like
        SPY where every contract has z>20 just because spreads are
        a basis point. ``zscore`` ranks by spread-normalised deviation
        (statistically rigorous but noisy on tight quotes).
        ``dollar`` ranks by residual × vega (PnL impact in $/contract).
        """
        scored = self.scored
        if by == "residual_iv":
            key = scored["residual_iv"]
        elif by == "zscore":
            key = scored["zscore"]
        elif by == "dollar":
            key = scored["residual_iv"] * scored["vega"]
        else:  # pragma: no cover
            raise ValueError(f"Unknown ranking key: {by!r}")

        df = scored.assign(_key=key).dropna(subset=["_key"]).copy()
        if side == "rich":
            df = df[df["_key"] > 0].sort_values("_key", ascending=False)
        elif side == "cheap":
            df = df[df["_key"] < 0].sort_values("_key", ascending=True)
        elif side == "both":
            df = df.assign(_abs=df["_key"].abs()).sort_values(
                "_abs", ascending=False).drop(columns="_abs")
        else:  # pragma: no cover
            raise ValueError(f"Unknown side {side!r}")
        return df.drop(columns="_key").head(n)


def scan_mispricing(
    df: pd.DataFrame,
    *,
    side: str = "OTM",
    min_iv: float = 0.05,
    max_iv: float = 2.0,
    min_per_expiry: int = 5,
    score_k_range: float = 0.6,
    score_max_iv: float = 1.0,
) -> MispricingReport:
    """Fit SVI per expiry and score every contract by IV residual.

    Parameters
    ----------
    df
        Enriched chain DataFrame from ``DataFetcher.get(...)``.
    side
        Which contracts to use *for the fit*: ``"OTM"`` (default),
        ``"calls"``, ``"puts"``, or ``"both"``. Scoring still applies
        to every contract in the output.
    min_iv, max_iv
        IV band for the fit. Contracts outside are excluded from
        fitting but still scored against the resulting curve.
    min_per_expiry
        Skip expiries with fewer fitable points than this. SVI has 5
        free parameters; below ~5 points the fit is meaningless.
    score_k_range
        Drop contracts with ``|log_moneyness| > score_k_range`` from
        scoring entirely. Deep ITM/OTM contracts have stale quotes and
        the IV solver produces meaningless values when fed those mids.
        Default 0.6 (~50 % above/below ATM) keeps actively-traded
        strikes only.
    score_max_iv
        Drop contracts with ``iv > score_max_iv`` from scoring. An IV
        above 100 % almost always means the contract is sitting at
        intrinsic and the solver inverted noise into a giant σ.
    """
    if not {"expiry", "iv", "log_moneyness", "ttm",
            "type", "bid", "ask", "strike"} <= set(df.columns):
        raise ValueError("Need an enriched-chain DataFrame from DataFetcher.get()")

    spot = df.attrs.get("spot")
    if spot is None:
        raise ValueError("DataFrame is missing 'spot' in .attrs — did you "
                         "construct it via DataFetcher? If hand-rolling, set "
                         "df.attrs['spot'] before calling.")

    work = df.copy()

    # --- Step 1: vega for every contract (needed for the IV-spread proxy) ---
    work["vega"] = _bs_vega_vec(work, float(spot))

    # --- Step 2: per-expiry SVI fits ---
    fits: dict[pd.Timestamp, RawSVIParams] = {}
    work["svi_iv"] = np.nan
    for expiry, slice_df in work.groupby("expiry"):
        T = float(slice_df["ttm"].iloc[0])
        sub = _fit_subset(slice_df, side, min_iv, max_iv)
        if len(sub) < min_per_expiry:
            continue
        try:
            # Uniform weights by default. Spread-weighting biases the
            # fit toward ATM contracts (which have tight spreads, hence
            # large 1/spread weight), but ATM-area smiles are mild on
            # SPX/SPY — the strong put-skew signal lives in the wings.
            # Letting wings count equally gets the curve shape right.
            params = fit_raw_svi(
                k=sub["log_moneyness"].to_numpy(),
                iv=sub["iv"].to_numpy(),
                T=T,
                weights=None,
            )
        except Exception as exc:  # noqa: BLE001
            # A failed fit on one expiry shouldn't poison the whole report.
            print(f"  ⚠ SVI fit failed for {pd.Timestamp(expiry).date()}: {exc}")
            continue

        fits[pd.Timestamp(expiry)] = params

        # Score the *full* slice (incl. ITM, low-IV, etc.) against the fit.
        mask = work["expiry"] == expiry
        k_all = work.loc[mask, "log_moneyness"].to_numpy()
        work.loc[mask, "svi_iv"] = params.iv(k_all)

    # --- Step 3: residual + z-score ---
    work["residual_iv"] = work["iv"] - work["svi_iv"]
    spread_price = (work["ask"] - work["bid"]).clip(lower=0)
    spread_iv = spread_price / (2.0 * work["vega"].replace({0: np.nan}))
    spread_iv = spread_iv.replace([np.inf, -np.inf], np.nan)
    work["spread_iv"] = spread_iv
    work["zscore"] = work["residual_iv"] / spread_iv

    # Final scoring filter — drop contracts that shouldn't even be in
    # the ranking. We do this AFTER computing residuals so the columns
    # are still available for inspection (NaN'd out, not dropped from
    # the DataFrame), but they won't appear in `top()`.
    bad_mask = (
        (work["iv"].fillna(0) > score_max_iv) |
        (work["log_moneyness"].abs() > score_k_range)
    )
    for col in ("residual_iv", "zscore"):
        work.loc[bad_mask, col] = np.nan

    return MispricingReport(scored=work, fits=fits)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bs_vega_vec(df: pd.DataFrame, spot: float) -> np.ndarray:
    """Compute vega per contract via the vectorized C++ Greeks batch.

    Single FFI call instead of N — on a 5,000-contract chain this is
    the difference between ~30 seconds (Python loop) and a few
    milliseconds.
    """
    n = len(df)
    types = np.where(df["type"].to_numpy() == "C", 0, 1).astype(np.int8)
    greeks = bs_greeks_batch(
        types=types,
        S=np.full(n, spot, dtype=float),
        K=df["strike"].to_numpy(dtype=float),
        T=df["ttm"].to_numpy(dtype=float),
        r=df["r"].to_numpy(dtype=float),
        q=df["q"].to_numpy(dtype=float),
        sigma=df["iv"].to_numpy(dtype=float),
    )
    # Row 2 of the (5, N) array is vega; degenerate inputs already
    # came back as NaN from the batch helper.
    return greeks[2]


def _fit_subset(df: pd.DataFrame, side: str, min_iv: float, max_iv: float) -> pd.DataFrame:
    sub = df.dropna(subset=["iv", "log_moneyness"]).copy()
    sub = sub[(sub["iv"] >= min_iv) & (sub["iv"] <= max_iv)]
    if side == "OTM":
        otm_calls = (sub["type"] == "C") & (sub["log_moneyness"] >= 0)
        otm_puts  = (sub["type"] == "P") & (sub["log_moneyness"] <  0)
        sub = sub[otm_calls | otm_puts]
    elif side == "calls":
        sub = sub[sub["type"] == "C"]
    elif side == "puts":
        sub = sub[sub["type"] == "P"]
    elif side == "both":
        pass
    else:  # pragma: no cover
        raise ValueError(f"Unknown side {side!r}")
    return sub


def _spread_weights(df: pd.DataFrame) -> np.ndarray:
    """Higher weight = tighter market = trust this point more.

    We use ``1 / max(spread_pct, 0.01)``: caps at 100x weight for the
    tightest quotes. Flat 1.0 if the column isn't present.
    """
    if "spread_pct" not in df.columns:
        return np.ones(len(df))
    sp = df["spread_pct"].to_numpy()
    sp = np.where(np.isfinite(sp) & (sp > 0), sp, 0.5)
    return 1.0 / np.maximum(sp, 0.01)
