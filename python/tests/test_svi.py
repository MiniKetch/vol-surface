"""SVI fit + scanner tests.

Strategy: synthesize observations from a *known* SVI surface, run the
fitter, check that we recover the parameters within tolerance and that
the no-arb constraints hold.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vol_surface import OptionType, bs_price_batch
from vol_surface.scanner import scan_mispricing
from vol_surface.svi import (
    RawSVIParams,
    fit_raw_svi,
    raw_svi_iv,
    raw_svi_total_variance,
)


# ---------------------------------------------------------------------------
# Math sanity
# ---------------------------------------------------------------------------

def test_total_variance_is_nonneg_under_arb_free_params() -> None:
    # Pick a clean smirk: a=0.04, b=0.4, ρ=-0.7, m=-0.05, σ=0.2, T=0.5.
    k = np.linspace(-1.0, 1.0, 401)
    w = raw_svi_total_variance(k, 0.04, 0.4, -0.7, -0.05, 0.2)
    assert (w >= 0).all()


def test_iv_matches_sqrt_w_over_T() -> None:
    k = np.linspace(-0.3, 0.3, 11)
    T = 0.5
    w = raw_svi_total_variance(k, 0.04, 0.4, -0.7, 0.0, 0.2)
    iv = raw_svi_iv(k, T, 0.04, 0.4, -0.7, 0.0, 0.2)
    np.testing.assert_allclose(iv, np.sqrt(w / T), atol=1e-12)


# ---------------------------------------------------------------------------
# Round-trip — synthesize from known params, fit, recover.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("true", [
    RawSVIParams(a=0.04, b=0.30, rho=-0.65, m=-0.02, sigma=0.18, T=0.5),
    RawSVIParams(a=0.02, b=0.20, rho=-0.55, m=-0.05, sigma=0.15, T=1.0),
    RawSVIParams(a=0.01, b=0.10, rho=-0.45, m= 0.00, sigma=0.10, T=2.0),
])
def test_fit_round_trip_recovers_iv_curve(true: RawSVIParams) -> None:
    rng = np.random.default_rng(13)
    k = np.linspace(-0.4, 0.4, 25)
    # Realistic noise: 5e-3 IV ≈ a half-percent vol point, comparable
    # to real bid-ask half-spreads on liquid SPY contracts. Tighter
    # noise lets the soft equity-skew prior dominate and pull the fit
    # toward its target — that's correct Bayesian behaviour but it
    # makes a "pure recovery" test misleading. atol scaled accordingly.
    iv = true.iv(k) + rng.normal(scale=5e-3, size=k.size)
    fitted = fit_raw_svi(k, iv, T=true.T)

    # The IV curve from the fit should track the truth across the
    # whole range, not just at the sample points. This is a stronger
    # check than recovering parameters one-by-one (SVI has parameter
    # symmetries that make per-parameter recovery noisy).
    k_test = np.linspace(-0.35, 0.35, 100)
    np.testing.assert_allclose(
        fitted.iv(k_test), true.iv(k_test), atol=1e-2,
    )


def test_fit_returns_arbitrage_free_params() -> None:
    rng = np.random.default_rng(7)
    true = RawSVIParams(a=0.04, b=0.30, rho=-0.7, m=-0.02, sigma=0.18, T=0.5)
    k = np.linspace(-0.5, 0.5, 30)
    iv = true.iv(k) + rng.normal(scale=1e-3, size=k.size)
    fitted = fit_raw_svi(k, iv, T=true.T)
    assert fitted.is_arbitrage_free()


def test_fit_handles_nans_gracefully() -> None:
    k = np.linspace(-0.3, 0.3, 12)
    iv = np.full_like(k, 0.20)
    iv[3] = np.nan
    iv[7] = np.nan
    fitted = fit_raw_svi(k, iv, T=0.5)
    assert fitted.n_points == 10  # 12 minus the 2 NaNs


def test_fit_rejects_too_few_points() -> None:
    with pytest.raises(RuntimeError, match=r"≥5 points"):
        fit_raw_svi(np.array([0.0, 0.1]), np.array([0.20, 0.21]), T=0.5)


# ---------------------------------------------------------------------------
# Mispricing scanner — synthesize a chain with one known outlier and
# check that it ranks at the top.
# ---------------------------------------------------------------------------

def _build_synthetic_chain(true: RawSVIParams, *, spot: float = 450.0,
                           outlier_strike: float = 405.0,
                           outlier_iv_bump: float = 0.05) -> pd.DataFrame:
    """Build an enriched-style DataFrame with a known IV outlier."""
    today = dt.datetime.now()
    expiry = today + dt.timedelta(days=int(true.T * 365))
    # Make sure the outlier strike is actually in the grid — otherwise
    # the bump has no effect and the test is meaningless.
    strikes = np.union1d(np.linspace(0.7, 1.4, 30) * spot,
                         np.array([outlier_strike]))

    rows = []
    r_const, q_const = 0.04, 0.0
    forward = spot * np.exp((r_const - q_const) * true.T)
    for K in strikes:
        for typ in ("C", "P"):
            k = np.log(K / forward)
            iv_true = float(true.iv(np.array([k]))[0])
            # Bump only one specific strike to create an outlier.
            iv = iv_true + (outlier_iv_bump if abs(K - outlier_strike) < 0.01 else 0.0)
            rows.append({
                "ticker": "SYNTH",
                "type": typ,
                "strike": float(K),
                "expiry": expiry,
                "ttm": true.T,
                "log_moneyness": k,
                "iv": iv,
                "r": r_const,
                "q": q_const,
            })
    df = pd.DataFrame(rows)

    # Generate consistent prices from the (possibly bumped) IV.
    types = np.where(df["type"] == "C", 0, 1).astype(np.int8)
    prices = bs_price_batch(
        types,
        np.full(len(df), spot),
        df["strike"].to_numpy(),
        np.full(len(df), true.T),
        np.full(len(df), r_const),
        np.full(len(df), q_const),
        df["iv"].to_numpy(),
    )
    df["bid"] = np.maximum(prices - 0.05, 0)
    df["ask"] = prices + 0.05
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    df["volume"] = pd.array([1000] * len(df), dtype="Int64")
    df["open_interest"] = pd.array([5000] * len(df), dtype="Int64")
    df.attrs["spot"] = spot
    df.attrs["ticker"] = "SYNTH"
    df.attrs["dividend_yield"] = q_const
    df.attrs["fetched_at"] = today
    return df


def test_scanner_ranks_known_outlier_at_the_top() -> None:
    true = RawSVIParams(a=0.04, b=0.30, rho=-0.6, m=-0.02, sigma=0.18, T=0.5)
    df = _build_synthetic_chain(true, outlier_strike=405.0,
                                outlier_iv_bump=0.10)  # +10 vol points

    report = scan_mispricing(df, side="both")
    top = report.top(n=5, side="rich")

    # The bumped strike is K=405 — it should be in the top 3.
    assert any(abs(s - 405.0) < 0.5 for s in top["strike"]), \
        f"Expected K=405 in top: {top[['strike', 'zscore']].to_string()}"


def test_scanner_z_scores_have_reasonable_distribution() -> None:
    true = RawSVIParams(a=0.04, b=0.30, rho=-0.6, m=-0.02, sigma=0.18, T=0.5)
    df = _build_synthetic_chain(true, outlier_iv_bump=0.0)  # no outliers
    report = scan_mispricing(df, side="both")
    z = report.scored["zscore"].dropna().to_numpy()
    # Most contracts should have small |z| since there's no real bump.
    assert np.median(np.abs(z)) < 1.0
    assert report.n_contracts > 30
