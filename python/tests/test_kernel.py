"""Smoke + correctness tests for the Python bindings.

These are intentionally *not* a re-run of the C++ tests — those already
pin the math. These tests only check that the bindings round-trip
correctly across the FFI boundary and that the vectorized batch APIs
behave numpy-natively.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vol_surface import (
    Greeks,
    OptionType,
    bs_greeks,
    bs_price,
    bs_price_batch,
    implied_vol,
    implied_vol_batch,
)


def test_bs_price_atm_reference() -> None:
    p = bs_price(OptionType.Call, S=100, K=100, T=1, r=0.05, q=0.0, sigma=0.20)
    assert math.isclose(p, 10.45058357, rel_tol=0, abs_tol=1e-7)


def test_bs_greeks_returns_struct() -> None:
    g = bs_greeks(OptionType.Call, S=100, K=100, T=1, r=0.05, q=0.0, sigma=0.20)
    assert isinstance(g, Greeks)
    assert math.isclose(g.delta, 0.63683066, abs_tol=1e-7)
    # And it has a useful repr — handy when poking around in a notebook.
    assert "delta" in repr(g)


def test_implied_vol_round_trip() -> None:
    p = bs_price(OptionType.Call, S=100, K=100, T=1, r=0.05, q=0.0, sigma=0.20)
    iv = implied_vol(OptionType.Call, market_price=p,
                     S=100, K=100, T=1, r=0.05, q=0.0)
    assert iv is not None
    assert math.isclose(iv, 0.20, abs_tol=1e-6)


def test_implied_vol_returns_none_on_failure() -> None:
    # Sub-intrinsic price → no IV exists.
    iv = implied_vol(OptionType.Call, market_price=1.0,
                     S=110, K=100, T=1, r=0.05, q=0.0)
    assert iv is None


@pytest.mark.parametrize("n", [1, 100, 10_000])
def test_batch_round_trip_consistency(n: int) -> None:
    rng = np.random.default_rng(seed=42)
    types = rng.integers(0, 2, size=n, dtype=np.int8)
    S = np.full(n, 100.0)
    K = rng.uniform(70, 130, size=n)
    T = rng.uniform(0.05, 2.0, size=n)
    r = np.full(n, 0.04)
    q = np.full(n, 0.01)
    sigma = rng.uniform(0.10, 0.60, size=n)

    prices = bs_price_batch(types, S, K, T, r, q, sigma)
    ivs = implied_vol_batch(types, prices, S, K, T, r, q)

    # Skip any pathological NaN — the C++ tests already vet which combos
    # are FP-resolvable. We use a slightly looser atol than the C++
    # round-trip test (1e-6) because random draws will occasionally
    # land near the vega-resolution edge; the C++ tests guarantee the
    # solver is correct on well-conditioned inputs.
    good = ~np.isnan(ivs)
    assert good.sum() > 0.8 * n  # at least 80 % should resolve
    assert np.allclose(ivs[good], sigma[good], atol=1e-4)


def test_batch_mismatched_lengths_raises() -> None:
    types = np.array([0], dtype=np.int8)
    with pytest.raises(ValueError):
        bs_price_batch(types,
                       np.array([100.0]),
                       np.array([100.0, 100.0]),  # wrong length
                       np.array([1.0]),
                       np.array([0.05]),
                       np.array([0.0]),
                       np.array([0.20]))
