"""Tests for the shared utils module. Most of these are tiny, but
they pin down the behaviour the rest of the codebase relies on —
forward-price arithmetic, side filtering, attrs survival, IV-spread
conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_surface.utils import (
    attach_attrs,
    extract_attrs,
    filter_by_side,
    forward,
    get_attr,
    iv_spread_from_price,
    log_moneyness,
)


# ---------------------------------------------------------------------------
# Forward / log-moneyness
# ---------------------------------------------------------------------------

def test_forward_no_carry_is_spot() -> None:
    assert forward(100.0, 0.0, 0.0, 1.0) == pytest.approx(100.0)


def test_forward_compounds_with_carry() -> None:
    F = forward(100.0, 0.05, 0.02, 1.0)
    assert F == pytest.approx(100.0 * np.exp(0.03), rel=1e-12)


def test_forward_vectorized() -> None:
    S = np.array([100.0, 200.0, 50.0])
    F = forward(S, 0.04, 0.01, 0.5)
    assert F.shape == (3,)
    np.testing.assert_allclose(F, S * np.exp(0.015))


def test_log_moneyness_at_forward_is_zero() -> None:
    F = forward(100.0, 0.05, 0.0, 1.0)
    assert log_moneyness(F, 100.0, 0.05, 0.0, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_log_moneyness_otm_call_positive() -> None:
    k = log_moneyness(110.0, 100.0, 0.0, 0.0, 1.0)
    assert k > 0  # K > F means OTM call


def test_log_moneyness_otm_put_negative() -> None:
    k = log_moneyness(90.0, 100.0, 0.0, 0.0, 1.0)
    assert k < 0  # K < F means OTM put


# ---------------------------------------------------------------------------
# Side filter
# ---------------------------------------------------------------------------

def _toy_chain() -> pd.DataFrame:
    return pd.DataFrame({
        "type":          ["C", "C", "C", "P", "P", "P"],
        "log_moneyness": [-0.1, 0.0, 0.1, -0.1, 0.0, 0.1],
        "strike":        [90, 100, 110, 90, 100, 110],
    })


def test_filter_by_side_OTM_keeps_otm_wings_only() -> None:
    out = filter_by_side(_toy_chain(), "OTM")
    # OTM calls: k >= 0 → call rows at k=0 and k=0.1 → strikes 100, 110
    # OTM puts:  k <  0 → put row at k=-0.1                 → strike 90
    assert sorted(out["strike"].tolist()) == [90, 100, 110]
    # Type/k consistency: every kept row is on the OTM side.
    for _, row in out.iterrows():
        if row["type"] == "C":
            assert row["log_moneyness"] >= 0
        else:
            assert row["log_moneyness"] < 0


def test_filter_by_side_calls_only() -> None:
    out = filter_by_side(_toy_chain(), "calls")
    assert (out["type"] == "C").all()
    assert len(out) == 3


def test_filter_by_side_puts_only() -> None:
    out = filter_by_side(_toy_chain(), "puts")
    assert (out["type"] == "P").all()


def test_filter_by_side_both_returns_full_chain() -> None:
    chain = _toy_chain()
    out = filter_by_side(chain, "both")
    assert len(out) == len(chain)


def test_filter_by_side_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown side"):
        filter_by_side(_toy_chain(), "weird")


# ---------------------------------------------------------------------------
# IV-spread conversion
# ---------------------------------------------------------------------------

def test_iv_spread_from_price_normal() -> None:
    # spread of 1¢ on a contract with vega 50 → IV-half-spread of 1e-4
    out = iv_spread_from_price(np.array([0.01]), np.array([50.0]))
    np.testing.assert_allclose(out, [0.0001])


def test_iv_spread_from_price_zero_vega_returns_nan() -> None:
    out = iv_spread_from_price(np.array([0.05]), np.array([0.0]))
    assert np.isnan(out[0])


def test_iv_spread_from_price_negative_inputs_become_nan() -> None:
    # Negative vega is degenerate.
    out = iv_spread_from_price(np.array([0.05]), np.array([-1.0]))
    assert np.isnan(out[0])


# ---------------------------------------------------------------------------
# Attrs survival helpers
# ---------------------------------------------------------------------------

def test_extract_attrs_pulls_canonical_keys() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    df.attrs["ticker"] = "SPY"
    df.attrs["spot"] = 724.5
    df.attrs["random_unrelated_key"] = "ignore me"
    out = extract_attrs(df)
    assert out["ticker"] == "SPY"
    assert out["spot"] == 724.5
    assert "random_unrelated_key" not in out


def test_attach_attrs_round_trip() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    df.attrs.update({"ticker": "AAPL", "spot": 285.0})

    # Simulate Streamlit's pickle-roundtrip-strips-attrs pattern.
    attrs = extract_attrs(df)
    df_after_pickle = pd.DataFrame({"x": [1, 2]})  # no .attrs
    attach_attrs(df_after_pickle, attrs)

    assert df_after_pickle.attrs["ticker"] == "AAPL"
    assert df_after_pickle.attrs["spot"] == 285.0


def test_get_attr_with_default() -> None:
    df = pd.DataFrame({"x": [1]})
    assert get_attr(df, "ticker", "fallback") == "fallback"
    df.attrs["ticker"] = "QQQ"
    assert get_attr(df, "ticker") == "QQQ"
