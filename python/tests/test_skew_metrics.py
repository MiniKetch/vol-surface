"""Tests for the skew metrics (RR, BF) computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_surface import OptionType, strike_at_delta
from vol_surface.analytics import compute_skew_metrics
from vol_surface.svi import RawSVIParams


def test_strike_at_delta_smoke() -> None:
    """Quick check the C++ binding works through Python."""
    K = strike_at_delta(OptionType.Call, 0.25,
                        S=100.0, T=0.5, r=0.04, q=0.0, sigma=0.20)
    assert K is not None
    # 25-delta call sits above forward.
    assert K > 100


def test_compute_skew_metrics_equity_skew() -> None:
    """An SVI fit with strong negative skew should yield negative
    25-delta RR (puts richer than calls)."""
    fits = {
        pd.Timestamp("2026-08-15"): RawSVIParams(
            a=0.04, b=0.30, rho=-0.7, m=-0.02, sigma=0.18, T=0.5
        ),
        pd.Timestamp("2027-02-19"): RawSVIParams(
            a=0.05, b=0.25, rho=-0.5, m=-0.02, sigma=0.20, T=1.0
        ),
    }
    df = compute_skew_metrics(fits, spot=100.0, risk_free=0.04, dividend_yield=0.01)
    assert len(df) == 2
    assert (df["rr_25"] < 0).all(), "Equity-style: RR should be negative"
    assert (df["bf_25"] > 0).all(), "Healthy smile: BF should be positive"
    # Matures sorted ascending.
    assert df["ttm"].is_monotonic_increasing


def test_compute_skew_metrics_empty_input() -> None:
    df = compute_skew_metrics({}, spot=100.0)
    assert df.empty
