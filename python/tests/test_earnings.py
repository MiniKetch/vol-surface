"""Tests for the earnings calendar helper.

We don't hit yfinance — that's flaky for unit tests. Instead test the
flagging logic which is the bit that actually has decisions in it."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from vol_surface.data import flag_event_expiries


def test_flag_no_earnings_means_all_false() -> None:
    expiries = [pd.Timestamp("2026-06-18"), pd.Timestamp("2026-07-17")]
    out = flag_event_expiries(expiries, None)
    assert all(v is False for v in out.values())


def test_flag_expiry_after_earnings_within_window() -> None:
    earnings = dt.date(2026, 6, 10)
    expiries = [
        pd.Timestamp("2026-06-05"),  # before earnings → False
        pd.Timestamp("2026-06-12"),  # 2 days after → True
        pd.Timestamp("2026-07-10"),  # 30 days after → True
        pd.Timestamp("2026-08-15"),  # 66 days after → False (>30)
    ]
    out = flag_event_expiries(expiries, earnings)
    assert out[expiries[0]] is False
    assert out[expiries[1]] is True
    assert out[expiries[2]] is True
    assert out[expiries[3]] is False


def test_flag_custom_window() -> None:
    earnings = dt.date(2026, 6, 10)
    expiries = [pd.Timestamp("2026-08-01")]   # 52 days after
    out = flag_event_expiries(expiries, earnings, window_days=60)
    assert out[expiries[0]] is True
