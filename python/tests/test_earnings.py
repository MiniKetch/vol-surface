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


# ---------------------------------------------------------------------------
# Regression: fetch_earnings now picks the *next* future date, not the
# furthest-out one, when falling back to earnings_dates.
# ---------------------------------------------------------------------------

def test_fetch_earnings_history_picks_nearest_future_date(monkeypatch) -> None:
    """The original implementation indexed `[-1]` (last) on a
    chronologically-sorted future series, which gave the *furthest*
    upcoming earnings date instead of the *next* one. After the fix
    it should pick `[0]` (the nearest future date)."""
    import pandas as pd
    from vol_surface.data import earnings as earnings_module

    # Build a fake yfinance Ticker with .calendar=None and an
    # .earnings_dates Series spanning past + future dates.
    class FakeTicker:
        calendar = None
        earnings_dates = pd.Series(
            [None, None, None, None],
            index=pd.DatetimeIndex([
                "2025-01-15", "2025-04-30",   # past
                "2026-08-01", "2027-02-15",   # future
            ]),
        )

    class FakeYFModule:
        Ticker = lambda self, t: FakeTicker()
    fake_yf = FakeYFModule()

    # Patch the import inside fetch_earnings.
    import sys
    sys.modules["yfinance"] = fake_yf
    try:
        # Also need pd.Timestamp.now to be deterministic so 'future' is
        # well-defined relative to the test.
        info = earnings_module.fetch_earnings("FAKE")
    finally:
        del sys.modules["yfinance"]

    # Should pick the *nearest* future date = 2026-08-01, NOT 2027-02-15.
    assert info.next_date == dt.date(2026, 8, 1), \
        f"expected nearest future earnings, got {info.next_date}"
    assert info.source == "history"
