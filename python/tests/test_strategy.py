"""Tests for the strategy parser + payoff computation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vol_surface import OptionType, bs_price_batch
from vol_surface.strategy import compute_payoff, parse_strategy


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_single_long_call() -> None:
    s = parse_strategy("long 10 SPY 740C 2026-06-18")
    assert len(s.legs) == 1
    leg = s.legs[0]
    assert leg.side == "long"
    assert leg.quantity == 10
    assert leg.option_type == "C"
    assert leg.strike == 740
    assert leg.expiry == pd.Timestamp("2026-06-18")
    assert leg.signed_qty == 10


def test_parse_short_put() -> None:
    s = parse_strategy("short 5 700P 2026-06-18")
    assert s.legs[0].side == "short"
    assert s.legs[0].option_type == "P"
    assert s.legs[0].signed_qty == -5


def test_parse_default_long_when_side_omitted() -> None:
    s = parse_strategy("1 740C 2026-06-18")
    assert s.legs[0].side == "long"


def test_parse_call_spread_two_legs() -> None:
    s = parse_strategy("long 1 730C 2026-06-18, short 1 740C 2026-06-18")
    assert len(s.legs) == 2
    assert s.legs[0].strike == 730 and s.legs[0].signed_qty == +1
    assert s.legs[1].strike == 740 and s.legs[1].signed_qty == -1


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="Couldn't parse leg"):
        parse_strategy("buy a banana")


def test_parse_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="Empty strategy"):
        parse_strategy("")


# ---------------------------------------------------------------------------
# Payoff math
# ---------------------------------------------------------------------------

def _synthetic_chain():
    """Tiny enriched-chain DataFrame for payoff tests."""
    today = pd.Timestamp.now().normalize()
    expiry = today + pd.Timedelta(days=30)
    spot = 100.0

    rows = []
    for K in [80, 90, 100, 110, 120]:
        for typ in ("C", "P"):
            rows.append({
                "ticker": "SYN", "type": typ, "strike": K, "expiry": expiry,
                "ttm": 30/365.0, "iv": 0.25,
                "bid": 1.0, "ask": 1.1, "mid": 1.05,
                "spread_pct": 0.1, "volume": 100, "open_interest": 100,
                "r": 0.04, "q": 0.0,
                "moneyness": K / spot,
                "log_moneyness": np.log(K / spot),
            })
    df = pd.DataFrame(rows)
    df.attrs["spot"] = spot
    df.attrs["ticker"] = "SYN"
    df.attrs["fetched_at"] = today
    return df


def test_long_call_payoff_is_max_S_minus_K_zero() -> None:
    df = _synthetic_chain()
    expiry = df["expiry"].iloc[0]
    s = parse_strategy(f"long 1 100C {expiry.date()}")
    pnl = compute_payoff(s, df, spot_lo=80, spot_hi=120, n_grid=21)

    # At spot=100, payoff is 0; at spot=120, payoff is 20.
    assert pnl.loc[pnl["spot"] == 100, "payoff"].iloc[0] == pytest.approx(0)
    assert pnl.loc[pnl["spot"] == 120, "payoff"].iloc[0] == pytest.approx(20)
    # Below strike, zero payoff.
    assert (pnl[pnl["spot"] < 100]["payoff"] == 0).all()


def test_call_spread_caps_payoff() -> None:
    df = _synthetic_chain()
    expiry = df["expiry"].iloc[0]
    s = parse_strategy(
        f"long 1 100C {expiry.date()}, short 1 110C {expiry.date()}"
    )
    pnl = compute_payoff(s, df, spot_lo=80, spot_hi=130, n_grid=51)

    # Below 100: zero payoff.
    assert pnl[pnl["spot"] < 100]["payoff"].max() == pytest.approx(0)
    # Above 110: capped at 10.
    high = pnl[pnl["spot"] > 110]["payoff"]
    assert high.min() == pytest.approx(10)
    assert high.max() == pytest.approx(10)


def test_mtm_curve_above_payoff_outside_breakeven() -> None:
    """Time-value should make the MTM curve above the expiry payoff
    everywhere except deep ITM where they converge."""
    df = _synthetic_chain()
    expiry = df["expiry"].iloc[0]
    s = parse_strategy(f"long 1 100C {expiry.date()}")
    pnl = compute_payoff(s, df, spot_lo=80, spot_hi=120, n_grid=21)
    # Around ATM, MTM should exceed expiry payoff (positive time value).
    atm_row = pnl[pnl["spot"].between(95, 105)]
    assert (atm_row["mtm"] >= atm_row["payoff"] - 1e-9).all()
