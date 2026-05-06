"""Next earnings date lookup, plus a helper that flags which expiries
straddle the announcement.

yfinance exposes a ``Ticker.calendar`` field (and historically the
``earnings_dates`` series) that returns the next scheduled earnings
date when one is announced. We treat any expiry whose date is on or
after the earnings date — but within 30 calendar days of it — as
"event-containing." The market typically marks up event-vol on the
nearest expiry past the announcement.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd


@dataclass
class EarningsInfo:
    """Lightweight result. ``next_date`` may be None if not announced."""
    ticker: str
    next_date: Optional[dt.date]
    source: str  # 'calendar', 'history', or 'unavailable'


def fetch_earnings(ticker: str) -> EarningsInfo:
    """Best-effort next-earnings lookup. Returns ``next_date=None`` if
    the field isn't available — yfinance metadata is sometimes empty
    for ETFs or non-US-listed names, and that's fine."""
    try:
        import yfinance as yf  # noqa: WPS433
    except ImportError:
        return EarningsInfo(ticker, None, "unavailable")

    try:
        tk = yf.Ticker(ticker)
    except Exception:  # noqa: BLE001
        return EarningsInfo(ticker, None, "unavailable")

    # Try Ticker.calendar first — it's the modern field. Different
    # yfinance versions return it as either a DataFrame or a dict.
    cal = getattr(tk, "calendar", None)
    next_date = _extract_calendar_date(cal)
    if next_date is not None:
        return EarningsInfo(ticker, next_date, "calendar")

    # Fall back to earnings_dates (a Series indexed by date).
    try:
        ed = tk.earnings_dates
    except (AttributeError, ValueError, KeyError):
        ed = None
    if ed is not None and len(ed) > 0:
        try:
            today = pd.Timestamp.now(tz=ed.index.tz) if ed.index.tz else pd.Timestamp.now()
            future = ed[ed.index >= today].sort_index()
            if len(future) > 0:
                # The *next* earnings date is the earliest future
                # observation, not the latest. The original code used
                # [-1] which silently picked the furthest-out date and
                # caused expiry flagging to mark wrong expiries.
                return EarningsInfo(ticker, future.index[0].date(), "history")
        except (AttributeError, ValueError, TypeError, KeyError):
            pass

    return EarningsInfo(ticker, None, "unavailable")


def flag_event_expiries(
    expiries: List[pd.Timestamp],
    earnings_date: Optional[dt.date],
    *,
    window_days: int = 30,
) -> dict:
    """Return ``{expiry: is_event}`` mapping.

    ``is_event`` is True for any expiry on or after the earnings date
    and within ``window_days`` of it — the standard "this expiry
    contains the event" rule.
    """
    if earnings_date is None:
        return {pd.Timestamp(e): False for e in expiries}
    flagged = {}
    for e in expiries:
        e_date = pd.Timestamp(e).date()
        in_window = (e_date >= earnings_date and
                     (e_date - earnings_date).days <= window_days)
        flagged[pd.Timestamp(e)] = in_window
    return flagged


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _extract_calendar_date(cal) -> Optional[dt.date]:
    """yfinance's ``.calendar`` shape varies across versions:
    * older: pandas DataFrame with row label "Earnings Date" and one
      or two timestamp columns;
    * newer: dict with key "Earnings Date" mapping to a list of
      one or two datetimes (start/end of event window).
    Return the first valid date, else None.
    """
    if cal is None:
        return None
    try:
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed is None:
                return None
            if isinstance(ed, (list, tuple)) and ed:
                v = ed[0]
            else:
                v = ed
            return pd.Timestamp(v).date()
        # DataFrame fallback.
        if "Earnings Date" in cal.index:
            row = cal.loc["Earnings Date"]
            v = row.iloc[0] if hasattr(row, "iloc") else row[0]
            return pd.Timestamp(v).date()
    except Exception:  # noqa: BLE001
        return None
    return None
