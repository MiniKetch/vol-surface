"""Risk-free rate provider.

Two sources, in priority order:

1. **Live FRED** — set the ``FREDAPI_KEY`` environment variable. We pull
   the standard Treasury constant-maturity series and use the most
   recent observation. Requires a (free) FRED API key from
   https://fred.stlouisfed.org/docs/api/api_key.html.

2. **Bundled snapshot** — a recent Treasury yield curve shipped in the
   repo at ``vol_surface/data/snapshots/treasury_curve.csv``. No
   internet or API key needed; refreshed periodically.

The snapshot keeps the project working out-of-the-box for anyone who
just wants to run the demo. The FRED path is for users who want
current rates for live scanning.

Rates are stored as **continuously-compounded** decimals (e.g. 0.0412
for 4.12 %). The CSV stores the same. Treasury yields are technically
bond-equivalent yields (semi-annual coupon), so the conversion is
``r_cc ≈ ln(1 + y_bey)`` — for short tenors at 4–5 % this is a sub-bp
approximation, more than fine for IV work.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

# FRED constant-maturity Treasury series IDs, keyed by maturity in years.
# Source: https://fred.stlouisfed.org/categories/115 (constant-maturity).
_FRED_SERIES = {
    1 / 12: "DGS1MO",
    2 / 12: "DGS2MO",
    3 / 12: "DGS3MO",
    6 / 12: "DGS6MO",
    1.0:    "DGS1",
    2.0:    "DGS2",
    3.0:    "DGS3",
    5.0:    "DGS5",
    7.0:    "DGS7",
    10.0:   "DGS10",
    20.0:   "DGS20",
    30.0:   "DGS30",
}


class RiskFreeCurve:
    """Continuously-compounded zero-coupon rates by tenor.

    Linearly interpolates in tenor (years to expiry) between observed
    points. Flat-extrapolates outside the range.

    Parameters
    ----------
    source
        ``"auto"`` (default) tries FRED if ``FREDAPI_KEY`` is set, else
        falls back to the bundled snapshot. ``"snapshot"`` forces the
        bundled file. ``"fred"`` requires a key and raises if not set.
    snapshot_path
        Override the path to the snapshot CSV (mostly for tests).

    Examples
    --------
    >>> curve = RiskFreeCurve()                  # auto-pick source
    >>> curve.rate(0.5)                          # 6-month rate, scalar
    0.041
    >>> import numpy as np
    >>> curve.rate(np.array([0.25, 1.0, 5.0]))   # vectorized
    array([0.0408, 0.0412, 0.0438])
    """

    def __init__(
        self,
        source: Literal["auto", "snapshot", "fred"] = "auto",
        snapshot_path: str | Path | None = None,
    ) -> None:
        self._source: str
        if source == "auto":
            if os.environ.get("FREDAPI_KEY"):
                tenors, rates = _load_from_fred()
                self._source = "fred"
            else:
                tenors, rates = _load_from_snapshot(snapshot_path)
                self._source = "snapshot"
        elif source == "snapshot":
            tenors, rates = _load_from_snapshot(snapshot_path)
            self._source = "snapshot"
        elif source == "fred":
            tenors, rates = _load_from_fred()
            self._source = "fred"
        else:  # pragma: no cover
            raise ValueError(f"Unknown source: {source!r}")

        # Sort by tenor so np.interp behaves predictably.
        order = np.argsort(tenors)
        self._tenors = np.asarray(tenors)[order]
        self._rates  = np.asarray(rates)[order]

    @property
    def source(self) -> str:
        """Which data source we ended up using ('fred' or 'snapshot')."""
        return self._source

    def rate(self, T):
        """Interpolated continuously-compounded zero rate for tenor(s) ``T``.

        ``T`` may be a scalar or array-like (in years). Outside the
        observed tenor range, extrapolates flat (most-extreme value).
        """
        T_arr = np.asarray(T, dtype=float)
        out = np.interp(T_arr, self._tenors, self._rates)
        if T_arr.ndim == 0:
            return float(out)
        return out

    def __repr__(self) -> str:
        return (
            f"<RiskFreeCurve source={self._source!r} "
            f"tenors={list(self._tenors)} "
            f"rates={[round(r, 5) for r in self._rates]}>"
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_from_snapshot(path: str | Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Load the bundled Treasury curve CSV."""
    if path is None:
        # `resources.files` works whether we're running from source or from
        # an installed wheel — that's the modern resource-loading pattern.
        snapshot = (resources.files("vol_surface.data.snapshots")
                    / "treasury_curve.csv")
        with resources.as_file(snapshot) as p:
            df = pd.read_csv(p)
    else:
        df = pd.read_csv(path)

    # Use the most-recent observation date.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        latest = df["date"].max()
        df = df.loc[df["date"] == latest]

    if "maturity_years" not in df.columns or "rate" not in df.columns:
        raise ValueError(
            "Treasury snapshot must have columns: date, maturity_years, rate")
    return df["maturity_years"].to_numpy(), df["rate"].to_numpy()


def _load_from_fred() -> tuple[np.ndarray, np.ndarray]:
    """Pull current Treasury constant-maturity yields from FRED.

    Each series is in *percent*, so we divide by 100. The most recent
    non-NaN observation is used per series.
    """
    api_key = os.environ.get("FREDAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "FREDAPI_KEY env var is not set. Either set it (free key from "
            "https://fred.stlouisfed.org/docs/api/api_key.html) or use "
            "RiskFreeCurve(source='snapshot').")

    try:
        from fredapi import Fred  # noqa: WPS433  (optional dep)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "fredapi not installed. `pip install vol-surface[data]` to add it."
        ) from exc

    fred = Fred(api_key=api_key)
    tenors: list[float] = []
    rates:  list[float] = []
    for tenor, series_id in _FRED_SERIES.items():
        try:
            series = fred.get_series(series_id)
        except Exception:
            # Some series occasionally have no recent print — skip.
            continue
        latest = series.dropna()
        if latest.empty:
            continue
        tenors.append(float(tenor))
        # FRED returns percent; we want decimal. BEY → continuous via
        # ln(1+y). Difference is negligible for short tenors at 4–5 %.
        y = float(latest.iloc[-1]) / 100.0
        rates.append(float(np.log1p(y)))
    if not tenors:
        raise RuntimeError("FRED returned no usable Treasury data.")
    return np.array(tenors), np.array(rates)
