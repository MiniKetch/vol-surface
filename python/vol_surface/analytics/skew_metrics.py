"""Standard delta-anchored skew metrics.

Two industry-standard summaries reduce a whole smile to two numbers:

* **25-delta risk reversal (RR)** = IV(25Δ call) − IV(25Δ put).
  Negative for equity index (puts richer than calls). Used to gauge
  *directional* skew — how much the market is willing to pay for
  downside protection over upside.

* **25-delta butterfly (BF)** = ½ · [IV(25Δ call) + IV(25Δ put)]
  − IV(ATM). Always positive in healthy markets. Measures *convexity*
  of the smile — how much extra vol the wings carry over the body.

The point of quoting at fixed deltas (instead of fixed strikes) is
that delta normalises across underlyings, vol regimes, and maturities:
a 25-delta strike in low vol is closer to ATM than in high vol, and
that's what we want for cross-section comparison.

Strike-at-delta inversion uses the C++ kernel — Brent under the hood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from vol_surface import OptionType, strike_at_delta


@dataclass
class SkewMetrics:
    """One row per expiry."""
    expiry:   pd.Timestamp
    ttm:      float
    atm_iv:   float
    iv_25d_call: float
    iv_25d_put:  float
    iv_10d_call: float
    iv_10d_put:  float
    rr_25:    float            # IV(25Δc) − IV(25Δp). Negative = put skew.
    bf_25:    float            # ½(call+put) − ATM. Convexity.
    rr_10:    float
    bf_10:    float


def compute_skew_metrics(
    fits: Mapping[pd.Timestamp, "object"],
    *,
    spot: float,
    risk_free: float | None = None,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Compute 25-delta and 10-delta RR / BF for each fitted expiry.

    Parameters
    ----------
    fits
        ``{expiry: RawSVIParams}`` — typically ``MispricingReport.fits``.
    spot
        Current spot price.
    risk_free
        Constant risk-free rate. If None we'll use 0.04 as a fallback;
        the answer is barely sensitive to this for delta inversion.
    dividend_yield
        Continuous dividend yield. Defaults to 0.0.

    Returns
    -------
    DataFrame, one row per expiry, sorted by maturity.
    """
    if not fits:
        return pd.DataFrame()

    r = 0.04 if risk_free is None else float(risk_free)
    q = float(dividend_yield)

    rows: list[SkewMetrics] = []
    for expiry, params in fits.items():
        T = float(params.T)
        atm_iv = float(params.iv(np.array([0.0]))[0])

        c25 = _iv_at_delta(params, OptionType.Call,  0.25, spot, T, r, q, atm_iv)
        p25 = _iv_at_delta(params, OptionType.Put,  -0.25, spot, T, r, q, atm_iv)
        c10 = _iv_at_delta(params, OptionType.Call,  0.10, spot, T, r, q, atm_iv)
        p10 = _iv_at_delta(params, OptionType.Put,  -0.10, spot, T, r, q, atm_iv)

        rows.append(SkewMetrics(
            expiry=pd.Timestamp(expiry),
            ttm=T,
            atm_iv=atm_iv,
            iv_25d_call=c25,
            iv_25d_put=p25,
            iv_10d_call=c10,
            iv_10d_put=p10,
            rr_25 = c25 - p25,
            bf_25 = 0.5 * (c25 + p25) - atm_iv,
            rr_10 = c10 - p10,
            bf_10 = 0.5 * (c10 + p10) - atm_iv,
        ))

    df = pd.DataFrame([m.__dict__ for m in rows]).sort_values("ttm")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal: solve "what's the IV at this delta level on this SVI smile?"
# ---------------------------------------------------------------------------

def _iv_at_delta(
    params, opt_type: OptionType, target_delta: float,
    S: float, T: float, r: float, q: float, sigma_seed: float,
    max_iters: int = 4,
) -> float:
    """Iterate strike → IV → strike until the strike consistent with
    ``target_delta`` and the SVI-implied IV at that strike converge.

    Convergence is fast in practice (3–4 iterations gets you to <1e-4
    on σ) because each fixed-point step uses the C++ Brent strike
    finder. We start from ``sigma_seed`` (typically ATM IV) and refine.
    """
    sigma = max(sigma_seed, 0.05)
    last_K = None
    for _ in range(max_iters):
        K_opt = strike_at_delta(opt_type, target_delta,
                                S=S, T=T, r=r, q=q, sigma=sigma)
        if K_opt is None:
            return float("nan")
        K = float(K_opt)
        # Convert to log-moneyness against the forward and ask SVI
        # for the IV at that point.
        F = S * np.exp((r - q) * T)
        k = np.log(K / F)
        new_sigma = float(params.iv(np.array([k]))[0])
        if last_K is not None and abs(K - last_K) < 1e-6:
            sigma = new_sigma
            break
        last_K = K
        sigma = new_sigma
    return sigma
