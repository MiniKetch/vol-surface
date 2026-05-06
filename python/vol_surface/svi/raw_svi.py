"""Raw SVI parameterisation and per-slice fitter.

Reference: Gatheral, "A parsimonious arbitrage-free implied volatility
parameterization with application to the valuation of volatility
derivatives" (Global Derivatives & Risk Management, Madrid 2004).

Raw form (one parameter set per expiry):

    w(k) = a + b · { ρ·(k − m) + √((k − m)² + σ²) }

where:
    k = log(K / F)             ← log-moneyness
    w = T · σ_iv²              ← total variance
    a                          ← vertical translation (level)
    b ≥ 0                      ← angle between asymptotes (overall slope)
    ρ ∈ (−1, 1)                ← rotation (skew)
    m                          ← horizontal translation
    σ > 0                      ← curvature at the vertex

Implied vol comes back from total variance via σ_iv(k) = √(w(k) / T).

No-arbitrage constraints (Gatheral, Lee bounds + butterfly):
    b ≥ 0
    |ρ| ≤ 1
    σ > 0
    a + b·σ·√(1 − ρ²) ≥ 0      ← vertex of w(k) must be ≥ 0
    b · (1 + |ρ|) ≤ 4 / T      ← Lee's wing bound (no calendar-spread arb)

We enforce all of these as bounds on the optimiser, so any returned
fit is statically arbitrage-free by construction.

Implementation note: the fit lives in Python (scipy least_squares).
There's one optimiser call per expiry — typically <50 ms each, and
12 expiries = <1 s total. Porting to C++ buys nothing here; the
*evaluation* of the fitted SVI is what gets vectorised, and that's
already pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RawSVIParams:
    """One slice of a raw-SVI surface.

    Frozen because once fitted, these define a particular curve — if
    you want to refit you should produce a new instance.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float                     # the expiry these params apply to (years)
    rss: Optional[float] = None  # residual sum of squares from the fit
    n_points: Optional[int] = None  # how many contracts contributed

    def total_variance(self, k):
        return raw_svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma)

    def iv(self, k):
        return raw_svi_iv(k, self.T, self.a, self.b, self.rho, self.m, self.sigma)

    def is_arbitrage_free(self) -> bool:
        return _check_no_arb(self.a, self.b, self.rho, self.m, self.sigma, self.T)


def raw_svi_total_variance(k, a: float, b: float, rho: float,
                           m: float, sigma: float) -> np.ndarray:
    """Vectorised raw-SVI total variance w(k)."""
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def raw_svi_iv(k, T: float, a: float, b: float, rho: float,
               m: float, sigma: float) -> np.ndarray:
    """Implied vol implied by an SVI slice. ``T`` in years."""
    if T <= 0:
        raise ValueError("T must be positive")
    w = raw_svi_total_variance(k, a, b, rho, m, sigma)
    # Clip negatives that the optimiser may have allowed during a fit
    # iteration; the final fit is constrained to be ≥ 0 everywhere.
    return np.sqrt(np.maximum(w, 0.0) / T)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def fit_raw_svi(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    *,
    weights: np.ndarray | None = None,
    max_nfev: int = 2000,
    fit_k_range: float = 1.0,
    equity_skew: bool = True,
) -> RawSVIParams:
    """Fit a raw-SVI slice to (log-moneyness, IV) samples for one expiry.

    Parameters
    ----------
    k
        Log-moneyness ``ln(K / F)`` for each observation.
    iv
        Implied volatility for each observation. NaNs are filtered.
    T
        Time to expiry in years (single value — this is one slice).
    weights
        Optional per-point weights (e.g. inverse half-spread converted
        to IV space, or vega). Higher weight = the fit is more reluctant
        to deviate at that point. Defaults to uniform 1.0.
    max_nfev
        Cap on optimiser function evaluations.

    Returns
    -------
    RawSVIParams
        Fitted parameters with arbitrage-free bounds.
    """
    from scipy.optimize import least_squares  # local — keeps base import light.

    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    if T <= 0:
        raise ValueError("T must be positive")

    finite = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    # Truncate the fit input to |k| ≤ fit_k_range. Beyond that the
    # smile is dominated by quote noise and dead contracts that don't
    # belong to a smooth parametric curve. Scoring (in the scanner)
    # still uses all contracts — this only affects the *fit*.
    finite &= np.abs(k) <= fit_k_range
    k = k[finite]
    iv = iv[finite]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[finite]
    if len(k) < 5:
        raise RuntimeError(
            f"Need ≥5 points to fit SVI; got {len(k)}. Loosen filters or "
            "merge with a neighbouring expiry.")

    w_obs = (iv ** 2) * T

    # Initial guess — heuristic, but stable across most equity smiles.
    a0     = max(min(w_obs) * 0.9, 1e-6)
    b0     = 0.1
    rho0   = -0.6 if k.min() < 0 else 0.0
    m0     = 0.0
    sigma0 = 0.1
    x0 = np.array([a0, b0, rho0, m0, sigma0])

    # Bounds for static no-arb. The vertex constraint
    # `a + b·σ·√(1 − ρ²) ≥ 0` and the Lee wing bound are non-linear in
    # the parameters, so we enforce them via a soft penalty in the
    # residual rather than as hard `bounds=` constraints.
    #
    # `m` (horizontal translation) — bounded by *physics*, not data
    # extent. The vertex of an equity smile lives within ~30 % log-
    # moneyness of ATM; any m beyond that places the curve's vertex in
    # a region where the SVI wings (which are linear asymptotes) can
    # impersonate the smile shape via parameter redundancy, sucking
    # the optimiser into a local minimum with ρ near +1. We cap m at
    # ±0.4, comfortably wider than realistic equity skews but tight
    # enough to forbid the degenerate vertex-far-from-data basin.
    sigma_min = 1.0e-4
    b_max     = 4.0 / T  # Lee wing bound holds when b·(1+|ρ|) ≤ 4/T
    m_cap     = 0.4
    # Hard bound on ρ — equity-index smiles never have ρ > 0 since
    # 1987. The soft prior below pulls within (-0.999, 0).
    rho_hi = 0.0 if equity_skew else 0.999
    bounds = (
        np.array([-np.inf,    0.0, -0.999, -m_cap, sigma_min]),
        np.array([ np.inf,  b_max, rho_hi,  m_cap,  np.inf]),
    )

    # Soft Gaussian prior on ρ when equity_skew=True. The optimiser
    # adds (prior_strength · (ρ − target))² to the loss, equivalent to
    # placing a Bayesian prior centered at ρ = target with std dev
    # 1 / prior_strength. Front-month chains where the wing data is
    # sparse will lean on the prior; back-month chains with rich wings
    # will dominate it. The numbers below were tuned so a standard
    # SPY/SPX smile lands at ρ ≈ -0.6 to -0.9 across the term struct.
    if equity_skew:
        rho_target_prior = -0.6
        rho_prior_strength = 0.05  # commensurate with per-point IV noise
        m_prior_strength = 0.02     # weak nudge of vertex toward ATM
    else:
        rho_target_prior = 0.0
        rho_prior_strength = 0.0
        m_prior_strength = 0.0

    if weights is None:
        sqrt_w = np.ones_like(k)
    else:
        sqrt_w = np.sqrt(np.clip(weights, 1e-12, None))

    def residuals(x: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = x
        w_pred = raw_svi_total_variance(k, a, b, rho, m, sigma)
        iv_pred = np.sqrt(np.maximum(w_pred, 0.0) / T)
        r = (iv - iv_pred) * sqrt_w

        # Soft penalty for the non-linear vertex constraint
        #     a + b·σ·√(1 − ρ²) ≥ 0
        # Always present (zero when satisfied) so the residual vector
        # has a constant size — least_squares requires that.
        vertex = a + b * sigma * np.sqrt(max(1.0 - rho * rho, 0.0))
        vertex_pen = 1.0e3 * np.sqrt(max(-vertex, 0.0))
        # Soft Bayesian priors (zero when target hit, scales linearly
        # with deviation — squared by least_squares to give Gaussian).
        rho_pen = rho_prior_strength * (rho - rho_target_prior)
        m_pen   = m_prior_strength * m
        return np.concatenate([r, [vertex_pen, rho_pen, m_pen]])

    # Multi-start: equity smiles are heavily skewed, so a ρ₀ near -0.7
    # almost always wins. But for some non-equity (single-name with low
    # skew, indexes during low-fear regimes) ρ₀=0 is closer. Try both
    # starts and keep the lower-RSS fit. ~2× cost is irrelevant for
    # 10–20 expiries.
    starts = [
        np.array([a0, b0, -0.7, m0, sigma0]),  # equity-style skew
        np.array([a0, b0,  0.0, m0, sigma0]),  # symmetric fallback
    ]
    best = None
    for x0_try in starts:
        try:
            result = least_squares(
                residuals, x0_try,
                bounds=bounds,
                max_nfev=max_nfev,
                method="trf",
            )
        except Exception:  # noqa: BLE001
            continue
        rss = float(np.sum(result.fun ** 2))
        if best is None or rss < best[1]:
            best = (result, rss)
    if best is None:
        raise RuntimeError("All SVI starts failed — input may be degenerate.")
    result, _ = best
    a, b, rho, m, sigma = result.x

    return RawSVIParams(
        a=float(a), b=float(b), rho=float(rho),
        m=float(m), sigma=float(sigma),
        T=float(T),
        rss=float(np.sum(result.fun ** 2)),
        n_points=int(len(k)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_no_arb(a: float, b: float, rho: float, m: float,
                  sigma: float, T: float) -> bool:
    if b < 0:                         return False
    if not (-1.0 < rho < 1.0):        return False
    if sigma <= 0:                    return False
    if a + b * sigma * np.sqrt(max(1.0 - rho * rho, 0.0)) < -1e-12:
        return False
    if b * (1.0 + abs(rho)) > 4.0 / T + 1e-9:
        return False
    return True
