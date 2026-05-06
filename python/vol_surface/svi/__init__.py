"""SVI parametric volatility surface model.

The "Stochastic Volatility Inspired" parameterization (Gatheral 2004)
is the industry-standard parametric form for an equity-index implied
volatility smile. It's flexible enough to fit real market smiles
closely, has a clean no-arbitrage constraint set, and gives you smooth
derivatives — which matters for downstream modelling (delta-hedging,
local-vol bootstrapping, etc.).
"""

from vol_surface.svi.raw_svi import RawSVIParams, fit_raw_svi, raw_svi_total_variance, raw_svi_iv

__all__ = [
    "RawSVIParams",
    "fit_raw_svi",
    "raw_svi_total_variance",
    "raw_svi_iv",
]
