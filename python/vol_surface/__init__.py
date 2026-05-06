"""vol_surface — live options vol surface + mispricing scanner.

Public API:

    from vol_surface import OptionType, bs_price, bs_greeks, implied_vol
    from vol_surface import bs_price_batch, implied_vol_batch

Scalar examples:

    >>> from vol_surface import OptionType, bs_price, implied_vol
    >>> bs_price(OptionType.Call, S=100, K=100, T=1, r=0.05, q=0, sigma=0.20)
    10.45058357...
    >>> implied_vol(OptionType.Call, market_price=10.45, S=100, K=100,
    ...             T=1, r=0.05, q=0)
    0.1999...

Batch examples (vectorized — one Python call, all the work in C++):

    >>> import numpy as np
    >>> from vol_surface import implied_vol_batch
    >>> n = 1_000
    >>> ivs = implied_vol_batch(
    ...     types=np.zeros(n, dtype=np.int8),  # 0 = Call
    ...     market_price=np.full(n, 10.45),
    ...     S=np.full(n, 100.0),
    ...     K=np.full(n, 100.0),
    ...     T=np.full(n, 1.0),
    ...     r=np.full(n, 0.05),
    ...     q=np.zeros(n),
    ... )
    >>> # Failures come back as NaN.
    >>> good = ~np.isnan(ivs)
"""

from ._vol_kernel import (
    OptionType,
    Greeks,
    bs_price,
    bs_greeks,
    implied_vol,
    bs_price_batch,
    bs_greeks_batch,
    implied_vol_batch,
    strike_at_delta,
)

__all__ = [
    "OptionType",
    "Greeks",
    "bs_price",
    "bs_greeks",
    "implied_vol",
    "bs_price_batch",
    "bs_greeks_batch",
    "implied_vol_batch",
    "strike_at_delta",
]

__version__ = "0.1.0"
