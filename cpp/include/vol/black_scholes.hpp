// vol/black_scholes.hpp
// Black-Scholes-Merton pricing and Greeks for European options on a
// dividend-paying underlying (continuous yield q).
//
// Design notes:
//   * Free functions, not a class. There's no state to encapsulate —
//     a price is a pure function of its inputs. Classes here would
//     just be namespaces with extra ceremony.
//   * `noexcept` because we never throw — we return finite numbers
//     (or NaN, if the caller hands us garbage) and let the caller
//     decide what to do.
//   * `[[nodiscard]]` so you can't silently throw away a computed
//     price. If you want to ignore it you must `(void)` cast it.
//   * Inputs are passed as scalars rather than a struct because
//     pybind11 vectorization and direct callers both prefer that
//     shape; we'll add a struct overload if it becomes useful.

#pragma once

#include "vol/types.hpp"

namespace vol {

// ---------------------------------------------------------------------------
// Standard normal helpers — exposed so tests can target them directly,
// and so callers building their own model formulas don't have to roll
// their own. Implemented via std::erfc for numerical stability in the
// tails.
// ---------------------------------------------------------------------------

[[nodiscard]] double std_norm_cdf(double x) noexcept;
[[nodiscard]] double std_norm_pdf(double x) noexcept;

// ---------------------------------------------------------------------------
// Black-Scholes-Merton price.
//
// Parameters:
//   type   Call or Put.
//   S      Spot price of the underlying.
//   K      Strike.
//   T      Time to expiry, in years (e.g. 30 days = 30.0/365.0).
//   r      Continuously-compounded risk-free rate (e.g. 0.04 = 4 %).
//   q      Continuous dividend yield (0.0 if none).
//   sigma  Volatility (annualised, e.g. 0.30 = 30 %).
//
// Edge cases (returned exactly, no NaN):
//   * T <= 0  → undiscounted intrinsic max(S-K, 0) (call) / max(K-S, 0) (put).
//   * sigma <= 0 → forward intrinsic, properly discounted.
// Anything else with non-finite inputs → NaN propagates.
// ---------------------------------------------------------------------------

[[nodiscard]] double bs_price(
    OptionType type,
    double S, double K, double T,
    double r, double q,
    double sigma) noexcept;

// ---------------------------------------------------------------------------
// All Greeks in one shot.
//
// Returning a struct is faster than five separate calls because we only
// compute d1, d2, sqrt(T), and the discount factors once.
// ---------------------------------------------------------------------------

[[nodiscard]] Greeks bs_greeks(
    OptionType type,
    double S, double K, double T,
    double r, double q,
    double sigma) noexcept;

}  // namespace vol
