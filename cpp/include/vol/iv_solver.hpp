// vol/iv_solver.hpp — implied volatility from observed market price.
//
// Returns std::optional<double> rather than throwing or returning a
// sentinel: many real-world contracts have no defensible IV (price below
// intrinsic, expired, locked-limit, etc.) and the caller needs to know.
// `std::optional` makes that explicit at the type level — you can't
// accidentally use a missing IV without writing `.value()` or checking
// `.has_value()`.

#pragma once

#include "vol/types.hpp"

#include <optional>

namespace vol {

/// Solve for the volatility σ such that
///     bs_price(type, S, K, T, r, q, σ) == market_price.
///
/// @param type           Call or Put.
/// @param market_price   The observed price (typically the mid of bid/ask).
/// @param S, K, T, r, q  Standard BS inputs.
/// @param tol            Convergence tolerance on σ (default 1e-8).
/// @param max_iter       Hard iteration cap (default 100).
///
/// @return σ if found, else std::nullopt. Returns nullopt when:
///   * T <= 0 (no time value left).
///   * market_price is below intrinsic (no real IV exists).
///   * market_price exceeds the maximum BS price (would imply σ → ∞).
///   * Solver fails to converge within max_iter.
[[nodiscard]] std::optional<double> implied_vol(
    OptionType type,
    double market_price,
    double S, double K, double T,
    double r, double q,
    double tol = 1e-8,
    int max_iter = 100) noexcept;

}  // namespace vol
