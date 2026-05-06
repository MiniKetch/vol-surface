// vol/delta_solver.hpp — invert delta(K) → K.
//
// Why this exists: skew metrics are quoted at *fixed deltas*, not
// fixed strikes. The "25-delta risk reversal" is the IV at the K such
// that the call's delta is +0.25 minus the IV at the K such that the
// put's delta is −0.25. Same idea for butterfly. So we need to find
// the strike that produces a given delta at a given σ.
//
// BS delta is monotonic in K (calls strictly decreasing, puts strictly
// increasing as K rises), so this is a 1-D root problem with a
// guaranteed sign change. Brent eats it in a handful of iterations.

#pragma once

#include "vol/types.hpp"

#include <optional>

namespace vol {

/// Find the strike K such that the BS delta of an option with the
/// given parameters equals ``target_delta``.
///
/// @param type         OptionType::Call (target_delta in (0, exp(-q·T)))
///                     or OptionType::Put (target_delta in (-exp(-q·T), 0)).
/// @param target_delta The delta to solve for. Sign must match type.
/// @param S, T, r, q, sigma
///                     Standard BS inputs. ``sigma`` is the IV at the
///                     strike you want to *find*; for the typical
///                     "25-delta" use case you'll plug in the SVI-fit
///                     IV at that delta level (iterate if you want
///                     consistency, but one-shot is usually close enough).
/// @param tol          Strike-space tolerance.
/// @param max_iter     Cap on Brent iterations.
///
/// @return The strike, or std::nullopt if the target delta isn't
/// achievable (e.g. asking for delta=0.99 on a low-vol option).
[[nodiscard]] std::optional<double> strike_at_delta(
    OptionType type,
    double target_delta,
    double S, double T,
    double r, double q,
    double sigma,
    double tol = 1.0e-8,
    int max_iter = 100) noexcept;

}  // namespace vol
