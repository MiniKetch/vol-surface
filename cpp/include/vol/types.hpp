// vol/types.hpp
// Shared types for the vol_kernel library.
//
// Design notes:
//   * Header-only — these are zero-cost abstractions (POD structs, enums)
//     that the compiler can fully inline. No .cpp file needed.
//   * `enum class` (not plain `enum`) gives us strong typing: you can't
//     accidentally compare an OptionType to an int, or to a different
//     scoped enum. Modern C++ default.
//   * Greeks live in a single struct so a single function call can return
//     the whole bundle — better cache locality and fewer redundant
//     d1/d2 / pdf evaluations than calling delta(), gamma(), vega() etc.
//     separately.

#pragma once

namespace vol {

enum class OptionType : int {
    Call,
    Put
};

/// Standard option Greeks. All in "raw" mathematical form:
///   - vega is dPrice/dSigma (per 1.0 of vol). Divide by 100 for
///     "per 1 vol point" (the trader convention).
///   - theta is dPrice/dT in years (negative for long options). Divide
///     by 365 (or 252) for daily theta.
///   - rho is per 1.0 of rate. Divide by 100 for "per 1 bp".
/// We keep raw units in the kernel so callers pick their own scaling.
struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

}  // namespace vol
