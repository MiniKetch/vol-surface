// vol/brent.hpp — Brent's method, a robust 1D root-finder.
//
// Why Brent (vs Newton-Raphson) for implied vol?
//   * It's a *bracketing* method — once you've found a sign change on
//     [a, b] it can't escape that interval. Newton-Raphson can shoot
//     off into the next county when vega is tiny (deep ITM/OTM).
//   * It mixes inverse quadratic interpolation (super-linear convergence
//     near the root) with bisection (guaranteed halving when the fast
//     step misbehaves). Practical convergence ≈ 6–10 iterations for IV.
//   * No derivatives needed — just price evaluations.
//
// This is a templated header so the compiler can fully inline both the
// algorithm and the function `f`. Callers keep their lambdas / functors
// without virtual dispatch overhead.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>

namespace vol::detail {

/// Brent's method.
///
/// @param f         A callable taking double, returning double. Must satisfy
///                  f(a) and f(b) of opposite sign — caller's responsibility.
/// @param a         Lower bracket.
/// @param b         Upper bracket.
/// @param x_tol     Stop when the bracket width (or step) is below this.
/// @param max_iter  Hard cap. Returns std::nullopt if exhausted.
/// @return          Approximate root, or std::nullopt on failure (no sign
///                  change in the bracket, or iterations exhausted).
template <typename F>
[[nodiscard]] std::optional<double> brent(
    F&& f, double a, double b,
    double x_tol = 1e-10,
    std::size_t max_iter = 100) noexcept(noexcept(f(0.0))) {

    double fa = f(a);
    double fb = f(b);

    // Caller is responsible for the bracket — but we double-check.
    if (fa * fb > 0.0) {
        return std::nullopt;
    }

    // Conventional Brent setup: |f(b)| should be the smaller, so b is
    // the "best so far" estimate.
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c   = a;        // previous contrapoint
    double fc  = fa;
    double d   = b - a;    // last *successful* step
    double e   = d;        // step before that
    bool   used_bisection = true;

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        if (fb == 0.0) return b;
        if (std::abs(b - a) < x_tol) return b;

        // Re-bracket: keep the contrapoint c on the opposite side of b.
        if (std::abs(fc) < std::abs(fb)) {
            a  = b;  b  = c;  c  = a;
            fa = fb; fb = fc; fc = fa;
        }

        const double tol1 = 2.0 * std::numeric_limits<double>::epsilon() *
                            std::abs(b) + 0.5 * x_tol;
        const double xm   = 0.5 * (c - b);

        if (std::abs(xm) <= tol1) return b;

        // Try inverse quadratic interpolation (or secant if a == c),
        // falling back to bisection when the step is unreliable.
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            const double s = fb / fa;
            double p, q;
            if (a == c) {
                // Secant: only two distinct points.
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                // Full inverse quadratic.
                const double q_ = fa / fc;
                const double r  = fb / fc;
                p = s * (2.0 * xm * q_ * (q_ - r) - (b - a) * (r - 1.0));
                q = (q_ - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::abs(p);

            // Accept the interpolated step only if it stays in-bounds and
            // is genuinely shrinking the interval.
            const double min1 = 3.0 * xm * q - std::abs(tol1 * q);
            const double min2 = std::abs(e * q);
            if (2.0 * p < std::min(min1, min2)) {
                e = d;
                d = p / q;
                used_bisection = false;
            } else {
                d = xm;
                e = d;
                used_bisection = true;
            }
        } else {
            // Bisection fallback.
            d = xm;
            e = d;
            used_bisection = true;
        }
        (void)used_bisection;  // reserved for stats / tracing later

        a  = b;
        fa = fb;
        if (std::abs(d) > tol1) {
            b += d;
        } else {
            b += (xm > 0.0 ? tol1 : -tol1);
        }
        fb = f(b);

        // Maintain the sign-change invariant on [b, c].
        if ((fb > 0.0) == (fc > 0.0)) {
            c  = a;
            fc = fa;
            d  = b - a;
            e  = d;
        }
    }
    return std::nullopt;  // exhausted
}

}  // namespace vol::detail
