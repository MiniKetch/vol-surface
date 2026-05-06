// vol/black_scholes.cpp — implementation.
//
// Numerical notes (the bits worth understanding):
//
//   * The standard normal CDF is implemented as `0.5 * erfc(-x/√2)` rather
//     than `0.5 * (1 + erf(x/√2))`. Both are mathematically identical, but
//     the erfc form preserves precision in the tails — when x is very
//     negative, `1 + erf(x/√2)` suffers catastrophic cancellation as you
//     subtract two near-equal numbers. erfc returns the small tail directly.
//
//   * Edge cases (T → 0 or σ → 0) are handled exactly, returning the
//     deterministic forward intrinsic. We use `!(x > 0)` rather than
//     `x <= 0` so NaN inputs propagate as NaN rather than collapsing to a
//     bogus zero.
//
//   * All Greeks share the same d1, d2, sqrt(T), and discount factors,
//     so we compute them once. That's why we return a Greeks struct
//     instead of having five separate functions.

#include "vol/black_scholes.hpp"

#include <algorithm>
#include <cmath>

namespace vol {

namespace {

// Pre-computed constants. constexpr means they live in .rodata, not BSS,
// and the compiler folds them into immediates.
constexpr double kInvSqrt2   = 0.7071067811865475;  // 1 / √2
constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1 / √(2π)

// Closed-form value when σ = 0 or T = 0: the option is worth the
// discounted forward intrinsic. With continuous dividend yield q,
// the forward is F = S · exp((r-q)·T), and the call payoff at expiry is
// max(F-K, 0) discounted by exp(-r·T) — which simplifies to
// max(S·exp(-q·T) - K·exp(-r·T), 0). Same logic for puts with sign flipped.
double deterministic_value(OptionType type, double S, double K, double T,
                           double r, double q) noexcept {
    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);
    if (type == OptionType::Call) {
        return std::max(S * disc_q - K * disc_r, 0.0);
    }
    return std::max(K * disc_r - S * disc_q, 0.0);
}

}  // namespace

double std_norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

double std_norm_pdf(double x) noexcept {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

double bs_price(OptionType type,
                double S, double K, double T,
                double r, double q,
                double sigma) noexcept {
    if (!(T > 0.0) || !(sigma > 0.0)) {
        return deterministic_value(type, S, K, T, r, q);
    }
    const double sqrt_T = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T)
                      / (sigma * sqrt_T);
    const double d2 = d1 - sigma * sqrt_T;
    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);

    if (type == OptionType::Call) {
        return S * disc_q * std_norm_cdf(d1) - K * disc_r * std_norm_cdf(d2);
    }
    return K * disc_r * std_norm_cdf(-d2) - S * disc_q * std_norm_cdf(-d1);
}

Greeks bs_greeks(OptionType type,
                 double S, double K, double T,
                 double r, double q,
                 double sigma) noexcept {
    Greeks g{};

    if (!(T > 0.0) || !(sigma > 0.0)) {
        // Degenerate: gamma, vega, and the smooth Greeks collapse to zero
        // (or are undefined at the strike). We return zeros except for
        // delta, which keeps a defensible value for ITM/OTM. Real callers
        // should filter these contracts out; this is a safety net.
        const double T_safe  = std::max(T, 0.0);
        const double disc_q  = std::exp(-q * T_safe);
        const double disc_r  = std::exp(-r * T_safe);
        const bool   itm     = (type == OptionType::Call)
                                   ? (S * disc_q > K * disc_r)
                                   : (S * disc_q < K * disc_r);
        if (itm) {
            g.delta = (type == OptionType::Call) ? disc_q : -disc_q;
        }
        return g;
    }

    const double sqrt_T = std::sqrt(T);
    const double d1     = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T)
                          / (sigma * sqrt_T);
    const double d2     = d1 - sigma * sqrt_T;
    const double disc_q = std::exp(-q * T);
    const double disc_r = std::exp(-r * T);
    const double pdf_d1 = std_norm_pdf(d1);

    // Gamma and vega are sign-symmetric across calls and puts.
    g.gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T);
    g.vega  = S * disc_q * pdf_d1 * sqrt_T;

    if (type == OptionType::Call) {
        const double Nd1 = std_norm_cdf(d1);
        const double Nd2 = std_norm_cdf(d2);
        g.delta = disc_q * Nd1;
        g.theta = -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_T)
                  - r * K * disc_r * Nd2
                  + q * S * disc_q * Nd1;
        g.rho   = K * T * disc_r * Nd2;
    } else {
        const double Nm_d1 = std_norm_cdf(-d1);
        const double Nm_d2 = std_norm_cdf(-d2);
        g.delta = -disc_q * Nm_d1;
        g.theta = -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_T)
                  + r * K * disc_r * Nm_d2
                  - q * S * disc_q * Nm_d1;
        g.rho   = -K * T * disc_r * Nm_d2;
    }
    return g;
}

}  // namespace vol
