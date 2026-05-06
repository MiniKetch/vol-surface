// vol/delta_solver.cpp — implementation of strike_at_delta.

#include "vol/delta_solver.hpp"

#include "vol/black_scholes.hpp"
#include "vol/brent.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vol {

namespace {

// Bracket bounds for K. We bracket aggressively in log-space:
// [F·exp(-5σ√T), F·exp(+5σ√T)] covers anything you'll see in practice
// (5 standard deviations of log-moneyness). Outside that, the delta is
// numerically saturated and there's no useful root.
double bracket_low(double S, double T, double r, double q,
                   double sigma) noexcept {
    const double F = S * std::exp((r - q) * T);
    const double width = std::max(5.0 * sigma * std::sqrt(T), 0.1);
    return F * std::exp(-width);
}

double bracket_high(double S, double T, double r, double q,
                    double sigma) noexcept {
    const double F = S * std::exp((r - q) * T);
    const double width = std::max(5.0 * sigma * std::sqrt(T), 0.1);
    return F * std::exp(+width);
}

}  // namespace

std::optional<double> strike_at_delta(
    OptionType type,
    double target_delta,
    double S, double T,
    double r, double q,
    double sigma,
    double tol,
    int max_iter) noexcept {

    // Pre-flight — reject inputs that can't have a real answer.
    if (!(T > 0.0) || !(S > 0.0) || !(sigma > 0.0)) return std::nullopt;
    if (!std::isfinite(target_delta)) return std::nullopt;

    // The reachable delta range is sign-constrained:
    //   call delta ∈ (0, exp(-q·T))
    //   put  delta ∈ (-exp(-q·T), 0)
    const double disc_q = std::exp(-q * T);
    if (type == OptionType::Call) {
        if (!(target_delta > 0.0 && target_delta < disc_q)) return std::nullopt;
    } else {
        if (!(target_delta < 0.0 && target_delta > -disc_q)) return std::nullopt;
    }

    // Objective: f(K) = delta(K) − target. Sign-monotone in K.
    auto objective = [&](double K) noexcept {
        if (!(K > 0.0)) return type == OptionType::Call ? -target_delta
                                                         : -target_delta;
        const auto g = bs_greeks(type, S, K, T, r, q, sigma);
        return g.delta - target_delta;
    };

    return detail::brent(
        objective,
        bracket_low(S, T, r, q, sigma),
        bracket_high(S, T, r, q, sigma),
        tol,
        static_cast<std::size_t>(max_iter));
}

}  // namespace vol
