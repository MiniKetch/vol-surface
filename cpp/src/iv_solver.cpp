// vol/iv_solver.cpp — implementation of implied_vol via Brent root-finding.

#include "vol/iv_solver.hpp"

#include "vol/black_scholes.hpp"
#include "vol/brent.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vol {

namespace {

// Bracket bounds for σ. 1e-8 is effectively zero (deterministic forward),
// 5.0 = 500 % vol — well past anything we'd ever see on a real chain.
// If the root is outside this range, the input is broken.
constexpr double kSigmaLow  = 1.0e-8;
constexpr double kSigmaHigh = 5.0;

}  // namespace

std::optional<double> implied_vol(
    OptionType type,
    double market_price,
    double S, double K, double T,
    double r, double q,
    double tol,
    int max_iter) noexcept {

    // ------------------------------------------------------------------
    // Pre-flight sanity checks. These are the *most common* causes of
    // "the IV solver returned NaN" complaints from beginners:
    //
    //   1. T <= 0          — option has expired, no IV exists.
    //   2. price < intrinsic — would imply negative time value, which
    //      is impossible for European options under BS.
    //   3. price >= upper bound — would require σ → ∞.
    //
    // Bailing out early with std::nullopt makes the failure mode
    // visible to callers instead of silently returning a garbage σ.
    // ------------------------------------------------------------------
    if (!(T > 0.0) || !(S > 0.0) || !(K > 0.0)) {
        return std::nullopt;
    }
    if (!std::isfinite(market_price) || market_price < 0.0) {
        return std::nullopt;
    }

    // We evaluate the bounds via bs_price itself rather than the analytical
    // closed forms. Why: at deep ITM with tiny σ, the closed-form forward
    // intrinsic can sit a few ULPs *above* the actual computed BS price
    // (subtractive cancellation in S·N(d1) − K·N(d2)). Using the same
    // function the solver will minimise eliminates that mismatch — the
    // bracket is now self-consistent by construction.
    const double price_lo = bs_price(type, S, K, T, r, q, kSigmaLow);
    const double price_hi = bs_price(type, S, K, T, r, q, kSigmaHigh);

    // Generous tolerance: 1e-10 absolute, or 1e-10 relative to the
    // upper-bound price, whichever is larger. Below this we're inside FP
    // noise and can't resolve σ meaningfully anyway.
    const double slack = std::max(1.0e-10, 1.0e-10 * std::abs(price_hi));

    if (market_price < price_lo - slack) return std::nullopt;
    if (market_price > price_hi + slack) return std::nullopt;

    // Prices sitting at either bracket edge are *ill-posed*: the BS
    // function is essentially flat there in σ, so any value in a wide
    // range fits. Returning nullopt rather than kSigmaLow/kSigmaHigh
    // signals to the caller "we can't recover σ from this price",
    // which is the truthful answer and propagates as NaN through the
    // numpy batch path.
    if (market_price <= price_lo + slack) return std::nullopt;
    if (market_price >= price_hi - slack) return std::nullopt;

    // ------------------------------------------------------------------
    // The objective: f(σ) = bs_price(σ) − market_price. BS price is
    // strictly monotonic in σ for European options, so f has exactly
    // one root in (kSigmaLow, kSigmaHigh) given the bounds above.
    //
    // We capture by value — these are all doubles, no aliasing.
    // ------------------------------------------------------------------
    auto objective = [&](double sigma) noexcept {
        return bs_price(type, S, K, T, r, q, sigma) - market_price;
    };

    const auto root = detail::brent(
        objective,
        kSigmaLow,
        kSigmaHigh,
        tol,
        static_cast<std::size_t>(max_iter));

    if (!root.has_value()) return std::nullopt;

    // Post-solve sanity gate. Even when Brent converges, the answer is
    // only meaningful if the BS price is sensitive to σ at that point —
    // i.e. vega is non-trivial. Deep-OTM contracts with tiny prices
    // suffer subtractive cancellation in S·N(d1)−K·N(d2) and can
    // accept any σ across a wide range.
    //
    // The threshold is *relative to S* rather than absolute, because
    // raw vega scales with S: a $5 stock's ATM vega is ~50× smaller
    // than a $500 stock's. An absolute 1e-4 cutoff that's harmless
    // for SPY silently NaNs out cheap-stock weeklies. ~1e-6 of S keeps
    // the same intent (price has at least 1 ULP of sensitivity to σ
    // on a typical mid-price) without that artefact.
    const double kMinRelVegaForResolvability = 1.0e-6 * S;
    const auto greeks = bs_greeks(type, S, K, T, r, q, *root);
    if (!(greeks.vega > kMinRelVegaForResolvability)) return std::nullopt;

    return root;
}

}  // namespace vol
