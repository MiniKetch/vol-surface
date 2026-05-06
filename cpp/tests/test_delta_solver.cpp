// Tests for strike_at_delta — round-trip via the BS Greeks.

#include <doctest/doctest.h>

#include "vol/black_scholes.hpp"
#include "vol/delta_solver.hpp"
#include "vol/types.hpp"

#include <cmath>

using vol::OptionType;
using vol::bs_greeks;
using vol::strike_at_delta;

TEST_CASE("strike_at_delta — round-trip across delta values for calls") {
    // For a typical call: S=100, T=0.5, r=0.04, q=0.0, σ=0.20.
    // Sweep target delta from 0.05 (deep OTM) to 0.95 (deep ITM).
    constexpr double S = 100.0, T = 0.5, r = 0.04, q = 0.0, sigma = 0.20;
    for (double target : {0.10, 0.25, 0.40, 0.55, 0.70, 0.85}) {
        const auto K = strike_at_delta(OptionType::Call, target,
                                        S, T, r, q, sigma);
        CAPTURE(target);
        REQUIRE(K.has_value());
        // Verify by going back through the Greeks.
        const auto g = bs_greeks(OptionType::Call, S, *K, T, r, q, sigma);
        CHECK(g.delta == doctest::Approx(target).epsilon(1e-7));
    }
}

TEST_CASE("strike_at_delta — round-trip for puts (negative deltas)") {
    constexpr double S = 100.0, T = 0.5, r = 0.04, q = 0.015, sigma = 0.25;
    for (double target : {-0.10, -0.25, -0.40, -0.55, -0.70, -0.85}) {
        const auto K = strike_at_delta(OptionType::Put, target,
                                        S, T, r, q, sigma);
        CAPTURE(target);
        REQUIRE(K.has_value());
        const auto g = bs_greeks(OptionType::Put, S, *K, T, r, q, sigma);
        CHECK(g.delta == doctest::Approx(target).epsilon(1e-7));
    }
}

TEST_CASE("strike_at_delta — rejects targets with the wrong sign") {
    // Negative delta on a call → impossible.
    CHECK_FALSE(strike_at_delta(OptionType::Call, -0.25,
                                100.0, 0.5, 0.04, 0.0, 0.2).has_value());
    // Positive delta on a put → impossible.
    CHECK_FALSE(strike_at_delta(OptionType::Put, 0.25,
                                100.0, 0.5, 0.04, 0.0, 0.2).has_value());
}

TEST_CASE("strike_at_delta — rejects targets outside the reachable range") {
    constexpr double S = 100.0, T = 0.5, r = 0.04, q = 0.0, sigma = 0.20;
    // Call delta cap is exp(-q·T) = 1.0 here (q=0). Asking for 0.999 is
    // technically reachable but tests the boundary.
    CHECK_FALSE(strike_at_delta(OptionType::Call, 1.001,
                                S, T, r, q, sigma).has_value());
    CHECK_FALSE(strike_at_delta(OptionType::Call, 0.0,
                                S, T, r, q, sigma).has_value());
}

TEST_CASE("strike_at_delta — 25-delta call sits above forward, 25-d put below") {
    // Useful sanity check for skew metrics: at 25-delta, the call
    // strike is OTM (above forward) and the put strike is OTM (below).
    constexpr double S = 100.0, T = 0.5, r = 0.04, q = 0.015, sigma = 0.25;
    const double F = S * std::exp((r - q) * T);
    const auto K_call = strike_at_delta(OptionType::Call, 0.25,
                                         S, T, r, q, sigma);
    const auto K_put  = strike_at_delta(OptionType::Put, -0.25,
                                         S, T, r, q, sigma);
    REQUIRE(K_call.has_value());
    REQUIRE(K_put.has_value());
    CHECK(*K_call > F);
    CHECK(*K_put  < F);
}
