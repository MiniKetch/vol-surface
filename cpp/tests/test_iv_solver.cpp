// IV solver tests.
//
// Strategy: round-trip. Pick a σ, compute the BS price, hand the price
// back to the solver, check we recover σ. If both directions are right,
// we'll match. If either is wrong, we won't.
//
// We also test the failure modes — sub-intrinsic, expired, garbage —
// because in production these are far more common than "valid input but
// math is wrong."

#include <doctest/doctest.h>

#include "vol/black_scholes.hpp"
#include "vol/brent.hpp"
#include "vol/iv_solver.hpp"
#include "vol/types.hpp"

#include <cmath>
#include <initializer_list>

using vol::OptionType;
using vol::bs_price;
using vol::implied_vol;

// ---------------------------------------------------------------------------
// Round-trip across the whole surface — moneyness × maturity × IV × type.
// If any combination fails to recover σ within tolerance, we want to know
// which one (hence the CAPTURE calls).
// ---------------------------------------------------------------------------

TEST_CASE("implied_vol — round-trip recovery across the surface") {
    constexpr double S = 100.0;
    constexpr double r = 0.04;
    constexpr double q = 0.015;

    // The principled "is σ recoverable from this price?" test is vega.
    // If vega is, say, 5e-5, then a 1e-6 move in σ shifts the price by
    // ~5e-11. On a $100-ish option that's already below FP precision in
    // the subtraction S·N(d1) − K·N(d2). For our 1e-6 σ tolerance we
    // need vega > ~1e-3 to have headroom over FP noise.
    //
    // This isn't a solver bug — it's a fundamental information limit.
    // Real data layers filter these contracts (deep ITM/OTM at low σ,
    // very short DTE) before they ever reach an IV computation.
    auto is_resolvable = [&](OptionType t, double K, double T, double sigma) {
        const auto g = vol::bs_greeks(t, S, K, T, r, q, sigma);
        const double price = bs_price(t, S, K, T, r, q, sigma);
        return g.vega > 1.0e-3 && price > 1.0e-4;
    };

    int tested = 0;
    int skipped = 0;
    for (auto type : {OptionType::Call, OptionType::Put}) {
        for (double K : {60.0, 80.0, 100.0, 120.0, 150.0}) {
            for (double T : {0.05, 0.25, 1.0, 2.0}) {
                for (double sigma : {0.05, 0.15, 0.30, 0.60, 1.20}) {
                    if (!is_resolvable(type, K, T, sigma)) {
                        ++skipped;
                        continue;
                    }
                    ++tested;
                    const double price = bs_price(type, S, K, T, r, q, sigma);
                    const auto iv = implied_vol(type, price, S, K, T, r, q);
                    CAPTURE(static_cast<int>(type));
                    CAPTURE(K); CAPTURE(T); CAPTURE(sigma); CAPTURE(price);
                    REQUIRE(iv.has_value());
                    CHECK(*iv == doctest::Approx(sigma).epsilon(1e-6));
                }
            }
        }
    }
    // Sanity: we should still be testing the bulk of the grid.
    CHECK(tested > 100);
    MESSAGE("round-trip: tested=", tested, " skipped (pathological)=", skipped);
}

// ---------------------------------------------------------------------------
// Failure modes — the solver should *say no*, not return garbage.
// ---------------------------------------------------------------------------

TEST_CASE("implied_vol — rejects sub-intrinsic price") {
    // Call with S=110, K=100 has intrinsic ≥ 9.something. Asking for IV
    // when price = 1.0 is impossible — should return nullopt.
    const auto iv = implied_vol(OptionType::Call, 1.0,
                                110.0, 100.0, 1.0, 0.05, 0.0);
    CHECK_FALSE(iv.has_value());
}

TEST_CASE("implied_vol — rejects price above upper bound") {
    // Call price can't exceed S·exp(-q·T) ≈ 100. Ask for 200.
    const auto iv = implied_vol(OptionType::Call, 200.0,
                                100.0, 100.0, 1.0, 0.05, 0.0);
    CHECK_FALSE(iv.has_value());
}

TEST_CASE("implied_vol — rejects expired contracts") {
    const auto iv = implied_vol(OptionType::Call, 5.0,
                                100.0, 100.0, 0.0, 0.05, 0.0);
    CHECK_FALSE(iv.has_value());
}

TEST_CASE("implied_vol — rejects nonsense inputs") {
    CHECK_FALSE(implied_vol(OptionType::Call, -1.0,
                            100.0, 100.0, 1.0, 0.05, 0.0).has_value());
    CHECK_FALSE(implied_vol(OptionType::Call, 5.0,
                            -100.0, 100.0, 1.0, 0.05, 0.0).has_value());
    CHECK_FALSE(implied_vol(OptionType::Call, 5.0,
                            100.0, -100.0, 1.0, 0.05, 0.0).has_value());
}

// ---------------------------------------------------------------------------
// Smile-shape sanity check: deep OTM puts and OTM calls on the same
// underlying should both be solvable cleanly with realistic dollar prices.
// This is the regime where Newton-Raphson notoriously fails (vega → 0).
// ---------------------------------------------------------------------------

TEST_CASE("implied_vol — handles OTM wings (Newton would fail here)") {
    constexpr double S = 100.0, T = 0.25, r = 0.04, q = 0.0;
    constexpr double sigma_true = 0.40;

    // Deep OTM call: K = 150, σ = 40 %. Tiny price, tiny vega.
    {
        const double price = bs_price(OptionType::Call, S, 150.0, T, r, q, sigma_true);
        const auto iv = implied_vol(OptionType::Call, price, S, 150.0, T, r, q);
        REQUIRE(iv.has_value());
        CHECK(*iv == doctest::Approx(sigma_true).epsilon(1e-6));
    }
    // Deep OTM put: K = 50.
    {
        const double price = bs_price(OptionType::Put, S, 50.0, T, r, q, sigma_true);
        const auto iv = implied_vol(OptionType::Put, price, S, 50.0, T, r, q);
        REQUIRE(iv.has_value());
        CHECK(*iv == doctest::Approx(sigma_true).epsilon(1e-6));
    }
}

// ---------------------------------------------------------------------------
// Brent itself: a polynomial root we can verify by hand. This catches
// algorithm bugs that wouldn't show up in IV tests because BS price is
// well-behaved.
// ---------------------------------------------------------------------------

TEST_CASE("brent — finds the root of x^3 - x - 2 ≈ 1.521379706") {
    auto f = [](double x) { return x * x * x - x - 2.0; };
    const auto root = vol::detail::brent(f, 1.0, 2.0, 1e-12);
    REQUIRE(root.has_value());
    CHECK(*root == doctest::Approx(1.5213797068045676).epsilon(1e-10));
}

TEST_CASE("brent — returns nullopt when bracket has no sign change") {
    auto f = [](double x) { return x * x + 1.0; };  // never zero
    CHECK_FALSE(vol::detail::brent(f, -1.0, 1.0).has_value());
}
