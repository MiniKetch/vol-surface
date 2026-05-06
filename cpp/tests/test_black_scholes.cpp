// Black-Scholes tests.
//
// We pin to *known* reference values so a regression in either the
// pricing formula or the underlying erfc/exp implementation is caught
// immediately. The reference numbers below are computed analytically
// (hand-derived in the project notes); each test names its inputs.

#include <doctest/doctest.h>

#include "vol/black_scholes.hpp"
#include "vol/types.hpp"

#include <cmath>
#include <initializer_list>

using vol::OptionType;
using vol::bs_price;
using vol::bs_greeks;
using vol::std_norm_cdf;
using vol::std_norm_pdf;

// ---------------------------------------------------------------------------
// Normal-distribution sanity checks.
// ---------------------------------------------------------------------------

TEST_CASE("std_norm_cdf — symmetric and bounded") {
    CHECK(std_norm_cdf(0.0)  == doctest::Approx(0.5).epsilon(1e-12));
    CHECK(std_norm_cdf( 1.0) == doctest::Approx(0.8413447460685429).epsilon(1e-12));
    CHECK(std_norm_cdf(-1.0) == doctest::Approx(0.1586552539314571).epsilon(1e-12));
    // Tail behaviour — the erfc form keeps precision here.
    CHECK(std_norm_cdf(-8.0) > 0.0);
    CHECK(std_norm_cdf( 8.0) < 1.0);
}

TEST_CASE("std_norm_pdf — peak at zero, drops off") {
    CHECK(std_norm_pdf(0.0) == doctest::Approx(0.3989422804014327).epsilon(1e-12));
    CHECK(std_norm_pdf( 1.0) == doctest::Approx(0.24197072451914337).epsilon(1e-12));
    CHECK(std_norm_pdf(-1.0) == doctest::Approx(0.24197072451914337).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// Reference-value pricing test.
//
// Inputs:  S=100, K=100, T=1, r=5%, q=0, σ=20%.
// d1 = (ln(1) + (0.05 + 0.02)·1) / (0.20·1) = 0.35
// d2 = d1 − σ√T = 0.15
// N(d1) ≈ 0.636830651...
// N(d2) ≈ 0.559617692...
// Call  = 100·N(d1) − 100·exp(−0.05)·N(d2) ≈ 10.4505835...
// Put   = 100·exp(−0.05)·N(−d2) − 100·N(−d1) ≈ 5.5735265...
// ---------------------------------------------------------------------------

TEST_CASE("bs_price — ATM 1Y reference values") {
    constexpr double S = 100.0, K = 100.0, T = 1.0;
    constexpr double r = 0.05, q = 0.0, sigma = 0.20;
    CHECK(bs_price(OptionType::Call, S, K, T, r, q, sigma)
          == doctest::Approx(10.45058357).epsilon(1e-7));
    CHECK(bs_price(OptionType::Put, S, K, T, r, q, sigma)
          == doctest::Approx(5.57352602).epsilon(1e-7));
}

// ---------------------------------------------------------------------------
// Put-call parity: C − P = S·exp(−q·T) − K·exp(−r·T).
// Holding for arbitrary inputs is the strongest cross-check we have
// short of pinning every digit.
// ---------------------------------------------------------------------------

TEST_CASE("bs_price — put-call parity holds across the surface") {
    const double S = 75.0, T = 0.45;
    const double r = 0.04, q = 0.015;
    for (double K : {50.0, 65.0, 75.0, 90.0, 110.0}) {
        for (double sigma : {0.10, 0.25, 0.45, 0.85}) {
            const double c = bs_price(OptionType::Call, S, K, T, r, q, sigma);
            const double p = bs_price(OptionType::Put,  S, K, T, r, q, sigma);
            const double parity = S * std::exp(-q * T) - K * std::exp(-r * T);
            CAPTURE(K); CAPTURE(sigma);
            CHECK((c - p) == doctest::Approx(parity).epsilon(1e-10));
        }
    }
}

// ---------------------------------------------------------------------------
// No-arbitrage bounds: a European call is worth at least its forward
// intrinsic and at most its discounted spot. Same for puts (mirrored).
// ---------------------------------------------------------------------------

TEST_CASE("bs_price — bounded between intrinsic and underlying") {
    const double S = 100.0, K = 110.0, T = 0.75;
    const double r = 0.05, q = 0.02, sigma = 0.30;

    const double call = bs_price(OptionType::Call, S, K, T, r, q, sigma);
    const double lb_call = std::max(S * std::exp(-q * T)
                                    - K * std::exp(-r * T), 0.0);
    CHECK(call >= lb_call - 1e-12);
    CHECK(call <= S * std::exp(-q * T) + 1e-12);

    const double put = bs_price(OptionType::Put, S, K, T, r, q, sigma);
    const double lb_put = std::max(K * std::exp(-r * T)
                                   - S * std::exp(-q * T), 0.0);
    CHECK(put >= lb_put - 1e-12);
    CHECK(put <= K * std::exp(-r * T) + 1e-12);
}

// ---------------------------------------------------------------------------
// Degenerate inputs — should return deterministic intrinsic, never NaN.
// ---------------------------------------------------------------------------

TEST_CASE("bs_price — T=0 returns intrinsic value") {
    CHECK(bs_price(OptionType::Call, 110.0, 100.0, 0.0, 0.05, 0.0, 0.20)
          == doctest::Approx(10.0).epsilon(1e-12));
    CHECK(bs_price(OptionType::Call, 90.0, 100.0, 0.0, 0.05, 0.0, 0.20)
          == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(bs_price(OptionType::Put,  90.0, 100.0, 0.0, 0.05, 0.0, 0.20)
          == doctest::Approx(10.0).epsilon(1e-12));
}

TEST_CASE("bs_price — sigma=0 collapses to discounted forward intrinsic") {
    constexpr double S = 100.0, K = 100.0, T = 1.0;
    constexpr double r = 0.05, q = 0.0;
    // With σ=0, forward F = S·exp((r−q)T) = 100·exp(0.05) = 105.127...
    // Call = exp(−rT)·max(F−K, 0) = exp(−0.05)·5.127 = 4.8771
    CHECK(bs_price(OptionType::Call, S, K, T, r, q, 0.0)
          == doctest::Approx(4.8770575).epsilon(1e-7));
    // Put = 0 because the forward is above the strike.
    CHECK(bs_price(OptionType::Put,  S, K, T, r, q, 0.0)
          == doctest::Approx(0.0).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// Greeks reference values (same ATM 1Y inputs as the price test).
//
// delta_call  = N(d1)               ≈ 0.6368307
// delta_put   = −N(−d1)             ≈ −0.3631693
// gamma       = pdf(d1)/(Sσ√T)      ≈ 0.0187620
// vega        = S·pdf(d1)·√T        ≈ 37.52403   (per 1.0 vol)
// theta_call  = −S·pdf(d1)·σ/(2√T) − r·K·exp(−rT)·N(d2)
//             ≈ −6.41403            (per year)
// rho_call    = K·T·exp(−rT)·N(d2)  ≈ 53.23248
// ---------------------------------------------------------------------------

TEST_CASE("bs_greeks — ATM 1Y reference values for a call") {
    const auto g = bs_greeks(OptionType::Call,
                             100.0, 100.0, 1.0, 0.05, 0.0, 0.20);
    CHECK(g.delta == doctest::Approx( 0.63683066).epsilon(1e-7));
    CHECK(g.gamma == doctest::Approx( 0.01876202).epsilon(1e-7));
    CHECK(g.vega  == doctest::Approx(37.52403469).epsilon(1e-7));
    CHECK(g.theta == doctest::Approx(-6.41402737).epsilon(1e-6));
    CHECK(g.rho   == doctest::Approx(53.23248154).epsilon(1e-6));
}

TEST_CASE("bs_greeks — call/put delta sum equals exp(-q·T)") {
    // d_call − d_put = exp(-q·T) — a dividend-aware analogue of the
    // classic "delta sum is 1" identity. Holds for any inputs.
    const double q = 0.015, T = 0.5;
    const auto gc = bs_greeks(OptionType::Call, 90.0, 100.0, T, 0.04, q, 0.30);
    const auto gp = bs_greeks(OptionType::Put,  90.0, 100.0, T, 0.04, q, 0.30);
    CHECK((gc.delta - gp.delta)
          == doctest::Approx(std::exp(-q * T)).epsilon(1e-12));
}

TEST_CASE("bs_greeks — gamma and vega are sign-symmetric across types") {
    const auto gc = bs_greeks(OptionType::Call, 95.0, 100.0, 0.6, 0.04, 0.01, 0.25);
    const auto gp = bs_greeks(OptionType::Put,  95.0, 100.0, 0.6, 0.04, 0.01, 0.25);
    CHECK(gc.gamma == doctest::Approx(gp.gamma).epsilon(1e-12));
    CHECK(gc.vega  == doctest::Approx(gp.vega).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// Vega via finite difference — sanity check that our analytical vega
// matches the numerical derivative to a few decimal places. This catches
// algebra bugs that a single reference value might miss.
// ---------------------------------------------------------------------------

TEST_CASE("bs_greeks — vega matches central finite difference") {
    const double S = 105.0, K = 100.0, T = 0.4;
    const double r = 0.03, q = 0.01, sigma = 0.28;
    const double h = 1e-5;
    const double up = bs_price(OptionType::Call, S, K, T, r, q, sigma + h);
    const double dn = bs_price(OptionType::Call, S, K, T, r, q, sigma - h);
    const double numerical = (up - dn) / (2.0 * h);
    const auto g = bs_greeks(OptionType::Call, S, K, T, r, q, sigma);
    CHECK(g.vega == doctest::Approx(numerical).epsilon(1e-6));
}
