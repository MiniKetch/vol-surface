// vol_kernel/python_module.cpp — pybind11 bindings.
//
// Design notes for someone new to pybind11:
//
//   * py::class_ binds a C++ struct/class. Members exposed via .def_readonly
//     / .def_readwrite / .def() become Python attributes / methods.
//   * py::enum_ exposes a C++ scoped enum as a Python class with
//     constants. We choose .export_values() OFF so users must write
//     `OptionType.Call`, never just `Call` — keeps the namespace clean.
//   * std::optional<double> automatically maps to Python float | None.
//     No glue needed.
//   * For batch operations, we use py::array_t<double> (a numpy view)
//     directly. The .request() call gives us a buffer_info struct with
//     .ptr we can iterate over without GIL contention or copies.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // std::optional, std::vector

#include "vol/black_scholes.hpp"
#include "vol/delta_solver.hpp"
#include "vol/iv_solver.hpp"
#include "vol/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;
using vol::OptionType;
using vol::Greeks;

namespace {

// Helper: turn an int8 / int code into OptionType. We accept 0 = Call,
// 1 = Put for batch APIs since numpy doesn't carry our enum nicely.
OptionType decode_type(std::int8_t code) {
    if (code == 0) return OptionType::Call;
    if (code == 1) return OptionType::Put;
    throw std::invalid_argument(
        "type code must be 0 (Call) or 1 (Put)");
}

// ----------------------------------------------------------------------------
// Vectorized BS price.
//
// Inputs are numpy arrays of equal length (or scalars; numpy broadcasts).
// We loop in C++ — releasing the GIL — so calling this from Python on
// 100k contracts costs one Python call instead of 100k.
// ----------------------------------------------------------------------------
py::array_t<double> bs_price_batch(
    py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> types,
    py::array_t<double,        py::array::c_style | py::array::forcecast> S,
    py::array_t<double,        py::array::c_style | py::array::forcecast> K,
    py::array_t<double,        py::array::c_style | py::array::forcecast> T,
    py::array_t<double,        py::array::c_style | py::array::forcecast> r,
    py::array_t<double,        py::array::c_style | py::array::forcecast> q,
    py::array_t<double,        py::array::c_style | py::array::forcecast> sigma) {

    const auto n = types.size();
    if (S.size() != n || K.size() != n || T.size() != n
        || r.size() != n || q.size() != n || sigma.size() != n) {
        throw std::invalid_argument("All input arrays must have the same length");
    }

    py::array_t<double> out(n);

    // Borrow read-only pointers. The arrays are guaranteed contiguous
    // by the c_style template flag above.
    const auto* t_ptr = types.data();
    const auto* S_ptr = S.data();
    const auto* K_ptr = K.data();
    const auto* T_ptr = T.data();
    const auto* r_ptr = r.data();
    const auto* q_ptr = q.data();
    const auto* sg_ptr = sigma.data();
    auto* out_ptr = out.mutable_data();

    // Release the GIL so Python threads can run while we compute.
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
        const auto type = (t_ptr[i] == 0) ? OptionType::Call : OptionType::Put;
        out_ptr[i] = vol::bs_price(type, S_ptr[i], K_ptr[i], T_ptr[i],
                                   r_ptr[i], q_ptr[i], sg_ptr[i]);
    }
    return out;
}

// ----------------------------------------------------------------------------
// Vectorized implied vol.
//
// Returns NaN for any contract where IV could not be solved — this is
// the numpy-friendly equivalent of std::optional<double>. NaN propagates
// cleanly through downstream analytics; users can `np.isnan(ivs)` to
// filter.
// ----------------------------------------------------------------------------
py::array_t<double> implied_vol_batch(
    py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> types,
    py::array_t<double,        py::array::c_style | py::array::forcecast> market_price,
    py::array_t<double,        py::array::c_style | py::array::forcecast> S,
    py::array_t<double,        py::array::c_style | py::array::forcecast> K,
    py::array_t<double,        py::array::c_style | py::array::forcecast> T,
    py::array_t<double,        py::array::c_style | py::array::forcecast> r,
    py::array_t<double,        py::array::c_style | py::array::forcecast> q) {

    const auto n = types.size();
    if (market_price.size() != n || S.size() != n || K.size() != n
        || T.size() != n || r.size() != n || q.size() != n) {
        throw std::invalid_argument("All input arrays must have the same length");
    }

    py::array_t<double> out(n);
    const auto* t_ptr  = types.data();
    const auto* mp_ptr = market_price.data();
    const auto* S_ptr  = S.data();
    const auto* K_ptr  = K.data();
    const auto* T_ptr  = T.data();
    const auto* r_ptr  = r.data();
    const auto* q_ptr  = q.data();
    auto* out_ptr      = out.mutable_data();

    py::gil_scoped_release release;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (py::ssize_t i = 0; i < n; ++i) {
        const auto type = (t_ptr[i] == 0) ? OptionType::Call : OptionType::Put;
        const auto iv = vol::implied_vol(type, mp_ptr[i], S_ptr[i], K_ptr[i],
                                         T_ptr[i], r_ptr[i], q_ptr[i]);
        out_ptr[i] = iv.value_or(nan);
    }
    return out;
}

}  // namespace

// ----------------------------------------------------------------------------
// Module entry point. The macro creates `PyInit__vol_kernel` which Python's
// import machinery looks for. The module name MUST match the .so / .pyd
// filename (`_vol_kernel`).
// ----------------------------------------------------------------------------
PYBIND11_MODULE(_vol_kernel, m) {
    m.doc() = "Vol Surface — C++ math kernel: pricing, Greeks, implied vol.";

    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put",  OptionType::Put);

    py::class_<Greeks>(m, "Greeks",
                       "Standard Black-Scholes Greeks (raw mathematical units).")
        .def_readonly("delta", &Greeks::delta)
        .def_readonly("gamma", &Greeks::gamma)
        .def_readonly("vega",  &Greeks::vega,
                      "dPrice/dSigma per 1.0 vol — divide by 100 for per vol point.")
        .def_readonly("theta", &Greeks::theta,
                      "Per year — divide by 365 for daily theta.")
        .def_readonly("rho",   &Greeks::rho,
                      "Per 1.0 of rate — divide by 100 for per bp.")
        .def("__repr__", [](const Greeks& g) {
            return "<Greeks delta=" + std::to_string(g.delta)
                 + " gamma=" + std::to_string(g.gamma)
                 + " vega="  + std::to_string(g.vega)
                 + " theta=" + std::to_string(g.theta)
                 + " rho="   + std::to_string(g.rho) + ">";
        });

    // ---- scalar pricing ----
    m.def("bs_price",
          [](OptionType t, double S, double K, double T,
             double r, double q, double sigma) {
              return vol::bs_price(t, S, K, T, r, q, sigma);
          },
          py::arg("type"), py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          "Black-Scholes price for a European option on a continuous-yield underlying.");

    m.def("bs_greeks",
          [](OptionType t, double S, double K, double T,
             double r, double q, double sigma) {
              return vol::bs_greeks(t, S, K, T, r, q, sigma);
          },
          py::arg("type"), py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          "All five Greeks in one call (delta, gamma, vega, theta, rho).");

    m.def("implied_vol",
          [](OptionType t, double market_price, double S, double K,
             double T, double r, double q,
             double tol, int max_iter) {
              return vol::implied_vol(t, market_price, S, K, T, r, q,
                                      tol, max_iter);
          },
          py::arg("type"), py::arg("market_price"),
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"),
          py::arg("tol") = 1e-8, py::arg("max_iter") = 100,
          "Solve for σ via Brent. Returns None if no IV exists "
          "(price below intrinsic, expired, etc.).");

    // ---- batch / vectorized ----
    m.def("bs_price_batch", &bs_price_batch,
          py::arg("types"), py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"), py::arg("sigma"),
          "Vectorized BS price. `types` is an int8 array (0=Call, 1=Put). "
          "All arrays must have the same length.");

    m.def("implied_vol_batch", &implied_vol_batch,
          py::arg("types"), py::arg("market_price"),
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"),
          "Vectorized IV solver. Failures (price below intrinsic, etc.) "
          "return NaN — use np.isnan() to filter.");

    m.def("strike_at_delta",
          [](OptionType t, double target_delta,
             double S, double T, double r, double q, double sigma,
             double tol, int max_iter) {
              return vol::strike_at_delta(
                  t, target_delta, S, T, r, q, sigma, tol, max_iter);
          },
          py::arg("type"), py::arg("target_delta"),
          py::arg("S"), py::arg("T"), py::arg("r"), py::arg("q"),
          py::arg("sigma"),
          py::arg("tol") = 1e-8, py::arg("max_iter") = 100,
          "Find the strike such that BS-delta equals target_delta. "
          "Sign of target must match option type "
          "(positive for calls, negative for puts). "
          "Returns None on out-of-range targets or non-convergence.");
}
