# vol-surface

> Live options vol surface + SVI mispricing scanner. C++ math kernel, Python orchestration, interactive Streamlit dashboard. Runs locally — clone, install, open in your browser.

Pull current options chains, compute the implied volatility for every contract, render the 3D vol surface, fit an SVI smile per expiry, and flag where the market price diverges from the model. Everything math-heavy lives in C++ and is exposed to Python via pybind11 — about **3 million IV solves per second** on the batch path.

This project is built to be useful in two ways: as a tool you run on your machine to look at any US-listed underlying's options surface, and as an honest implementation you can read end-to-end if you're learning quant infrastructure.

---

## What's in here

The C++ kernel handles every numerically demanding piece — Black-Scholes pricing, all five Greeks, Brent's-method implied-volatility solver, and a strike-at-delta inversion. It ships with **408 unit assertions** covering the math, edge cases, and known FP-precision pathologies. The Python side is orchestration: yfinance for chains, FRED (or a bundled Treasury snapshot) for risk-free rates, scipy for SVI fitting, plotly for visualization, and Streamlit for the dashboard.

| Area | What you get |
|---|---|
| **C++ kernel** | Black-Scholes price + Greeks, Brent IV solver, strike-at-delta solver |
| **Python bindings** | Scalar + vectorized numpy-friendly batch APIs (~3M IV solves/sec) |
| **Data layer** | yfinance chains with stratified expiry sampling + Treasury rates + earnings dates |
| **Surface + smile renders** | Plasma-on-black 3D surface, 2D smiles with bid-ask bands & SVI overlay |
| **Term structure & evolution** | ATM IV curve and all-expiries-overlaid view |
| **SVI fits** | Gatheral raw form, Bayesian equity-skew prior, no-arb constraints |
| **Skew metrics** | 25Δ and 10Δ risk reversals & butterflies per expiry |
| **Mispricing scanner** | Three rankable metrics: absolute IV miss, z-score, dollar PnL |
| **Strategy P&L** | Type a position, see expiry payoff + current MTM under the surface |
| **Realized vs implied** | Rolling realized vol time series with current ATM IV |
| **Snapshot button** | One-click 1600×1000 PNG export of any chart |

---

## Quick start (≈ 5 minutes)

You need a C++17 compiler, CMake 3.20+, and Python 3.10+. All free.

**macOS:**
```bash
xcode-select --install                # one time, gets compiler
brew install cmake python             # if you don't have them
```

**Linux:**
```bash
sudo apt install build-essential cmake python3-pip python3-venv
```

**Windows:** Install [Visual Studio 2019+ Build Tools](https://visualstudio.microsoft.com/downloads/) with "Desktop development with C++" — that bundles CMake and the compiler.

Then:

```bash
git clone https://github.com/<your-user>/vol-surface.git
cd vol-surface

pip install -e ".[data,viz,dashboard]"     # builds the C++ kernel and pulls every dep
streamlit run streamlit_app.py             # opens http://localhost:8501
```

Pick a ticker in the sidebar, click through the tabs. That's it.

---

## A tour of the dashboard

Eight tabs, sized for content and reading.

**🌐 3D surface.** The hero visual. Plasma colormap on black — log-moneyness × time-to-expiry × IV. Drag-rotate, zoom, hover for the strike & expiry behind any data point. Toggle the wireframe overlay or the individual contract dots. The cyan vertical line marks at-the-money.

**🌙 Smile slices.** Pick any expiry, see the 2D smile with calls (cyan), puts (pink), the bid-ask spread as a faint shaded band, the SVI fit as a gold line on top, and a residuals strip below showing where market IV diverges from the fit. SVI parameters and an arbitrage-free check appear as metric tiles. Expiries that contain an upcoming earnings announcement get a 📅 prefix in the picker.

**📈 Term structure.** ATM IV vs maturity. Cyan dots are raw market quotes (IV at the strike closest to forward), gold diamonds are the SVI-fitted ATM (the smooth model's "true" ATM). Gaps between them flag event vol the smooth surface can't capture.

**🌈 Smile evolution.** Every per-expiry SVI smile overlaid on a single 2D plot, color-graded purple → yellow as you move from front-month to back-month. Watch how skew flattens and ATM rises as maturity grows.

**⚖️ Skew metrics.** Per-expiry table of 25Δ and 10Δ risk reversals (call IV minus put IV — negative = put skew, the equity standard) and butterflies (½·(call+put) − ATM — positive = wing convexity). Strikes-at-delta come from the C++ delta solver.

**💼 Strategy P&L.** Type a multi-leg position like `long 10 730C 2026-06-18, short 10 740C 2026-06-18`. The chart shows two curves: the dotted pink hockey-stick payoff at expiry and the gold mark-to-market curve under the current surface as you sweep the underlying price. Per-leg context table tells you which chain contracts each leg matched.

**📊 Realized vs implied.** Rolling annualized realized vol time series from the underlying's history, with the current ATM IV pinned as a gold diamond at the matching expiry. The chronic gap is the volatility risk premium — one of the cleanest empirical regularities in equity options.

**📋 Top deviations.** Ranked table of contracts whose market IV is furthest from SVI. Sort by absolute IV miss (most content-meaningful), spread-normalised z-score (statistically rigorous but noisy on tight markets), or dollar P&L impact. Download as CSV.

Every chart tab has a **📷 Snapshot for socials** button that downloads a 1600×1000 PNG.

---

## Optional: live risk-free rates from FRED

By default the project ships with a recent US Treasury yield-curve snapshot in `python/vol_surface/data/snapshots/treasury_curve.csv`, so it works fully offline. If you want current rates, get a free FRED API key and set it:

```bash
export FREDAPI_KEY="your_key_here"     # macOS / Linux
$env:FREDAPI_KEY = "your_key_here"     # Windows PowerShell
```

The library auto-detects the env var and switches sources. Force one or the other with `RiskFreeCurve(source='snapshot')` / `RiskFreeCurve(source='fred')`.

---

## Run the C++ tests directly

If you want to exercise the kernel without going through Python:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

You should see 24 test cases / 408 assertions, all passing.

---

## Architecture

```
vol-surface/
├── cpp/
│   ├── include/vol/         public C++ headers
│   ├── src/                 pricing, IV solver, delta solver
│   ├── tests/               doctest unit tests (408 assertions)
│   └── bindings/            pybind11 → Python module
├── python/vol_surface/
│   ├── data/                yfinance chains, FRED rates, earnings
│   ├── svi/                 raw-SVI fitter (Gatheral + Bayesian prior)
│   ├── scanner/             SVI-residual mispricing scanner
│   ├── analytics/           skew metrics, realized vol
│   ├── strategy/            position parser + payoff engine
│   └── viz/                 plotly renderers (surface, smile, term, evolution)
├── examples/                command-line demos
├── streamlit_app.py         the dashboard
├── pyproject.toml           scikit-build-core build config
└── CMakeLists.txt           top-level CMake
```

The C++ side is pure math — no I/O, no dependencies beyond the C++17 standard library + pybind11 for the bindings + doctest for tests. The Python side handles everything network-, file-, and visualization-related.

---

## License

MIT — see [LICENSE](LICENSE).
