# Portfolio Optimization — GMF Investments (Week 9 Challenge + Week 12 Capstone)

Time series forecasting and Modern Portfolio Theory applied to a 3-asset
portfolio (TSLA, BND, SPY), 2015-01-01 to 2026-06-30.

**Week 12 Capstone Update:** this repo was hardened from its original Week 9
submission per grader feedback — notebook logic was extracted into tested
`src/` modules, input validation and error handling were added throughout,
and automated test coverage reporting was wired into CI. See "Capstone
Improvements" below for full details.

## Status

| Task | Status |
|---|---|
| Task 1 — Data extraction, cleaning, EDA, stationarity, risk metrics | Complete |
| Task 2 — ARIMA and LSTM forecasting models | Complete (LSTM selected as best model) |
| Task 3 — Future forecast with confidence intervals | Complete |
| Task 4 — Efficient Frontier / MPT optimization | Complete |
| Task 5 — Strategy backtest vs. benchmark (single window + walk-forward) | Complete |

## Key Results

**Stationarity (ADF test, TSLA):**
- Close price: statistic -1.04, p-value 0.739 → non-stationary
- Daily returns: statistic -55.15, p-value ~0.000 → stationary

**Risk metrics (annualized, 2% risk-free rate):**

| Asset | Ann. Return | Ann. Volatility | VaR (95%, daily) | Sharpe |
|---|---|---|---|---|
| TSLA | 43.77% | 56.13% | -5.11% | 0.744 |
| BND | 1.92% | 5.22% | -0.47% | -0.015 |
| SPY | 13.91% | 17.33% | -1.64% | 0.687 |

**Model comparison (TSLA, test period 2025-01 to 2026-06):**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| ARIMA | 54.47 | 70.82 | 17.25% |
| LSTM | 24.09 | 27.88 | 6.29% |

**Portfolio optimization result:** TSLA's LSTM-forecasted 9-month return was
negative (-44.2% annualized), so both the Max Sharpe and Min Volatility
portfolios exclude TSLA entirely. Max Sharpe = 100% SPY (Sharpe 0.687);
Min Volatility = 94.5% BND / 5.5% SPY (Sharpe 0.113).

**Backtest results:**
- Single-window (2025-01 to 2026-06): strategy total return 28.60% vs.
  benchmark 20.56%, but strategy Sharpe (0.903) < benchmark Sharpe (0.979) —
  higher absolute return came with higher risk.
- Walk-forward (25 rolling 6-month windows, 2020–2026): strategy outperformed
  the 60/40 benchmark in 22/25 windows (88%), underperforming specifically
  during the COVID-19 crash and the 2022 rate-hiking bear market.

## Capstone Improvements (Week 12)

Per interim grader feedback ("harden the codebase: move notebook logic into
reusable src/ modules, broaden error handling, and add tests"), the
following was implemented:

1. **`src/portfolio.py`** — Task 4's expected-returns, covariance, and
   Efficient Frontier optimization logic extracted from the notebook into a
   tested module with explicit input validation (e.g. raises a clear error
   if all expected returns are negative, rather than an opaque failure deep
   inside PyPortfolioOpt).
2. **`src/data_loader.py`, `src/arima_model.py`, `src/lstm_model.py`** —
   added input validation and explicit error handling throughout (invalid
   date ranges, empty API responses, NaN data, mismatched array shapes,
   missing TensorFlow installation with an actionable message pointing to
   Google Colab as a fallback).
3. **`src/backtest.py`** — extended with `simulate_with_rebalancing()`
   (configurable periodic rebalancing with transaction costs) and
   `walk_forward_backtest()` (tests the strategy across multiple rolling
   windows instead of a single static period), directly addressing the
   "only one backtest window" limitation flagged in the original Task 5
   submission.
4. **Test suite** — 62 unit tests across all 5 `src/` modules, achieving
   **77% overall code coverage** (`pytest --cov=src`), including edge cases
   (negative expected returns, empty data, mismatched weights, missing
   dependencies).
5. **CI coverage reporting** — `.github/workflows/unittests.yml` runs the
   full test suite with coverage on every push/PR and uploads a coverage
   report as a build artifact. Tests run as isolated per-module `pytest`
   invocations (`--cov-append`) rather than one combined run, to avoid a
   native-library conflict between TensorFlow, pmdarima, and cvxpy when
   loaded together in a single Windows process.

### Coverage by module

| Module | Coverage |
|---|---|
| `arima_model.py` | 75% |
| `backtest.py` | 94% |
| `data_loader.py` | 72% |
| `lstm_model.py` | 53%* |
| `portfolio.py` | 89% |
| **TOTAL** | **77%** |

*`lstm_model.py`'s lower coverage reflects that its core training/prediction
logic requires a live TensorFlow installation to execute; all input
validation and guard-clause logic is fully tested, but the TensorFlow-
dependent code paths are only exercised when actually training a model
(e.g. in Google Colab).

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on TensorFlow:** if your local machine has a slow or unstable
> internet connection, the ~377 MB TensorFlow download can fail repeatedly.
> Run the Task 2 LSTM notebook in [Google Colab](https://colab.research.google.com)
> instead (TensorFlow comes pre-installed there) — everything else can run locally.

> **Note on yfinance:** needs outbound access to `query1.finance.yahoo.com` and
> `query2.finance.yahoo.com`. If blocked by your network or firewall, run
> locally or in Colab instead.

> **Note on cvxpy/numpy version conflict:** this project pins `numpy==1.26.4`
> (required by TensorFlow 2.16) and `cvxpy<1.5` (newer cvxpy versions require
> numpy 2.x). If you see `ModuleNotFoundError: No module named
> 'numpy.lib.array_utils'`, reinstall cvxpy with `pip install "cvxpy<1.5"
> --no-cache-dir`.

## Running Tests

```bash
# Run all tests (may hit a native library conflict on Windows if run together —
# see note below)
pytest tests/ -v

# Recommended on Windows: run each module's tests in its own process,
# accumulating coverage, to avoid a TensorFlow/pmdarima/cvxpy conflict
pytest tests/test_data_loader.py --cov=src --cov-append -v
pytest tests/test_arima_model.py --cov=src --cov-append -v
pytest tests/test_lstm_model.py --cov=src --cov-append -v
pytest tests/test_portfolio.py --cov=src --cov-append -v
pytest tests/test_backtest.py --cov=src --cov-append -v
coverage report -m
```

## Project Structure
portfolio-optimization/
├── data/processed/ # Cleaned CSVs, saved plots, forecast/weight artifacts
├── notebooks/
│ ├── task1_eda.ipynb # Task 1: extraction, cleaning, EDA, stationarity, risk metrics
│ ├── task2_modeling.ipynb # Task 2: ARIMA vs LSTM forecasting
│ ├── task3_forecast.ipynb # Task 3: future forecast + confidence intervals
│ ├── task4_portfolio.ipynb # Task 4: Efficient Frontier / MPT optimization
│ └── task5_backtest.ipynb # Task 5: single-window + walk-forward backtest
├── src/
│ ├── data_loader.py # YFinance extraction + cleaning (validated)
│ ├── eda.py # Returns, rolling stats, ADF test, VaR, Sharpe
│ ├── arima_model.py # auto_arima fitting, forecasting, evaluation (validated)
│ ├── lstm_model.py # LSTM sequence prep, training, forecasting, evaluation (validated)
│ ├── portfolio.py # Expected returns, covariance, Efficient Frontier (validated, tested)
│ └── backtest.py # Static + walk-forward backtesting, rebalancing (validated, tested)
├── tests/ # 62 unit tests, 77% coverage across src/
├── scripts/ # Optional CLI entry points
├── .coveragerc # Coverage configuration
└── requirements.txt
## Running the Pipeline

1. `notebooks/task1_eda.ipynb` — pulls raw data via `yfinance`, cleans it, saves
   `data/processed/adj_close_combined.csv`, runs EDA + ADF stationarity tests + VaR/Sharpe.
2. `notebooks/task2_modeling.ipynb` — reads the cleaned CSV, splits chronologically
   (train ≤2024, test 2025-2026), fits ARIMA (`pmdarima.auto_arima`) and an LSTM,
   compares MAE/RMSE/MAPE.
3. `notebooks/task3_forecast.ipynb` — retrains the LSTM on full history, forecasts
   9 months forward with confidence intervals, saves outputs for Task 4.
4. `notebooks/task4_portfolio.ipynb` — uses `src/portfolio.py` to build expected
   returns (LSTM forecast for TSLA, historical means for BND/SPY), computes the
   covariance matrix, and optimizes the Efficient Frontier.
5. `notebooks/task5_backtest.ipynb` — uses `src/backtest.py` to backtest the
   optimal portfolio against a 60/40 SPY/BND benchmark, both as a single static
   window and via 25-window walk-forward validation.

## Reports

- `GMF_Interim_Report.docx` — interim submission report (Tasks 1–2)
- `GMF_Final_Investment_Memo.docx` — final submission report (Tasks 1–5)
- Capstone improvement report (Week 12) — documents the hardening work above

## Key Dates

- Week 9 interim submission: Sun 05 Jul 2026 — Task 1 complete + Task 2 in progress
- Week 9 final submission: Tue 07 Jul 2026 — all tasks + investment memo
- Week 12 capstone submission: improvement plan + hardened codebase