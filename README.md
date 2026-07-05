# Portfolio Optimization — GMF Investments (Week 9 Challenge)

Time series forecasting and Modern Portfolio Theory applied to a 3-asset
portfolio (TSLA, BND, SPY), 2015-01-01 to 2026-06-30.

## Status

| Task | Status |
|---|---|
| Task 1 — Data extraction, cleaning, EDA, stationarity, risk metrics | ✅ Complete |
| Task 2 — ARIMA and LSTM forecasting models | ✅ Complete (LSTM selected as best model) |
| Task 3 — Future forecast with confidence intervals | ⬜ Not started |
| Task 4 — Efficient Frontier / MPT optimization | ⬜ Not started |
| Task 5 — Strategy backtest vs. benchmark | ⬜ Not started |

## Key Results So Far

**Stationarity (ADF test, TSLA):**
- Close price: statistic -1.04, p-value 0.739 → non-stationary
- Daily returns: statistic -55.15, p-value ≈ 0.000 → stationary
- Confirms d=1 differencing is appropriate for ARIMA

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

LSTM outperforms ARIMA substantially on all three metrics and is the model
carried forward into Task 3 (forecasting) and Task 4 (expected return input).

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on TensorFlow:** if your local machine has a slow/unstable internet
> connection, the ~377 MB TensorFlow download can fail repeatedly. If so, run
> the Task 2 LSTM notebook in [Google Colab](https://colab.research.google.com)
> instead (TensorFlow comes pre-installed there) — everything else can run locally.

> **Note on yfinance:** needs outbound access to `query1.finance.yahoo.com` /
> `query2.finance.yahoo.com`. If blocked by your network/firewall, run locally
> or in Colab.

## Project Structure

```
portfolio-optimization/
├── data/processed/       # Cleaned CSVs and saved plots (gitignored, regenerate locally)
├── notebooks/
│   ├── task1_eda.ipynb          # Task 1: extraction, cleaning, EDA, stationarity, risk metrics
│   ├── task2_modeling.ipynb     # Task 2: ARIMA vs LSTM forecasting
│   ├── task3_forecast.ipynb     # Task 3: future forecast + confidence intervals (TODO)
│   ├── task4_portfolio.ipynb    # Task 4: Efficient Frontier / MPT optimization (TODO)
│   └── task5_backtest.ipynb     # Task 5: strategy backtest vs benchmark (TODO)
├── src/
│   ├── data_loader.py     # YFinance extraction + cleaning
│   ├── eda.py              # Returns, rolling stats, ADF test, VaR, Sharpe
│   ├── arima_model.py      # auto_arima fitting, forecasting, evaluation
│   └── lstm_model.py       # LSTM sequence prep, training, forecasting, evaluation
├── tests/                  # Unit tests (pytest)
├── scripts/                 # Optional CLI entry points
└── requirements.txt
```

## Running the pipeline

1. `notebooks/task1_eda.ipynb` — pulls raw data via `yfinance`, cleans it, saves
   `data/processed/adj_close_combined.csv`, runs EDA + ADF stationarity tests + VaR/Sharpe.
2. `notebooks/task2_modeling.ipynb` — reads the cleaned CSV, splits chronologically
   (train ≤2024, test 2025-2026), fits ARIMA (`pmdarima.auto_arima`) and an LSTM,
   compares MAE/RMSE/MAPE.
3. Tasks 3–5 build on the LSTM (winning model from Task 2): future forecast →
   expected return input for MPT → Efficient Frontier → backtest vs 60/40 SPY/BND benchmark.

## Reports

- `GMF_Interim_Report.docx` — interim submission report (Tasks 1–2)
- Final investment memo (Tasks 1–5) due at final submission

## Key dates
- Interim submission: Sun 05 Jul 2026, 20:00 UTC — Task 1 complete + Task 2 in progress
- Final submission: Tue 07 Jul 2026, 20:00 UTC — all tasks + investment memo
