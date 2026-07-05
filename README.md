# Portfolio Optimization — GMF Investments (Week 9 Challenge)

Time series forecasting and Modern Portfolio Theory applied to a 3-asset
portfolio (TSLA, BND, SPY), 2015-01-01 to 2026-06-30.

## Status

| Task | Status |
|---|---|
| Task 1 — Data extraction, cleaning, EDA, stationarity, risk metrics | Complete |
| Task 2 — ARIMA and LSTM forecasting models | Complete (LSTM selected as best model) |
| Task 3 — Future forecast with confidence intervals | Not started |
| Task 4 — Efficient Frontier / MPT optimization | Not started |
| Task 5 — Strategy backtest vs. benchmark | Not started |

## Key Results So Far

**Stationarity (ADF test, TSLA):**
- Close price: statistic -1.04, p-value 0.739 → non-stationary
- Daily returns: statistic -55.15, p-value ~0.000 → stationary
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
## Project Structure
## Running the Pipeline

1. `notebooks/task1_eda.ipynb` — pulls raw data via `yfinance`, cleans it, saves
   `data/processed/adj_close_combined.csv`, runs EDA + ADF stationarity tests + VaR/Sharpe.
2. `notebooks/task2_modeling.ipynb` — reads the cleaned CSV, splits chronologically
   (train ≤2024, test 2025-2026), fits ARIMA (`pmdarima.auto_arima`) and an LSTM,
   compares MAE/RMSE/MAPE.
3. Tasks 3–5 build on the LSTM (winning model from Task 2): future forecast →
   expected return input for MPT → Efficient Frontier → backtest vs 60/40 SPY/BND benchmark.


