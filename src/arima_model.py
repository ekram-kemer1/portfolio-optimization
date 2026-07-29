"""
arima_model.py

ARIMA/SARIMA forecasting for Tesla closing prices, per Task 2.
All functions validate inputs explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error


def chronological_split(series: pd.Series, split_date: str) -> tuple[pd.Series, pd.Series]:
    """Split a series into train/test chronologically at split_date (exclusive on test start)."""
    if series.empty:
        raise ValueError("series is empty.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex.")

    try:
        split_ts = pd.Timestamp(split_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid split_date: {e}")

    if split_ts <= series.index.min() or split_ts >= series.index.max():
        raise ValueError(
            f"split_date {split_date} must fall strictly between the series' "
            f"date range ({series.index.min().date()} to {series.index.max().date()})."
        )

    train = series[series.index < split_ts]
    test = series[series.index >= split_ts]

    if len(train) < 30:
        raise ValueError(f"Training set only has {len(train)} observations; need at least 30.")
    if len(test) == 0:
        raise ValueError("Test set is empty after split.")

    return train, test


def fit_auto_arima(train: pd.Series, seasonal: bool = False, m: int = 5):
    """Fit auto_arima to find optimal (p, d, q) [and (P, D, Q, m) if seasonal]."""
    if train.empty:
        raise ValueError("train series is empty.")
    if train.isna().any():
        raise ValueError("train series contains NaN values.")

    try:
        model = pm.auto_arima(
            train, seasonal=seasonal, m=m if seasonal else 1,
            trace=False, error_action="ignore", suppress_warnings=True, stepwise=True,
        )
    except Exception as e:
        raise RuntimeError(f"auto_arima failed to fit: {e}") from e

    return model


def forecast_arima(model, n_periods: int, return_conf_int: bool = True):
    if n_periods <= 0:
        raise ValueError(f"n_periods must be positive, got {n_periods}")

    if return_conf_int:
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
        return forecast, conf_int
    forecast = model.predict(n_periods=n_periods)
    return forecast, None


def evaluate_forecast(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, MAPE between true and predicted values."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("y_true/y_pred are empty.")
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values; MAPE is undefined for zero actuals.")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}