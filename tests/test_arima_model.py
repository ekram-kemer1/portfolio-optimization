import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arima_model import chronological_split, fit_auto_arima, forecast_arima, evaluate_forecast


def _sample_series(n=200, seed=1):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return pd.Series(prices, index=dates)


def test_chronological_split_basic():
    series = _sample_series(200)
    split_date = series.index[150].strftime("%Y-%m-%d")
    train, test = chronological_split(series, split_date)
    assert len(train) + len(test) == len(series)
    assert train.index.max() < test.index.min()


def test_chronological_split_empty_series_raises():
    with pytest.raises(ValueError, match="empty"):
        chronological_split(pd.Series(dtype=float), "2020-01-01")


def test_chronological_split_bad_index_raises():
    series = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        chronological_split(series, "2020-01-01")


def test_chronological_split_date_out_of_range_raises():
    series = _sample_series(200)
    with pytest.raises(ValueError, match="date range"):
        chronological_split(series, "2050-01-01")


def test_chronological_split_too_few_train_obs_raises():
    series = _sample_series(50)
    split_date = series.index[5].strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="at least 30"):
        chronological_split(series, split_date)


def test_fit_auto_arima_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        fit_auto_arima(pd.Series(dtype=float))


def test_fit_auto_arima_nan_raises():
    series = pd.Series([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        fit_auto_arima(series)


def test_forecast_arima_invalid_n_periods_raises():
    class DummyModel:
        def predict(self, n_periods, return_conf_int=False):
            return np.zeros(n_periods), None
    with pytest.raises(ValueError, match="positive"):
        forecast_arima(DummyModel(), n_periods=0)


def test_evaluate_forecast_basic():
    y_true = np.array([100, 110, 120])
    y_pred = np.array([102, 108, 121])
    result = evaluate_forecast(y_true, y_pred)
    assert set(result.keys()) == {"MAE", "RMSE", "MAPE"}
    assert result["MAE"] > 0


def test_evaluate_forecast_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        evaluate_forecast([1, 2, 3], [1, 2])


def test_evaluate_forecast_zero_actual_raises():
    with pytest.raises(ValueError, match="zero"):
        evaluate_forecast([0, 1, 2], [1, 1, 1])


def test_evaluate_forecast_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        evaluate_forecast([], [])