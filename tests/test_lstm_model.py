import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lstm_model import make_sequences, evaluate_forecast, forecast_lstm_on_test, forecast_lstm_future


def test_make_sequences_basic():
    values = np.arange(100).reshape(-1, 1).astype(float)
    X, y = make_sequences(values, window=10)
    assert X.shape == (90, 10, 1)
    assert y.shape == (90,)


def test_make_sequences_1d_input():
    values = np.arange(50).astype(float)
    X, y = make_sequences(values, window=5)
    assert X.shape == (45, 5, 1)


def test_make_sequences_invalid_window_raises():
    values = np.arange(50).astype(float)
    with pytest.raises(ValueError, match="positive"):
        make_sequences(values, window=0)


def test_make_sequences_too_few_observations_raises():
    values = np.arange(5).astype(float)
    with pytest.raises(ValueError, match="Need more than"):
        make_sequences(values, window=10)


def test_evaluate_forecast_basic():
    result = evaluate_forecast([100, 110, 120], [102, 108, 121])
    assert set(result.keys()) == {"MAE", "RMSE", "MAPE"}


def test_evaluate_forecast_zero_actual_raises():
    with pytest.raises(ValueError, match="zero"):
        evaluate_forecast([0, 1, 2], [1, 1, 1])


def test_forecast_lstm_on_test_insufficient_history_raises():
    full_series = pd.Series(np.arange(20).astype(float))
    test_series = pd.Series(np.arange(15).astype(float))
    with pytest.raises(ValueError, match="too short"):
        forecast_lstm_on_test(model=None, scaler=None, full_series=full_series,
                               test_series=test_series, window=10)


def test_forecast_lstm_future_invalid_n_future_raises():
    full_series = pd.Series(np.arange(100).astype(float))
    with pytest.raises(ValueError, match="positive"):
        forecast_lstm_future(model=None, scaler=None, full_series=full_series, n_future=0)


def test_forecast_lstm_future_short_series_raises():
    full_series = pd.Series(np.arange(5).astype(float))
    with pytest.raises(ValueError, match="shorter than window"):
        forecast_lstm_future(model=None, scaler=None, full_series=full_series, n_future=10, window=60)


def test_build_lstm_missing_tensorflow_gives_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("No module named 'tensorflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from lstm_model import build_lstm
    with pytest.raises(ImportError, match="Google Colab"):
        build_lstm(window=60)