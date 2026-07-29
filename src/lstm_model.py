"""
lstm_model.py

LSTM forecasting for Tesla closing prices, per Task 2.
All functions validate inputs explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_sequences(values: np.ndarray, window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1D (or column) array of scaled prices into (X, y) sequences of length `window`."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if len(values) <= window:
        raise ValueError(
            f"Need more than {window} observations to build at least one sequence, got {len(values)}."
        )

    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i, 0])
        y.append(values[i, 0])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def build_lstm(window: int = 60, units: int = 50):
    """Build the LSTM Keras model. Imports TensorFlow lazily so this module
    can be imported (and its non-TF functions tested) even without TF installed."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if units <= 0:
        raise ValueError(f"units must be positive, got {units}")

    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for build_lstm/train_lstm but is not installed. "
            "If installing locally fails (e.g. due to a slow connection), run this "
            "step in Google Colab instead, where TensorFlow is pre-installed."
        ) from e

    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(25, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(train_series: pd.Series, window: int = 60, epochs: int = 25, batch_size: int = 32):
    """Scale training data to [0,1], build sequences, and train the LSTM."""
    if train_series.empty:
        raise ValueError("train_series is empty.")
    if train_series.isna().any():
        raise ValueError("train_series contains NaN values.")
    if len(train_series) <= window:
        raise ValueError(f"train_series has {len(train_series)} obs, needs more than window={window}.")

    try:
        from tensorflow import keras
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for train_lstm but is not installed. "
            "Run this step in Google Colab if local installation is unreliable."
        ) from e

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))

    X_train, y_train = make_sequences(scaled, window)
    model = build_lstm(window)

    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)
    return model, scaler


def forecast_lstm_on_test(model, scaler, full_series: pd.Series, test_series: pd.Series, window: int = 60):
    """Walk-forward one-step prediction across the test period."""
    if len(full_series) < len(test_series) + window:
        raise ValueError(
            f"full_series ({len(full_series)} obs) too short for test_series "
            f"({len(test_series)} obs) + window ({window})."
        )

    inputs = full_series[len(full_series) - len(test_series) - window:].values.reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)

    X_test, _ = make_sequences(inputs_scaled, window)
    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled)
    return preds.flatten()


def forecast_lstm_future(model, scaler, full_series: pd.Series, n_future: int, window: int = 60):
    """Iteratively forecast n_future steps beyond the end of full_series."""
    if n_future <= 0:
        raise ValueError(f"n_future must be positive, got {n_future}")
    if len(full_series) < window:
        raise ValueError(f"full_series ({len(full_series)} obs) shorter than window ({window}).")

    last_window = scaler.transform(full_series.values[-window:].reshape(-1, 1)).flatten().tolist()
    preds_scaled = []

    for _ in range(n_future):
        x = np.array(last_window[-window:]).reshape(1, window, 1)
        next_scaled = model.predict(x, verbose=0)[0, 0]
        preds_scaled.append(next_scaled)
        last_window.append(next_scaled)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
    return preds.flatten()


def evaluate_forecast(y_true, y_pred) -> dict:
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