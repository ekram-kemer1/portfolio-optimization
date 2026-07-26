"""
portfolio.py

Modern Portfolio Theory utilities for Task 4: building an expected-returns
vector that blends a model forecast (TSLA) with historical returns (other
assets), computing the covariance matrix, and optimizing the Efficient
Frontier for Max Sharpe / Min Volatility portfolios.

All public functions validate their inputs explicitly and raise informative
errors rather than failing deep inside a third-party library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models


def build_expected_returns(
    forecast_asset: str,
    forecast_final_price: float,
    forecast_last_known_price: float,
    forecast_horizon_days: int,
    historical_returns: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """
    Build an expected-returns vector for portfolio optimization.

    `forecast_asset` gets its expected return annualized from a forecast
    (e.g. an LSTM's future price prediction). Every other column in
    `historical_returns` gets its expected return from its historical
    annualized mean daily return.
    """
    if forecast_asset not in historical_returns.columns:
        raise ValueError(
            f"'{forecast_asset}' not found in historical_returns columns: "
            f"{list(historical_returns.columns)}"
        )
    if forecast_last_known_price <= 0:
        raise ValueError(f"forecast_last_known_price must be positive, got {forecast_last_known_price}")
    if forecast_horizon_days <= 0:
        raise ValueError(f"forecast_horizon_days must be positive, got {forecast_horizon_days}")
    if historical_returns.isna().any().any():
        raise ValueError(
            "historical_returns contains NaN values; clean the data before "
            "computing expected returns."
        )

    total_return = (forecast_final_price / forecast_last_known_price) - 1
    forecast_annual_return = (1 + total_return) ** (trading_days_per_year / forecast_horizon_days) - 1

    other_assets = [c for c in historical_returns.columns if c != forecast_asset]
    expected = {a: historical_returns[a].mean() * trading_days_per_year for a in other_assets}
    expected[forecast_asset] = forecast_annual_return

    return pd.Series(expected)[historical_returns.columns]


def compute_covariance(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the annualized sample covariance matrix from a price DataFrame."""
    if price_df.isna().any().any():
        raise ValueError("price_df contains NaNs; clean the data before computing covariance.")
    if price_df.shape[1] < 2:
        raise ValueError(f"Need at least 2 assets to compute a covariance matrix, got {price_df.shape[1]}")
    if price_df.shape[0] < 30:
        raise ValueError(
            f"Only {price_df.shape[0]} observations provided; at least 30 are recommended "
            "for a stable covariance estimate."
        )
    return risk_models.sample_cov(price_df, returns_data=False)


def _validate_optimizer_inputs(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> None:
    if not expected_returns.index.equals(cov_matrix.index):
        raise ValueError(
            "expected_returns and cov_matrix must have matching, identically-ordered "
            f"asset indices. Got {list(expected_returns.index)} vs {list(cov_matrix.index)}"
        )
    if expected_returns.isna().any():
        raise ValueError("expected_returns contains NaN values.")
    if expected_returns.max() <= 0:
        raise ValueError(
            "All expected returns are zero or negative; max_sharpe optimization requires "
            "at least one asset with a positive expected return above the risk-free rate."
        )


def optimize_max_sharpe(
    expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02
) -> tuple[dict, tuple[float, float, float]]:
    """Returns (weights_dict, (expected_return, volatility, sharpe_ratio))."""
    _validate_optimizer_inputs(expected_returns, cov_matrix)
    ef = EfficientFrontier(expected_returns, cov_matrix)
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    return dict(weights), performance


def optimize_min_volatility(
    expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02
) -> tuple[dict, tuple[float, float, float]]:
    """Returns (weights_dict, (expected_return, volatility, sharpe_ratio))."""
    if expected_returns.isna().any():
        raise ValueError("expected_returns contains NaN values.")
    if not expected_returns.index.equals(cov_matrix.index):
        raise ValueError("expected_returns and cov_matrix must have matching asset indices.")
    ef = EfficientFrontier(expected_returns, cov_matrix)
    weights = ef.min_volatility()
    performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    return dict(weights), performance


def build_efficient_frontier(
    expected_returns: pd.Series, cov_matrix: pd.DataFrame, n_points: int = 100
) -> tuple[list, list]:
    """
    Scan target returns between the min and max expected returns to trace
    the Efficient Frontier manually (avoids relying on pypfopt.plotting,
    which can break on unsupported matplotlib style names).
    """
    _validate_optimizer_inputs(expected_returns, cov_matrix)

    target_returns = np.linspace(expected_returns.min(), expected_returns.max() * 0.99, n_points)
    frontier_volatility, frontier_returns = [], []

    for target in target_returns:
        try:
            ef = EfficientFrontier(expected_returns, cov_matrix)
            ef.efficient_return(target_return=target)
            ret, vol, _ = ef.portfolio_performance()
            frontier_returns.append(ret)
            frontier_volatility.append(vol)
        except Exception:
            continue

    if not frontier_returns:
        raise RuntimeError(
            "Could not compute any feasible points on the efficient frontier; "
            "check expected_returns and cov_matrix for degenerate inputs."
        )

    return frontier_volatility, frontier_returns