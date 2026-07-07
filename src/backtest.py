"""Backtesting utilities for Task 5."""
from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_daily_returns(returns_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted daily portfolio returns. Validates weights sum to ~1."""
    missing = set(returns_df.columns) - set(weights.keys())
    if missing:
        raise ValueError(f"Missing weights for columns: {missing}")

    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0, atol=1e-3):
        raise ValueError(f"Weights must sum to 1.0, got {total_weight:.4f}")

    w = np.array([weights[c] for c in returns_df.columns])
    values = returns_df.values @ w
    return pd.Series(values, index=returns_df.index)


def performance_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """Total return, annualized return, Sharpe Ratio, and max drawdown."""
    if daily_returns.empty:
        raise ValueError("daily_returns is empty.")

    cumulative = (1 + daily_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_years = len(daily_returns) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    excess = daily_returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() != 0 else np.nan

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
   