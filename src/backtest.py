"""
backtest.py

Backtesting utilities for Task 5, extended for the Week 12 capstone with:
  - static single-window backtesting (original Task 5 behavior)
  - configurable periodic rebalancing with transaction costs
  - walk-forward backtesting across multiple rolling windows

All public functions validate their inputs explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_daily_returns(returns_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted daily portfolio returns, holding weights fixed (no rebalancing)."""
    if returns_df.empty:
        raise ValueError("returns_df is empty.")

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
    if daily_returns.isna().any():
        raise ValueError("daily_returns contains NaN values.")

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


def simulate_with_rebalancing(
    returns_df: pd.DataFrame,
    target_weights: dict,
    rebalance_every: int | None = None,
    transaction_cost: float = 0.0,
) -> pd.Series:
    """Simulate a portfolio's daily returns, optionally rebalancing back to
    target_weights every `rebalance_every` trading days, deducting a
    transaction cost (as a fraction of turnover) each time a rebalance occurs."""
    if returns_df.empty:
        raise ValueError("returns_df is empty.")

    missing = set(returns_df.columns) - set(target_weights.keys())
    if missing:
        raise ValueError(f"Missing weights for columns: {missing}")

    total_weight = sum(target_weights.values())
    if not np.isclose(total_weight, 1.0, atol=1e-3):
        raise ValueError(f"target_weights must sum to 1.0, got {total_weight:.4f}")

    if rebalance_every is not None and rebalance_every <= 0:
        raise ValueError(f"rebalance_every must be positive or None, got {rebalance_every}")

    if transaction_cost < 0:
        raise ValueError(f"transaction_cost must be non-negative, got {transaction_cost}")

    assets = list(returns_df.columns)
    target = np.array([target_weights[a] for a in assets])

    current_weights = target.copy()
    daily_portfolio_returns = []

    for i, (_, day_returns) in enumerate(returns_df.iterrows()):
        r = day_returns.values

        day_return = np.dot(current_weights, r)
        daily_portfolio_returns.append(day_return)

        grown = current_weights * (1 + r)
        current_weights = grown / grown.sum()

        if rebalance_every is not None and (i + 1) % rebalance_every == 0:
            turnover = np.abs(current_weights - target).sum() / 2
            cost = turnover * transaction_cost
            daily_portfolio_returns[-1] -= cost
            current_weights = target.copy()

    return pd.Series(daily_portfolio_returns, index=returns_df.index)


def generate_walk_forward_windows(
    returns_df: pd.DataFrame, window_days: int, step_days: int
) -> list:
    """Generate (start, end) timestamp pairs for rolling windows of length
    `window_days`, advancing by `step_days` each time."""
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}")
    if step_days <= 0:
        raise ValueError(f"step_days must be positive, got {step_days}")
    if len(returns_df) < window_days:
        raise ValueError(
            f"returns_df has {len(returns_df)} observations, fewer than "
            f"window_days ({window_days}); no complete window is possible."
        )

    windows = []
    start_idx = 0
    while start_idx + window_days <= len(returns_df):
        window_dates = returns_df.index[start_idx:start_idx + window_days]
        windows.append((window_dates[0], window_dates[-1]))
        start_idx += step_days

    return windows


def walk_forward_backtest(
    returns_df: pd.DataFrame,
    strategy_weights: dict,
    benchmark_weights: dict,
    window_days: int,
    step_days: int,
    risk_free_rate: float = 0.02,
    rebalance_every=None,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Run the backtest across multiple rolling windows and return a DataFrame
    with one row per window, comparing strategy vs. benchmark performance."""
    windows = generate_walk_forward_windows(returns_df, window_days, step_days)

    rows = []
    for start, end in windows:
        window_returns = returns_df.loc[start:end]

        strategy_daily = simulate_with_rebalancing(
            window_returns, strategy_weights, rebalance_every, transaction_cost
        )
        benchmark_daily = simulate_with_rebalancing(
            window_returns, benchmark_weights, rebalance_every, transaction_cost
        )

        strategy_perf = performance_metrics(strategy_daily, risk_free_rate)
        benchmark_perf = performance_metrics(benchmark_daily, risk_free_rate)

        rows.append({
            "window_start": start,
            "window_end": end,
            "strategy_total_return": strategy_perf["total_return"],
            "strategy_sharpe": strategy_perf["sharpe_ratio"],
            "strategy_max_drawdown": strategy_perf["max_drawdown"],
            "benchmark_total_return": benchmark_perf["total_return"],
            "benchmark_sharpe": benchmark_perf["sharpe_ratio"],
            "benchmark_max_drawdown": benchmark_perf["max_drawdown"],
            "strategy_outperformed": strategy_perf["total_return"] > benchmark_perf["total_return"],
        })

    return pd.DataFrame(rows)