import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from backtest import (
    portfolio_daily_returns, performance_metrics,
    simulate_with_rebalancing, generate_walk_forward_windows, walk_forward_backtest,
)


def _sample_returns(n=500, seed=1):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TSLA": rng.normal(0.001, 0.03, n),
        "BND": rng.normal(0.0002, 0.003, n),
        "SPY": rng.normal(0.0005, 0.01, n),
    }, index=dates)


def test_portfolio_daily_returns_correct():
    returns = pd.DataFrame({"TSLA": [0.01, -0.02], "BND": [0.001, 0.002]})
    weights = {"TSLA": 0.6, "BND": 0.4}
    result = portfolio_daily_returns(returns, weights)
    expected_day1 = 0.6 * 0.01 + 0.4 * 0.001
    assert np.isclose(result.iloc[0], expected_day1)


def test_portfolio_daily_returns_bad_weights_raises():
    returns = pd.DataFrame({"TSLA": [0.01], "BND": [0.001]})
    weights = {"TSLA": 0.9, "BND": 0.3}
    with pytest.raises(ValueError, match="sum to 1.0"):
        portfolio_daily_returns(returns, weights)


def test_portfolio_daily_returns_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        portfolio_daily_returns(pd.DataFrame(), {})


def test_performance_metrics_keys():
    daily = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 252))
    result = performance_metrics(daily)
    assert set(result.keys()) == {"total_return", "annualized_return", "sharpe_ratio", "max_drawdown"}


def test_performance_metrics_nan_raises():
    daily = pd.Series([0.01, np.nan, 0.02])
    with pytest.raises(ValueError, match="NaN"):
        performance_metrics(daily)


def test_simulate_no_rebalancing_matches_static():
    returns = _sample_returns(50)
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}
    result = simulate_with_rebalancing(returns, weights, rebalance_every=None, transaction_cost=0.0)
    expected_day1 = 0.5 * returns["TSLA"].iloc[0] + 0.3 * returns["BND"].iloc[0] + 0.2 * returns["SPY"].iloc[0]
    assert np.isclose(result.iloc[0], expected_day1)


def test_simulate_with_rebalancing_runs():
    returns = _sample_returns(100)
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}
    result = simulate_with_rebalancing(returns, weights, rebalance_every=21, transaction_cost=0.001)
    assert len(result) == 100
    assert not result.isna().any()


def test_simulate_with_rebalancing_higher_cost_reduces_return():
    returns = _sample_returns(200)
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}
    low_cost = simulate_with_rebalancing(returns, weights, rebalance_every=21, transaction_cost=0.0)
    high_cost = simulate_with_rebalancing(returns, weights, rebalance_every=21, transaction_cost=0.05)
    assert (1 + low_cost).prod() >= (1 + high_cost).prod()


def test_simulate_with_rebalancing_bad_weights_raises():
    returns = _sample_returns(50)
    weights = {"TSLA": 0.5, "BND": 0.6}
    with pytest.raises(ValueError):
        simulate_with_rebalancing(returns, weights)


def test_simulate_with_rebalancing_negative_cost_raises():
    returns = _sample_returns(50)
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}
    with pytest.raises(ValueError, match="non-negative"):
        simulate_with_rebalancing(returns, weights, transaction_cost=-0.01)


def test_simulate_with_rebalancing_invalid_frequency_raises():
    returns = _sample_returns(50)
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}
    with pytest.raises(ValueError, match="positive or None"):
        simulate_with_rebalancing(returns, weights, rebalance_every=0)


def test_generate_walk_forward_windows_basic():
    returns = _sample_returns(500)
    windows = generate_walk_forward_windows(returns, window_days=100, step_days=50)
    assert len(windows) > 1
    start, end = windows[0]
    span = returns.loc[start:end]
    assert len(span) == 100


def test_generate_walk_forward_windows_too_large_raises():
    returns = _sample_returns(50)
    with pytest.raises(ValueError, match="no complete window"):
        generate_walk_forward_windows(returns, window_days=100, step_days=10)


def test_generate_walk_forward_windows_invalid_params_raise():
    returns = _sample_returns(100)
    with pytest.raises(ValueError, match="positive"):
        generate_walk_forward_windows(returns, window_days=0, step_days=10)
    with pytest.raises(ValueError, match="positive"):
        generate_walk_forward_windows(returns, window_days=50, step_days=0)


def test_walk_forward_backtest_basic():
    returns = _sample_returns(500)
    strategy_weights = {"TSLA": 0.0, "BND": 0.0, "SPY": 1.0}
    benchmark_weights = {"TSLA": 0.0, "BND": 0.4, "SPY": 0.6}

    results = walk_forward_backtest(
        returns, strategy_weights, benchmark_weights,
        window_days=100, step_days=100, rebalance_every=21, transaction_cost=0.001,
    )

    assert "strategy_outperformed" in results.columns
    assert len(results) > 1
    assert results["strategy_sharpe"].notna().all()