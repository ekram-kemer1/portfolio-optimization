import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from backtest import portfolio_daily_returns, performance_metrics


def test_portfolio_daily_returns_correct():
    returns = pd.DataFrame({"TSLA": [0.01, -0.02], "BND": [0.001, 0.002]})
    weights = {"TSLA": 0.6, "BND": 0.4}
    result = portfolio_daily_returns(returns, weights)
    expected_day1 = 0.6 * 0.01 + 0.4 * 0.001
    assert np.isclose(result.iloc[0], expected_day1)


def test_portfolio_daily_returns_bad_weights_raises():
    returns = pd.DataFrame({"TSLA": [0.01], "BND": [0.001]})
    weights = {"TSLA": 0.9, "BND": 0.3}  # sums to 1.2
    with pytest.raises(ValueError):
        portfolio_daily_returns(returns, weights)


def test_portfolio_daily_returns_missing_weight_raises():
    returns = pd.DataFrame({"TSLA": [0.01], "BND": [0.001], "SPY": [0.002]})
    weights = {"TSLA": 0.5, "BND": 0.5}  # missing SPY
    with pytest.raises(ValueError):
        portfolio_daily_returns(returns, weights)


def test_performance_metrics_keys():
    daily = pd.Series(np.random.normal(0.0005, 0.01, 252))
    result = performance_metrics(daily)
    assert set(result.keys()) == {"total_return", "annualized_return", "sharpe_ratio", "max_drawdown"}


def test_performance_metrics_empty_raises():
    with pytest.raises(ValueError):
        performance_metrics(pd.Series(dtype=float))