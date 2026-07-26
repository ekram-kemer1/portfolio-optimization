import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from portfolio import (
    build_expected_returns, compute_covariance,
    optimize_max_sharpe, optimize_min_volatility, build_efficient_frontier,
)


def _sample_returns(seed=1, n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TSLA": rng.normal(0.001, 0.03, n),
        "BND": rng.normal(0.0002, 0.003, n),
        "SPY": rng.normal(0.0005, 0.01, n),
    }, index=dates)


def _sample_prices(seed=1, n=300):
    returns = _sample_returns(seed, n)
    return 100 * (1 + returns).cumprod()


def test_build_expected_returns_basic():
    returns = _sample_returns()
    result = build_expected_returns("TSLA", forecast_final_price=450, forecast_last_known_price=400,
                                     forecast_horizon_days=189, historical_returns=returns)
    assert set(result.index) == {"TSLA", "BND", "SPY"}
    assert np.isfinite(result).all()


def test_build_expected_returns_invalid_price_raises():
    returns = _sample_returns()
    with pytest.raises(ValueError, match="must be positive"):
        build_expected_returns("TSLA", forecast_final_price=450, forecast_last_known_price=0,
                                forecast_horizon_days=189, historical_returns=returns)


def test_build_expected_returns_unknown_asset_raises():
    returns = _sample_returns()
    with pytest.raises(ValueError, match="not found"):
        build_expected_returns("AAPL", forecast_final_price=450, forecast_last_known_price=400,
                                forecast_horizon_days=189, historical_returns=returns)


def test_build_expected_returns_nan_data_raises():
    returns = _sample_returns()
    returns.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_expected_returns("TSLA", forecast_final_price=450, forecast_last_known_price=400,
                                forecast_horizon_days=189, historical_returns=returns)


def test_compute_covariance_basic():
    prices = _sample_prices()
    cov = compute_covariance(prices)
    assert cov.shape == (3, 3)


def test_compute_covariance_rejects_nans():
    prices = pd.DataFrame({"TSLA": [100, np.nan, 105], "BND": [80, 81, 82]})
    with pytest.raises(ValueError, match="NaN"):
        compute_covariance(prices)


def test_compute_covariance_rejects_single_asset():
    prices = pd.DataFrame({"TSLA": np.linspace(100, 110, 40)})
    with pytest.raises(ValueError, match="at least 2 assets"):
        compute_covariance(prices)


def test_compute_covariance_rejects_too_few_observations():
    prices = pd.DataFrame({"TSLA": [100, 101, 102], "BND": [80, 81, 82]})
    with pytest.raises(ValueError, match="observations"):
        compute_covariance(prices)


def test_optimize_max_sharpe_all_negative_raises():
    expected_returns = pd.Series({"TSLA": -0.44, "BND": -0.01, "SPY": -0.02})
    cov_matrix = pd.DataFrame(
        [[0.315, 0.002, 0.048], [0.002, 0.003, 0.001], [0.048, 0.001, 0.030]],
        index=["TSLA", "BND", "SPY"], columns=["TSLA", "BND", "SPY"],
    )
    with pytest.raises(ValueError, match="positive expected return"):
        optimize_max_sharpe(expected_returns, cov_matrix)


def test_optimize_max_sharpe_excludes_negative_asset():
    expected_returns = pd.Series({"TSLA": -0.4420, "BND": 0.0192, "SPY": 0.1391})
    cov_matrix = pd.DataFrame(
        [[0.3151, 0.0017, 0.0480], [0.0017, 0.0027, 0.0010], [0.0480, 0.0010, 0.0300]],
        index=["TSLA", "BND", "SPY"], columns=["TSLA", "BND", "SPY"],
    )
    weights, performance = optimize_max_sharpe(expected_returns, cov_matrix)
    assert weights["TSLA"] < 0.01
    assert performance[2] > 0


def test_optimize_min_volatility_basic():
    expected_returns = pd.Series({"TSLA": -0.4420, "BND": 0.0192, "SPY": 0.1391})
    cov_matrix = pd.DataFrame(
        [[0.3151, 0.0017, 0.0480], [0.0017, 0.0027, 0.0010], [0.0480, 0.0010, 0.0300]],
        index=["TSLA", "BND", "SPY"], columns=["TSLA", "BND", "SPY"],
    )
    weights, performance = optimize_min_volatility(expected_returns, cov_matrix)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["BND"] > weights["TSLA"]


def test_mismatched_indices_raise():
    expected_returns = pd.Series({"TSLA": 0.1, "BND": 0.02})
    cov_matrix = pd.DataFrame([[0.3, 0.01], [0.01, 0.01]], index=["TSLA", "SPY"], columns=["TSLA", "SPY"])
    with pytest.raises(ValueError, match="matching"):
        optimize_max_sharpe(expected_returns, cov_matrix)


def test_build_efficient_frontier_basic():
    expected_returns = pd.Series({"TSLA": -0.4420, "BND": 0.0192, "SPY": 0.1391})
    cov_matrix = pd.DataFrame(
        [[0.3151, 0.0017, 0.0480], [0.0017, 0.0027, 0.0010], [0.0480, 0.0010, 0.0300]],
        index=["TSLA", "BND", "SPY"], columns=["TSLA", "BND", "SPY"],
    )
    vols, rets = build_efficient_frontier(expected_returns, cov_matrix, n_points=20)
    assert len(vols) == len(rets)
    assert len(vols) > 0