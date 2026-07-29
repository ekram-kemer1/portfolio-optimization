import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import fetch_price_data, combine_close_prices, clean_price_data, save_processed


def test_fetch_price_data_empty_tickers_raises():
    with pytest.raises(ValueError, match="empty"):
        fetch_price_data(tickers=[], start="2020-01-01", end="2020-06-01")


def test_fetch_price_data_invalid_dates_raises():
    with pytest.raises(ValueError, match="before"):
        fetch_price_data(tickers=["TSLA"], start="2024-01-01", end="2020-01-01")


def test_fetch_price_data_bad_date_string_raises():
    with pytest.raises(ValueError, match="Invalid start/end date"):
        fetch_price_data(tickers=["TSLA"], start="not-a-date", end="2020-01-01")


def test_fetch_price_data_invalid_ticker_type_raises():
    with pytest.raises(ValueError, match="Invalid ticker"):
        fetch_price_data(tickers=["", "TSLA"], start="2020-01-01", end="2020-06-01")


def _sample_close_data():
    dates = pd.bdate_range("2020-01-01", periods=50)
    return {
        "TSLA": pd.DataFrame({"Adj Close": np.linspace(100, 150, 50)}, index=dates),
        "BND": pd.DataFrame({"Adj Close": np.linspace(80, 82, 50)}, index=dates),
    }


def test_combine_close_prices_basic():
    data = _sample_close_data()
    combined = combine_close_prices(data)
    assert list(combined.columns) == ["TSLA", "BND"]
    assert len(combined) == 50


def test_combine_close_prices_empty_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        combine_close_prices({})


def test_combine_close_prices_missing_field_raises():
    data = {"TSLA": pd.DataFrame({"Close": [1, 2, 3]})}
    with pytest.raises(ValueError, match="not found"):
        combine_close_prices(data, field="Adj Close")


def test_clean_price_data_fills_gaps():
    dates = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"TSLA": np.linspace(100, 110, 10)}, index=dates)
    cleaned = clean_price_data(df)
    assert not cleaned.isna().any().any()


def test_clean_price_data_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        clean_price_data(pd.DataFrame())


def test_clean_price_data_non_numeric_raises():
    df = pd.DataFrame({"TSLA": ["a", "b", "c"]}, index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="cast"):
        clean_price_data(df)


def test_save_processed_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_processed(pd.DataFrame(), str(tmp_path / "out.csv"))


def test_save_processed_writes_file(tmp_path):
    df = pd.DataFrame({"TSLA": [1, 2, 3]})
    path = str(tmp_path / "out.csv")
    save_processed(df, path)
    assert os.path.exists(path)