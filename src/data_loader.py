"""
data_loader.py

Handles extraction and cleaning of historical price data for TSLA, BND, and SPY
using the yfinance API, per Task 1. All functions validate inputs explicitly.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["TSLA", "BND", "SPY"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-06-30"


def fetch_price_data(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data for a list of tickers from YFinance."""
    if not tickers:
        raise ValueError("tickers list is empty.")

    try:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid start/end date: {e}")

    if start_ts >= end_ts:
        raise ValueError(f"start ({start}) must be before end ({end}).")

    data = {}
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f"Invalid ticker: {ticker!r}")

        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        except Exception as e:
            raise ValueError(f"yfinance request failed for {ticker}: {e}") from e

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for {ticker}. Check network access to "
                "Yahoo Finance (query1/query2.finance.yahoo.com), the ticker symbol, "
                "and that start/end fall within a valid trading range."
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        required_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"{ticker} response is missing expected columns: {missing_cols}")

        df.index.name = "Date"
        data[ticker] = df

    return data


def combine_close_prices(data: dict[str, pd.DataFrame], field: str = "Adj Close") -> pd.DataFrame:
    """Combine a single price field from multiple tickers into one wide DataFrame."""
    if not data:
        raise ValueError("data dict is empty.")
    for ticker, df in data.items():
        if field not in df.columns:
            raise ValueError(f"'{field}' not found in {ticker}'s columns: {list(df.columns)}")
    return pd.DataFrame({ticker: df[field] for ticker, df in data.items()})


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a wide price DataFrame: reindex onto a full business-day calendar
    and forward/backward-fill gaps."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()
    try:
        df = df.astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not cast all columns to float: {e}")

    full_index = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(full_index)
    df.index.name = "Date"

    df = df.ffill().bfill()

    if df.isna().any().any():
        remaining = df.isna().sum()
        raise ValueError(
            f"NaNs remain after ffill/bfill (likely leading gaps with no prior data): {dict(remaining)}"
        )

    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        raise ValueError("Refusing to save an empty DataFrame.")
    df.to_csv(path)