from datetime import date, timedelta
import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from data_collection import fetch_stock_data
from preprocessing import format_retrieval_time, get_retrieval_time


def test_daily_request_remains_default_and_normalizes_yfinance_multiindex():
    index = pd.date_range("2024-01-02", periods=2)
    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["RELIANCE.NS"]])
    response = pd.DataFrame([[100, 102, 99, 101, 101, 500], [101, 103, 100, 102, 102, 600]], index=index, columns=columns)
    with patch("data_collection.yf.download", return_value=response) as download:
        result = fetch_stock_data("reliance.ns", period="1y")

    assert download.call_args.kwargs["interval"] == "1d"
    assert download.call_args.kwargs["period"] == "1y"
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert result.index.equals(index)


def test_download_records_retrieval_time_for_display_in_india_timezone():
    index = pd.date_range("2024-01-02", periods=1)
    response = pd.DataFrame({"Close": [101]}, index=index)
    with patch("data_collection.yf.download", return_value=response):
        result = fetch_stock_data("RELIANCE.NS", period="1mo")

    retrieved_at = get_retrieval_time(result)
    assert retrieved_at is not None
    assert str(retrieved_at.tz) == "Asia/Kolkata"
    assert format_retrieval_time(result).endswith("IST (Asia/Kolkata)")


def test_unsupported_interval_is_rejected_before_download():
    with patch("data_collection.yf.download") as download:
        with pytest.raises(ValueError, match="1d, 5m, or 15m"):
            fetch_stock_data("TCS.NS", interval="1m")
    download.assert_not_called()


def test_intraday_custom_range_older_than_sixty_days_is_rejected():
    start = date.today() - timedelta(days=61)
    end = date.today() + timedelta(days=1)
    with patch("data_collection.yf.download") as download:
        with pytest.raises(ValueError, match="most recent 60 days"):
            fetch_stock_data("INFY.NS", start=start, end=end, interval="5m")
    download.assert_not_called()


def test_intraday_request_converts_indian_timestamps_to_kolkata():
    index = pd.DatetimeIndex([pd.Timestamp.now(tz="UTC").floor("5min")])
    response = pd.DataFrame({"Open": [100], "High": [102], "Low": [99], "Close": [101], "Volume": [500]}, index=index)
    with patch("data_collection.yf.download", return_value=response) as download:
        result = fetch_stock_data("TCS.NS", period="1d", interval="5m")

    assert download.call_args.kwargs["interval"] == "5m"
    assert str(result.index.tz) == "Asia/Kolkata"
    assert result.index[0] == index.tz_convert("Asia/Kolkata")[0]


def test_naive_indian_intraday_timestamps_are_localized_not_stripped():
    index = pd.DatetimeIndex([pd.Timestamp.now(tz="Asia/Kolkata").floor("15min").tz_localize(None)])
    response = pd.DataFrame({"Close": [101]}, index=index)
    with patch("data_collection.yf.download", return_value=response):
        result = fetch_stock_data("INFY.NS", period="5d", interval="15m")

    assert str(result.index.tz) == "Asia/Kolkata"
    assert result.index[0].hour == index[0].hour


def test_empty_intraday_response_has_clear_unavailable_data_error():
    with patch("data_collection.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="Yahoo Finance returned no intraday data.*connection problem"):
            fetch_stock_data("RELIANCE.NS", period="5d", interval="15m")


def test_yfinance_download_exception_is_reported_without_losing_cause():
    with patch("data_collection.yf.download", side_effect=sqlite3.OperationalError("unable to open database file")):
        with pytest.raises(RuntimeError, match="Yahoo Finance could not retrieve TCS.NS: unable to open database file"):
            fetch_stock_data("TCS.NS", period="1mo")


def test_intraday_period_outside_supported_short_ranges_is_rejected():
    with patch("data_collection.yf.download") as download:
        with pytest.raises(ValueError, match="only 1d, 5d, or 1mo"):
            fetch_stock_data("RELIANCE.NS", period="1y", interval="15m")
    download.assert_not_called()
