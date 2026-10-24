import pandas as pd
import pytest

import market_data_storage as storage
from market_data_storage import (
    MarketDataStorageError,
    get_stored_history,
    query_date_range,
    save_candles,
)


def candles(index=None, close_values=None):
    index = index if index is not None else pd.date_range(
        "2026-10-12 09:15", periods=3, freq="5min", tz="Asia/Kolkata", name="Date"
    )
    close_values = close_values if close_values is not None else [100.0, 101.0, 102.0]
    return pd.DataFrame(
        {
            "Open": [100.0, 100.0, 101.0][:len(index)],
            "High": [101.0, 102.0, 103.0][:len(index)],
            "Low": [99.0, 99.0, 100.0][:len(index)],
            "Close": close_values,
            "Volume": [10, 20, 30][:len(index)],
        },
        index=index,
    )


def test_database_creation_and_round_trip_preserves_ist_timestamps(tmp_path):
    database = tmp_path / "nested" / "candles.sqlite3"
    data = candles()

    assert save_candles(data, "RELIANCE.NS", "5m", "Yahoo Finance (yfinance)", database) == 3
    assert database.is_file()
    restored = get_stored_history("RELIANCE.NS", "5m", "Yahoo Finance (yfinance)", database)

    pd.testing.assert_frame_equal(restored, data, check_freq=False, check_dtype=False)
    assert str(restored.index.tz) == "Asia/Kolkata"
    assert restored.attrs["stored_data"] is True
    assert "retrieved_at" not in restored.attrs


def test_range_is_start_inclusive_and_end_exclusive(tmp_path):
    database = tmp_path / "candles.sqlite3"
    data = candles()
    save_candles(data, "RELIANCE.NS", "5m", "Yahoo", database)

    result = query_date_range(
        "RELIANCE.NS", "5m", "Yahoo",
        start="2026-10-12 09:20", end="2026-10-12 09:30", database_path=database,
    )

    assert list(result["Close"]) == [101.0, 102.0]
    assert str(result.index.tz) == "Asia/Kolkata"


def test_provider_source_isolation(tmp_path):
    database = tmp_path / "candles.sqlite3"
    data = candles()
    save_candles(data, "TCS.NS", "5m", "Yahoo Finance (yfinance)", database)
    separate_source = candles(close_values=[200.0, 201.0, 202.0])
    separate_source["Open"] = [200.0, 201.0, 202.0]
    separate_source["High"] = [201.0, 202.0, 203.0]
    separate_source["Low"] = [199.0, 200.0, 201.0]
    save_candles(separate_source, "TCS.NS", "5m", "OpenChart", database)

    yahoo = get_stored_history("TCS.NS", "5m", "Yahoo Finance (yfinance)", database)
    openchart = get_stored_history("TCS.NS", "5m", "OpenChart", database)

    assert list(yahoo["Close"]) == [100.0, 101.0, 102.0]
    assert list(openchart["Close"]) == [200.0, 201.0, 202.0]
    assert yahoo.attrs["source"] != openchart.attrs["source"]


def test_duplicate_timestamp_updates_existing_candle_deterministically(tmp_path):
    database = tmp_path / "candles.sqlite3"
    original = candles().iloc[:1]
    updated = candles(close_values=[105.5]).iloc[:1].copy()
    updated.loc[:, "Open"] = 105.0
    updated.loc[:, "High"] = 106.0
    updated.loc[:, "Low"] = 104.0
    save_candles(original, "INFY.NS", "5m", "Yahoo", database)
    save_candles(updated, "INFY.NS", "5m", "Yahoo", database)

    restored = get_stored_history("INFY.NS", "5m", "Yahoo", database)
    assert len(restored) == 1
    assert restored.iloc[0]["Close"] == 105.5


def test_naive_indian_timestamps_are_interpreted_as_ist(tmp_path):
    database = tmp_path / "candles.sqlite3"
    naive_index = pd.DatetimeIndex(["2026-10-12 09:15"], name="Date")
    save_candles(candles(naive_index, [100.0]), "RELIANCE.NS", "5m", "Yahoo", database)

    restored = get_stored_history("RELIANCE.NS", "5m", "Yahoo", database)
    assert restored.index[0] == pd.Timestamp("2026-10-12 09:15", tz="Asia/Kolkata")


@pytest.mark.parametrize(
    "column,changes,expected",
    [
        ("Open", {"Open": [104.0, 100.0, 101.0]}, "High is below Open"),
        ("Close", {"Close": [103.0, 101.0, 102.0]}, "High is below Close"),
        ("Low", {"Low": [102.0, 99.0, 100.0]}, "Low is above Open"),
        ("Volume", {"Volume": [-1, 20, 30]}, "Volume is negative"),
    ],
)
def test_invalid_ohlcv_is_reported_and_rejected(tmp_path, column, changes, expected):
    data = candles()
    for name, values in changes.items():
        data[name] = values
    with pytest.raises(MarketDataStorageError, match=expected):
        save_candles(data, "RELIANCE.NS", "5m", "Yahoo", tmp_path / f"{column}.sqlite3")


def test_database_errors_are_reported(tmp_path):
    database_path_is_directory = tmp_path / "as_directory.sqlite3"
    database_path_is_directory.mkdir()

    with pytest.raises(MarketDataStorageError, match="Could not open candle database"):
        get_stored_history("TCS.NS", "5m", "Yahoo", database_path_is_directory)

