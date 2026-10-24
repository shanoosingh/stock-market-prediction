import pandas as pd
import pytest

from preprocessing import preprocess_market_data


def test_empty_data_rejected():
    with pytest.raises(ValueError, match="empty"):
        preprocess_market_data(pd.DataFrame())


def test_sorts_deduplicates_and_preserves_missing_values():
    dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-03", "2024-01-02"])
    frame = pd.DataFrame({"Close": [3, 1, 4, None], "Volume": [30, 10, 40, 20]}, index=dates)
    result = preprocess_market_data(frame)
    assert result.index.is_monotonic_increasing and result.index.is_unique
    assert result.iloc[-1]["Close"] == 4
    assert pd.isna(result.iloc[1]["Close"])


def test_rejects_missing_close_column():
    with pytest.raises(ValueError, match="Close"):
        preprocess_market_data(pd.DataFrame({"Volume": [1]}, index=pd.to_datetime(["2024-01-01"])))


def test_timezone_aware_intraday_index_is_preserved():
    index = pd.date_range("2024-01-01 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
    result = preprocess_market_data(pd.DataFrame({"Close": [100, 101]}, index=index))
    assert str(result.index.tz) == "Asia/Kolkata"
    assert result.index.equals(index)
