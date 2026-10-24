import numpy as np
import pandas as pd
import pytest

from exploratory_analysis import summarize_market_data


def test_summary_distinguishes_last_candle_from_latest_valid_close():
    index = pd.date_range("2026-10-09", periods=3, freq="D")
    data = pd.DataFrame({"Close": [1169.0, 1170.3, np.nan], "Volume": [100, 200, 300]}, index=index)

    summary = summarize_market_data(data)

    assert summary["candles"] == 3
    assert summary["observations"] == 2
    assert summary["latest_close"] == 1170.3
    assert summary["latest_close_date"] == index[1]
    assert pd.isna(summary["latest_return"])
    assert len(summary["daily_returns"]) == 1


@pytest.mark.parametrize("bad_price", [0.0, -1.0, np.inf, "bad"])
def test_market_summary_fails_safely_on_invalid_supplied_prices(bad_price):
    data = pd.DataFrame(
        {"Close": [100.0, bad_price], "Volume": [100, 200]},
        index=pd.date_range("2026-10-09", periods=2),
    )

    with pytest.raises(ValueError, match="summary is unavailable"):
        summarize_market_data(data)

