import numpy as np
import pandas as pd

from technical_indicators import add_technical_indicators


def test_sma_matches_trailing_means_and_warmup_is_nan():
    frame = pd.DataFrame({"Close": np.arange(1.0, 61.0)})
    result = add_technical_indicators(frame)
    assert result["SMA20"].iloc[:19].isna().all()
    assert result["SMA20"].iloc[19] == 10.5
    assert result["SMA50"].iloc[49] == 25.5


def test_rsi_monotonic_series_and_flat_series():
    rising = add_technical_indicators(pd.DataFrame({"Close": np.arange(1.0, 40.0)}))
    assert (rising["RSI14"].dropna() == 100).all()
    flat = add_technical_indicators(pd.DataFrame({"Close": np.ones(40)}))
    assert (flat["RSI14"].dropna() == 50).all()


def test_macd_columns_present_and_warmup_is_nan():
    result = add_technical_indicators(pd.DataFrame({"Close": np.arange(1.0, 80.0)}))
    assert {"MACD", "MACD Signal", "MACD Histogram"} <= set(result)
    assert result["MACD Signal"].iloc[:33].isna().all()
    assert result["MACD Signal"].notna().any()


def test_missing_prices_are_not_filled():
    result = add_technical_indicators(pd.DataFrame({"Close": [1.0, np.nan, 3.0]}))
    assert pd.isna(result["Close"].iloc[1])
