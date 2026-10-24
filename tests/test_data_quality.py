import pandas as pd

from preprocessing import assess_data_quality, get_latest_candle, preprocess_market_data


def test_quality_report_flags_empty_provider_response():
    assert "No rows were returned by the data provider." in assess_data_quality(pd.DataFrame())


def test_quality_report_flags_duplicate_and_unsorted_timestamps():
    index = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-03"])
    data = pd.DataFrame({"Close": [3, 2, 4]}, index=index)

    issues = assess_data_quality(data)

    assert any("duplicate timestamps" in issue for issue in issues)
    assert any("out of chronological order" in issue for issue in issues)
    # The warning survives preprocessing even though preprocessing sorts and deduplicates.
    cleaned = preprocess_market_data(data)
    assert any("duplicate timestamps" in issue for issue in cleaned.attrs["quality_issues"])
    assert any("out of chronological order" in issue for issue in cleaned.attrs["quality_issues"])


def test_quality_report_flags_missing_ohlcv_columns():
    data = pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2024-01-01"]))

    issues = assess_data_quality(data)

    assert any("Missing OHLCV columns: Open, High, Low, Volume" in issue for issue in issues)


def test_quality_report_flags_invalid_numeric_and_impossible_values():
    data = pd.DataFrame(
        {"Open": ["bad"], "High": [9], "Low": [10], "Close": [8], "Volume": [-1]},
        index=pd.to_datetime(["2024-01-01"]),
    )

    issues = assess_data_quality(data)

    assert any("Open contains 1 invalid numeric value" in issue for issue in issues)
    assert any("High is below Low" in issue for issue in issues)
    assert any("Volume is negative" in issue for issue in issues)


def test_quality_report_checks_high_and_low_against_open_and_close():
    data = pd.DataFrame(
        {
            "Open": [12.0, 8.0, 12.0],
            "High": [11.0, 13.0, 11.0],
            "Low": [9.0, 10.0, 10.0],
            "Close": [10.0, 9.0, 13.0],
            "Volume": [100, 100, 100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    issues = assess_data_quality(data)

    assert any("High is below Open in 2 row(s)" in issue for issue in issues)
    assert any("Low is above Close in 1 row(s)" in issue for issue in issues)
    assert any("High is below Close in 1 row(s)" in issue for issue in issues)
    assert all("2024-01-" in issue for issue in issues if "below" in issue or "above" in issue)


def test_quality_report_flags_nonpositive_ohlc_and_adjusted_prices():
    data = pd.DataFrame(
        {
            "Open": [100.0, 0.0],
            "High": [101.0, 0.0],
            "Low": [99.0, -1.0],
            "Close": [100.0, 0.0],
            "Adj Close": [100.0, -0.5],
            "Volume": [10, 10],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )

    issues = assess_data_quality(data)

    assert any("Open is nonpositive in 1 row(s)" in issue for issue in issues)
    assert any("High is nonpositive in 1 row(s)" in issue for issue in issues)
    assert any("Low is nonpositive in 1 row(s)" in issue for issue in issues)
    assert any("Close is nonpositive in 1 row(s)" in issue for issue in issues)
    assert any("Adj Close is nonpositive in 1 row(s)" in issue for issue in issues)


def test_latest_candle_uses_max_timestamp_and_reports_missing_close():
    data = pd.DataFrame(
        {"Close": [101.0, None]},
        index=pd.to_datetime(["2024-01-02 09:20", "2024-01-02 09:15"]),
    )

    latest_time, latest_close = get_latest_candle(data)

    assert latest_time == pd.Timestamp("2024-01-02 09:20")
    assert latest_close == 101.0
    data.loc[pd.Timestamp("2024-01-02 09:20"), "Close"] = None
    assert get_latest_candle(data)[1] is None
