from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from investment_analysis import (
    assess_data_freshness,
    calculate_investing_metrics,
    compare_benchmark_returns,
    read_benchmark_csv,
)


def test_recent_returns_compound_observed_sessions_without_filling_gaps():
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    frame = pd.DataFrame({"Adj Close": [100.0, 110.0, 99.0, 108.9, 119.79]}, index=dates)

    metrics = calculate_investing_metrics(frame, recent_sessions=(1, 2, 4))

    assert metrics["recent_returns"][1] == pytest.approx(0.10)
    assert metrics["recent_returns"][2] == pytest.approx(0.21)
    assert metrics["recent_returns"][4] == pytest.approx(0.1979)

    frame.iloc[-1, 0] = np.nan
    missing = calculate_investing_metrics(frame, recent_sessions=(1, 2))
    assert missing["recent_returns"][1] is None
    assert missing["recent_returns"][2] is None
    assert any("last observation has no usable price" in w for w in missing["warnings"])


def test_volatility_and_drawdown_use_documented_windows_and_formulas():
    returns = np.array([0.01, -0.02] * 15, dtype=float)
    prices = pd.Series([100.0, *list(100 * np.cumprod(1 + returns))], index=pd.bdate_range("2025-01-01", periods=31), name="Adj Close")
    frame = prices.to_frame()

    metrics = calculate_investing_metrics(frame, recent_sessions=(1, 5, 20))

    expected_volatility = pd.Series(returns[-20:]).std(ddof=1) * np.sqrt(252)
    expected_drawdown = (prices / prices.cummax() - 1).min()
    assert metrics["annualized_volatility"] == pytest.approx(expected_volatility)
    assert metrics["volatility_window"] == 20
    assert metrics["volatility_observations"] == 20
    assert metrics["max_drawdown"] == pytest.approx(expected_drawdown)
    assert metrics["drawdown_observations"] == 31


def test_latest_volume_warning_compares_with_prior_20_observations():
    dates = pd.bdate_range("2026-01-01", periods=22)
    frame = pd.DataFrame({"Adj Close": np.linspace(100, 110, len(dates)), "Volume": [100] * 21 + [400]}, index=dates)

    metrics = calculate_investing_metrics(frame)

    assert metrics["latest_volume_vs_prior_median"] == pytest.approx(4.0)
    assert any("4.0x the prior 20-observation median" in warning for warning in metrics["warnings"])


def test_volume_comparison_stays_aligned_to_latest_valid_price_date():
    dates = pd.bdate_range("2026-01-01", periods=23)
    frame = pd.DataFrame(
        {"Adj Close": [100.0] * 22 + [np.nan], "Volume": [100] * 21 + [400, 900]},
        index=dates,
    )

    metrics = calculate_investing_metrics(frame)

    assert metrics["volume_observation_date"] == dates[-2]
    assert metrics["latest_volume_vs_prior_median"] == pytest.approx(4.0)


def test_freshness_separates_provider_age_from_price_observation_age():
    dates = pd.date_range("2026-03-01", periods=2, freq="B")
    frame = pd.DataFrame({"Adj Close": [100.0, 101.0]}, index=dates)
    now = pd.Timestamp("2026-03-10 12:00", tz="Asia/Kolkata")

    stale = assess_data_freshness(frame, "Adj Close", retrieved_at="2026-03-10 11:30", stale=True, now=now)
    old_retrieval = assess_data_freshness(frame, "Adj Close", retrieved_at="2026-03-08 10:00", now=now)

    assert stale["status"] == "Stale cached snapshot"
    assert any("last successful snapshot" in warning for warning in stale["warnings"])
    assert old_retrieval["status"] == "Retrieval metadata is old"
    assert old_retrieval["observation_age_days"] == 7


def test_freshness_reports_latest_observation_in_ist_and_uses_its_local_date():
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2026-03-01 19:00", tz="UTC"), pd.Timestamp("2026-03-01 20:00", tz="UTC")]
    )
    frame = pd.DataFrame({"Adj Close": [100.0, 101.0]}, index=dates)
    now = pd.Timestamp("2026-03-02 03:00", tz="Asia/Kolkata")

    result = assess_data_freshness(
        frame, "Adj Close", retrieved_at=now, now=now
    )

    assert result["latest_valid_price_date"] == pd.Timestamp("2026-03-02 01:30", tz="Asia/Kolkata")
    assert result["observation_age_days"] == 0


def test_future_retrieval_timestamp_is_explicitly_flagged():
    dates = pd.date_range("2026-03-01", periods=2, freq="B")
    frame = pd.DataFrame({"Adj Close": [100.0, 101.0]}, index=dates)
    now = pd.Timestamp("2026-03-10 12:00", tz="Asia/Kolkata")

    result = assess_data_freshness(
        frame,
        "Adj Close",
        retrieved_at="2026-03-10 13:00+05:30",
        now=now,
    )

    assert result["status"] == "Retrieval timestamp is in the future"
    assert any("ahead of the current time" in warning for warning in result["warnings"])


def test_benchmark_comparison_uses_only_matching_dates_and_compounded_relative_return():
    stock_dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    benchmark_dates = pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-06"])
    stock = pd.Series([100.0, 110.0, 121.0], index=stock_dates)
    benchmark = pd.Series([200.0, 220.0, 242.0], index=pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-06"]))

    result = compare_benchmark_returns(stock, benchmark)

    assert result["matched_price_dates"] == 2
    assert result["matched_return_intervals"] == 1
    assert result["start_date"] == pd.Timestamp("2026-01-01")
    assert result["end_date"] == pd.Timestamp("2026-01-05")
    assert result["stock_absolute_return"] == pytest.approx(0.21)
    assert result["benchmark_absolute_return"] == pytest.approx(0.10)
    assert result["benchmark_relative_return"] == pytest.approx(0.10)


def test_benchmark_csv_requires_matching_price_basis_and_valid_dates():
    missing_basis = BytesIO(b"Date,Close\n2026-01-01,100\n2026-01-02,101\n")
    with pytest.raises(ValueError, match="Adj Close"):
        read_benchmark_csv(missing_basis, "Adj Close")

    duplicate_dates = BytesIO(b"Date,Adj Close\n2026-01-01,100\n2026-01-01,101\n")
    with pytest.raises(ValueError, match="duplicates"):
        read_benchmark_csv(duplicate_dates)

    invalid_dates = BytesIO(b"Date,Adj Close\nnot-a-date,100\n2026-01-02,101\n")
    with pytest.raises(ValueError, match="invalid dates"):
        read_benchmark_csv(invalid_dates)


def test_benchmark_requires_two_matching_price_dates():
    stock = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    benchmark = pd.Series([200.0, 201.0], index=pd.to_datetime(["2026-01-03", "2026-01-04"]))
    with pytest.raises(ValueError, match="two matching price dates"):
        compare_benchmark_returns(stock, benchmark)


def test_invalid_prices_and_insufficient_volatility_are_reported():
    frame = pd.DataFrame({"Adj Close": [100.0, 0.0]}, index=pd.date_range("2026-01-01", periods=2))
    with pytest.raises(ValueError, match="finite and positive"):
        calculate_investing_metrics(frame)

    short = pd.DataFrame({"Adj Close": [100.0, 101.0]}, index=pd.date_range("2026-01-01", periods=2))
    result = calculate_investing_metrics(short)
    assert result["annualized_volatility"] is None
    assert result["recent_returns"][5] is None


def test_nonnumeric_prices_are_visible_in_metric_and_freshness_warnings():
    frame = pd.DataFrame(
        {"Adj Close": [100.0, "bad", 102.0]},
        index=pd.date_range("2026-01-01", periods=3),
    )

    metrics = calculate_investing_metrics(frame)
    freshness = assess_data_freshness(
        frame, "Adj Close", retrieved_at="2026-01-03", now="2026-01-03"
    )

    assert any("1 supplied Adj Close value(s) are nonnumeric" in w for w in metrics["warnings"])
    assert any("1 supplied Adj Close value(s) are nonnumeric" in w for w in freshness["warnings"])
