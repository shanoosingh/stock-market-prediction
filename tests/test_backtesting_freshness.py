import pandas as pd
import pytest

from backtesting import (
    BacktestInputMetadata,
    DEFAULT_MAX_DAILY_OBSERVATION_AGE_DAYS,
    DEFAULT_MAX_DAILY_RETRIEVAL_AGE_HOURS,
    assess_backtest_input_freshness,
    build_backtest_report,
    run_backtest,
    run_validated_backtest,
)


def prices():
    return pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
        name="Adj Close",
    )


def metadata(**overrides):
    values = {
        "source": "Offline synthetic fixture",
        "latest_observation_at": "2026-01-03 15:30",
        "retrieved_at": "2026-01-03 15:45",
        "evaluated_at": "2026-01-03 16:00",
        "freshness_status": "fresh",
        "synthetic": True,
    }
    values.update(overrides)
    return BacktestInputMetadata(**values)


def test_fresh_daily_metadata_is_validated_with_configurable_defaults():
    assessed = assess_backtest_input_freshness(metadata())

    assert DEFAULT_MAX_DAILY_OBSERVATION_AGE_DAYS == 7
    assert DEFAULT_MAX_DAILY_RETRIEVAL_AGE_HOURS == 24
    assert assessed["status"] == "fresh_validated"
    assert assessed["allowed"] is True
    assert assessed["observation_age_days"] == 0
    assert assessed["retrieval_age_hours"] == pytest.approx(0.25)
    assert assessed["synthetic"] is True

    result = run_validated_backtest(
        prices(), "buy_and_hold", input_metadata=metadata()
    )
    assert build_backtest_report(result)["input_freshness_status"] == "fresh_validated"


def test_validated_request_rejects_metadata_for_a_different_latest_price_date():
    incorrect = metadata(latest_observation_at="2026-01-02 15:30")

    with pytest.raises(ValueError, match="does not match the supplied price data"):
        run_validated_backtest(prices(), "buy_and_hold", input_metadata=incorrect)


def test_stale_daily_input_is_rejected_by_default_and_policy_reason_is_reportable():
    stale = metadata(
        latest_observation_at="2025-12-20 15:30",
        retrieved_at="2026-01-03 15:45",
    )
    assessment = assess_backtest_input_freshness(stale)

    assert assessment["status"] == "stale_rejected"
    assert assessment["allowed"] is False
    assert any("calendar days old" in warning for warning in assessment["warnings"])
    with pytest.raises(ValueError, match="stale_rejected") as exc_info:
        run_validated_backtest(prices(), "buy_and_hold", input_metadata=stale)
    assert exc_info.value.assessment["status"] == "stale_rejected"
    with pytest.raises(ValueError, match="rejected by freshness policy"):
        run_backtest(prices(), "buy_and_hold", input_metadata=stale)


def test_explicit_stale_override_is_visible_in_result_report():
    stale = metadata(freshness_status="Stale cached snapshot")
    result = run_validated_backtest(
        prices(),
        "buy_and_hold",
        input_metadata=stale,
        allow_stale_override=True,
    )
    report = build_backtest_report(result)

    assert report["input_freshness_status"] == "stale_override"
    assert report["input_validation"]["stale_override"] is True
    assert any("explicit diagnostic override" in warning for warning in report["input_validation"]["warnings"])


def test_unknown_or_missing_provenance_is_not_treated_as_fresh():
    unknown = metadata(source=None, retrieved_at=None, latest_observation_at=None)
    assessment = assess_backtest_input_freshness(unknown, require_known=True)

    assert assessment["status"] == "unknown"
    assert assessment["allowed"] is False
    with pytest.raises(ValueError, match="unknown") as exc_info:
        run_validated_backtest(prices(), "buy_and_hold", input_metadata=unknown)
    assert exc_info.value.assessment["status"] == "unknown"

    # Low-level calculations remain backward-compatible but label freshness unknown.
    report = build_backtest_report(run_backtest(prices(), "buy_and_hold"))
    assert report["input_freshness_status"] == "unknown"
    assert report["input_validation"]["allowed"] is True


def test_timezone_aware_timestamps_are_compared_as_ist_instants():
    aware = metadata(
        latest_observation_at=pd.Timestamp("2026-01-01 20:00", tz="UTC"),
        retrieved_at=pd.Timestamp("2026-01-02 02:00", tz="UTC"),
        evaluated_at=pd.Timestamp("2026-01-02 03:00", tz="UTC"),
    )
    assessment = assess_backtest_input_freshness(aware)

    assert assessment["status"] == "fresh_validated"
    assert assessment["latest_observation_at"] == pd.Timestamp("2026-01-02 01:30", tz="Asia/Kolkata")
    assert assessment["evaluated_at"] == pd.Timestamp("2026-01-02 08:30", tz="Asia/Kolkata")
    assert assessment["observation_age_days"] == 0
    assert assessment["retrieval_age_hours"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("latest_observation_at", "2026-01-04 10:00"),
        ("retrieved_at", "2026-01-04 10:00"),
    ],
)
def test_future_observation_or_retrieval_timestamp_is_rejected(field, value):
    assessed = assess_backtest_input_freshness(metadata(**{field: value}))

    assert assessed["status"] == "stale_rejected"
    assert assessed["allowed"] is False
    assert any("future relative to evaluation time" in warning for warning in assessed["warnings"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("latest_observation_at", "2026-01-04 10:00"),
        ("retrieved_at", "2026-01-04 10:00"),
    ],
)
def test_future_timestamps_cannot_be_overridden(field, value):
    assessed = assess_backtest_input_freshness(
        metadata(**{field: value}), allow_stale_override=True, require_known=True
    )
    assert assessed["status"] == "stale_rejected"
    assert assessed["allowed"] is False
    assert assessed["stale_override"] is False


def test_intraday_input_does_not_use_daily_limits_or_claim_complete_bars():
    intraday = metadata(interval="5m")
    assessment = assess_backtest_input_freshness(intraday, require_known=True)

    assert assessment["status"] == "unknown"
    assert assessment["allowed"] is False
    assert any("bar completeness is unverified" in warning for warning in assessment["warnings"])
    with pytest.raises(ValueError, match="unknown"):
        run_validated_backtest(prices(), "buy_and_hold", input_metadata=intraday)

    stale_intraday = metadata(interval="5m", freshness_status="stale")
    overridden = assess_backtest_input_freshness(
        stale_intraday, allow_stale_override=True, require_known=True
    )
    assert overridden["status"] == "unknown"
    assert overridden["allowed"] is False
    assert overridden["stale_override"] is False
    with pytest.raises(ValueError, match="unknown"):
        run_validated_backtest(
            prices(),
            "buy_and_hold",
            input_metadata=stale_intraday,
            allow_stale_override=True,
        )


def test_custom_daily_freshness_limits_are_applied_and_validated():
    just_over_limit = metadata(latest_observation_at="2025-12-31 15:30")
    assessment = assess_backtest_input_freshness(
        just_over_limit,
        max_daily_observation_age_days=2,
        max_daily_retrieval_age_hours=1,
    )
    assert assessment["status"] == "stale_rejected"
    with pytest.raises(ValueError, match="finite, nonnegative"):
        assess_backtest_input_freshness(metadata(), max_daily_observation_age_days=-1)
