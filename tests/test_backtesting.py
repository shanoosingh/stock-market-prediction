import numpy as np
import pandas as pd
import pytest

from backtesting import (
    INDIAN_DELIVERY_COST_SCENARIOS,
    BacktestInputMetadata,
    IndianDeliveryCostScenario,
    build_backtest_report,
    compare_backtest_reports,
    chronological_periods,
    get_indian_delivery_cost_scenario,
    run_backtest as _run_backtest,
    run_chronological_evaluation as _run_chronological_evaluation,
    summarize_returns,
)


def _synthetic_metadata(data):
    index = data.index if isinstance(data, (pd.Series, pd.DataFrame)) else pd.DatetimeIndex([])
    latest = pd.Timestamp(index.max()) if len(index) else pd.Timestamp("2026-01-01")
    return BacktestInputMetadata(
        source="Synthetic unit-test fixture",
        latest_observation_at=latest,
        retrieved_at=latest,
        evaluated_at=latest,
        freshness_status="fresh synthetic fixture",
        synthetic=True,
    )


def run_backtest(data, strategy, *args, **kwargs):
    kwargs.setdefault("input_metadata", _synthetic_metadata(data))
    return _run_backtest(data, strategy, *args, **kwargs)


def run_chronological_evaluation(data, strategy, *args, **kwargs):
    kwargs.setdefault("input_metadata", _synthetic_metadata(data))
    return _run_chronological_evaluation(data, strategy, *args, **kwargs)


def test_buy_and_hold_and_no_trade_baselines_and_costs():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 110.0, 121.0], index=dates, name="Adj Close")

    hold = run_backtest(
        prices,
        "buy_and_hold",
        transaction_cost_bps=10,
        slippage_bps=10,
    )
    no_trade = run_backtest(prices, "no_trade")

    assert hold.daily.loc[dates[0], "Trading Cost Fraction"] == pytest.approx(0.002)
    assert hold.metrics["cumulative_return"] == pytest.approx((1 - 0.002) * 1.1 * 1.1 - 1)
    assert hold.metrics["turnover"] == pytest.approx(1.0)
    assert hold.metrics["trade_count"] == 1
    assert hold.metrics["total_transaction_cost_fraction"] == pytest.approx(0.002)
    assert no_trade.metrics["cumulative_return"] == pytest.approx(0.0)
    assert no_trade.metrics["trade_count"] == 0
    assert no_trade.metrics["frictionless"] is True


def test_zero_cost_mode_is_conspicuous_and_preserves_legacy_zero_behavior():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 102.0, 99.0, 105.0], index=dates, name="Adj Close")

    default = run_backtest(prices, "buy_and_hold")
    legacy_zero = run_backtest(
        prices,
        "buy_and_hold",
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    pd.testing.assert_series_equal(default.daily["Net Return"], legacy_zero.daily["Net Return"])
    assert default.metrics["frictionless"] is True
    assert default.metrics["cost_mode"] == "ZERO COST / FRICTIONLESS"


def test_illustrative_preset_values_and_buy_sell_charge_asymmetry():
    low = get_indian_delivery_cost_scenario("LOW")
    medium = get_indian_delivery_cost_scenario("medium")
    high = get_indian_delivery_cost_scenario("high")

    assert tuple(INDIAN_DELIVERY_COST_SCENARIOS) == ("low", "medium", "high")
    assert low.side_cost_bps("buy") == pytest.approx(12.87406)
    assert low.side_cost_bps("sell") == pytest.approx(11.37406)
    assert medium.side_cost_bps("buy") == pytest.approx(19.23406)
    assert medium.side_cost_bps("sell") == pytest.approx(17.73406)
    assert high.side_cost_bps("buy") == pytest.approx(32.77406)
    assert high.side_cost_bps("sell") == pytest.approx(31.27406)
    assert low.stt_bps_buy == low.stt_bps_sell == 10.0
    assert low.stamp_duty_bps_buy == 1.5
    assert low.stamp_duty_bps_sell == 0


@pytest.mark.parametrize(
    ("name", "brokerage", "slippage"),
    [("low", 0.0, 1.0), ("medium", 2.0, 5.0), ("high", 5.0, 15.0)],
)
def test_preset_arithmetic_reconciles_each_component_on_buys_and_sells(
    name, brokerage, slippage
):
    scenario = get_indian_delivery_cost_scenario(name)

    # ₹307 per crore means 307 / 10,000,000 of turnover; multiply by
    # 10,000 to express the fraction in basis points.
    exchange_bps = 307 / 10_000_000 * 10_000
    assert exchange_bps == pytest.approx(0.307)
    assert scenario.exchange_bps_buy == pytest.approx(exchange_bps)
    assert scenario.exchange_bps_sell == pytest.approx(exchange_bps)

    taxable_service_base = brokerage + exchange_bps + 0.01
    gst = 0.18 * taxable_service_base
    expected_buy = brokerage + slippage + 10.0 + exchange_bps + 0.01 + 1.5 + gst
    expected_sell = brokerage + slippage + 10.0 + exchange_bps + 0.01 + gst
    assert scenario.side_cost_bps("buy") == pytest.approx(expected_buy)
    assert scenario.side_cost_bps("sell") == pytest.approx(expected_sell)


def test_cost_presets_apply_buy_and_sell_charges_on_the_correct_orders():
    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    prices = pd.Series(
        [10, 9, 8, 9, 12, 13, 14, 15, 10, 8, 7, 6],
        index=dates,
        dtype=float,
        name="Adj Close",
    )
    scenario = get_indian_delivery_cost_scenario("low")

    result = run_backtest(prices, "sma_crossover", fast_window=2, slow_window=3, cost_scenario=scenario)

    assert result.daily.loc[dates[5], "Cost Rate Bps"] == pytest.approx(scenario.side_cost_bps("buy"))
    assert result.daily.loc[dates[9], "Cost Rate Bps"] == pytest.approx(scenario.side_cost_bps("sell"))
    assert result.daily.loc[dates[5], "Trading Cost Fraction"] == pytest.approx(
        scenario.side_cost_bps("buy") / 10_000
    )
    assert result.daily.loc[dates[9], "Trading Cost Fraction"] == pytest.approx(
        scenario.side_cost_bps("sell") / 10_000
    )
    assert result.metrics["cost_scenario"] == "low"
    assert result.metrics["frictionless"] is False


def test_higher_cost_scenarios_cannot_improve_identical_buy_and_hold_trades():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.Series([100, 102, 98, 104, 110], index=dates, dtype=float, name="Adj Close")

    results = {
        name: run_backtest(prices, "buy_and_hold", cost_scenario=name)
        for name in ("low", "medium", "high")
    }
    assert all(
        results["low"].daily[column].equals(results["high"].daily[column])
        for column in ("Executed Position", "Held Position", "Turnover")
    )
    assert (
        results["low"].metrics["cumulative_return"]
        > results["medium"].metrics["cumulative_return"]
        > results["high"].metrics["cumulative_return"]
    )


def test_invalid_scenario_rates_and_mixing_with_legacy_inputs_are_rejected():
    prices = pd.Series([100, 101, 102], index=pd.date_range("2026-01-01", periods=3), dtype=float)
    with pytest.raises(ValueError, match="finite, nonnegative"):
        IndianDeliveryCostScenario("bad", -1, 0, 0, 0)
    with pytest.raises(ValueError, match="fraction between 0 and 1"):
        IndianDeliveryCostScenario("bad", 0, 0, 0, 0, gst_rate=18)
    with pytest.raises(ValueError, match="finite, nonnegative"):
        IndianDeliveryCostScenario("bad", 0, 0, float("inf"), 0)
    with pytest.raises(ValueError, match="one of: low, medium, high"):
        get_indian_delivery_cost_scenario("ultra")
    with pytest.raises(ValueError, match="either cost_scenario or legacy"):
        run_backtest(prices, "buy_and_hold", cost_scenario="medium", slippage_bps=1)
    excessive = IndianDeliveryCostScenario(
        "excessive", 10_000, 10_000, 0, 0,
        stt_bps_buy=0, stt_bps_sell=0, exchange_bps_buy=0,
        exchange_bps_sell=0, sebi_bps_buy=0, sebi_bps_sell=0,
        stamp_duty_bps_buy=0, stamp_duty_bps_sell=0, gst_rate=0,
    )
    with pytest.raises(ValueError, match="below 100%"):
        run_backtest(prices, "no_trade", cost_scenario=excessive)


def test_crossover_signal_is_after_close_execution_and_return_are_lagged():
    dates = pd.date_range("2026-01-01", periods=9, freq="D")
    prices = pd.Series([10, 9, 8, 9, 12, 13, 14, 15, 16], index=dates, dtype=float, name="Adj Close")

    result = run_backtest(prices, "sma_crossover", fast_window=2, slow_window=3)

    # Fast SMA crosses above slow SMA at the close on index 4.
    assert result.daily.loc[dates[4], "Signal Position"] == 1
    # It is executed at the next close, and only earns the following interval.
    assert result.daily.loc[dates[5], "Executed Position"] == 1
    assert result.daily.loc[dates[5], "Held Position"] == 0
    assert result.daily.loc[dates[5], "Gross Return"] == 0
    assert result.daily.loc[dates[6], "Held Position"] == 1
    assert result.daily.loc[dates[6], "Gross Return"] == pytest.approx(1 / 13)


def test_entry_and_exit_each_add_turnover_and_charge_cost_and_slippage():
    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    prices = pd.Series(
        [10, 9, 8, 9, 12, 13, 14, 15, 10, 8, 7, 6],
        index=dates,
        dtype=float,
        name="Adj Close",
    )

    result = run_backtest(
        prices,
        "sma_crossover",
        fast_window=2,
        slow_window=3,
        transaction_cost_bps=50,
        slippage_bps=50,
    )

    assert result.daily.loc[dates[5], "Turnover"] == 1
    assert result.daily.loc[dates[5], "Trading Cost Fraction"] == pytest.approx(0.01)
    assert result.daily.loc[dates[9], "Turnover"] == 1
    assert result.daily.loc[dates[9], "Trading Cost Fraction"] == pytest.approx(0.01)
    assert result.daily.loc[dates[10], "Held Position"] == 0
    assert result.metrics["turnover"] == pytest.approx(2.0)
    assert result.metrics["trade_count"] == 2
    assert result.metrics["total_transaction_cost_fraction"] == pytest.approx(0.02)
    assert result.daily.loc[dates[9], "Net Return"] == pytest.approx((1 - 0.20) * (1 - 0.01) - 1)
    assert set(result.daily["Executed Position"].unique()).issubset({0.0, 1.0})
    assert set(result.daily["Held Position"].unique()).issubset({0.0, 1.0})


def test_drawdown_volatility_cagr_and_costs_are_reported():
    dates = pd.date_range("2025-01-01", periods=5, freq="90D")
    prices = pd.Series([100, 120, 90, 110, 100], index=dates, dtype=float, name="Adj Close")

    result = run_backtest(prices, "buy_and_hold")

    assert result.metrics["maximum_drawdown"] == pytest.approx(90 / 120 - 1)
    assert result.metrics["annualized_volatility"] == pytest.approx(
        result.daily["Net Return"].std(ddof=1) * np.sqrt(252)
    )
    assert result.metrics["cagr"] is not None
    assert result.metrics["cumulative_return"] == pytest.approx(100 / 100 - 1)


def test_cagr_formula_and_single_observation_summary_edge_case():
    dates = pd.to_datetime(["2019-01-01", "2020-01-01"])
    prices = pd.Series([100.0, 121.0], index=dates, name="Adj Close")
    result = run_backtest(prices, "buy_and_hold")

    assert result.metrics["cagr"] == pytest.approx(1.21 ** (365.25 / 365) - 1)

    one_row = result.daily.iloc[[0]]
    summary = summarize_returns(one_row)
    assert summary["cagr"] is None
    assert summary["annualized_volatility"] is None
    assert summary["maximum_drawdown"] == pytest.approx(0.0)


def test_constant_prices_and_short_summaries_have_safe_metrics():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series([100.0] * len(dates), index=dates, name="Adj Close")

    result = run_backtest(prices, "buy_and_hold")
    summary = summarize_returns(result.daily.iloc[:1])

    assert result.metrics["gross_cumulative_return"] == pytest.approx(0.0)
    assert result.metrics["net_cumulative_return"] == pytest.approx(0.0)
    assert result.metrics["annualized_volatility"] == pytest.approx(0.0)
    assert result.metrics["maximum_drawdown"] == pytest.approx(0.0)
    assert result.metrics["trade_count"] == 1
    assert summary["cagr"] is None
    assert summary["annualized_volatility"] is None


def test_timezone_aware_daily_dates_are_converted_to_ist_then_sorted():
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-02 19:00", tz="UTC"), pd.Timestamp("2026-01-01 19:00", tz="UTC")]
    )
    prices = pd.Series([102.0, 100.0], index=dates, name="Adj Close")

    result = run_backtest(prices, "buy_and_hold")

    assert list(result.daily.index) == list(pd.to_datetime(["2026-01-02", "2026-01-03"]))
    assert list(result.daily["Adjusted Close"]) == [100.0, 102.0]


def test_timezone_aware_values_collapsing_to_duplicate_ist_dates_are_rejected():
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01 20:00", tz="UTC"), pd.Timestamp("2026-01-02 05:00", tz="UTC")]
    )
    prices = pd.Series([100.0, 101.0], index=dates, name="Adj Close")

    with pytest.raises(ValueError, match="duplicates after date normalization"):
        run_backtest(prices, "buy_and_hold")


def test_benchmark_alignment_compounds_strategy_over_exact_common_price_intervals():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    stock = pd.Series([100, 110, 121, 133.1], index=dates, dtype=float, name="Adj Close")
    benchmark = pd.Series([200, 220, 230, 242], index=dates, dtype=float, name="Adj Close")

    result = run_backtest(stock, "buy_and_hold", benchmark=benchmark)

    assert result.metrics["benchmark_matched_observations"] == 3
    assert result.metrics["strategy_return_on_benchmark_dates"] == pytest.approx(0.331)
    assert result.metrics["benchmark_cumulative_return"] == pytest.approx(0.21)
    assert result.metrics["benchmark_relative_return"] == pytest.approx(1.331 / 1.21 - 1)
    assert result.metrics["benchmark_period_start"] == dates[0]
    assert result.metrics["benchmark_period_end"] == dates[3]
    assert result.metrics["benchmark_source"] == "Unknown (no source label supplied)"
    assert result.metrics["benchmark_source_provenance"] == "unknown"
    assert result.metrics["benchmark_source_independently_verified"] is False


def test_benchmark_matched_return_includes_boundary_execution_cost_only():
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    stock = pd.Series([100.0, 110.0], index=dates, name="Adj Close")
    benchmark = pd.Series([200.0, 220.0], index=dates, name="Benchmark fixture")

    result = run_backtest(
        stock, "buy_and_hold", transaction_cost_bps=100, benchmark=benchmark
    )

    assert result.metrics["strategy_return_on_benchmark_dates"] == pytest.approx(
        1.10 * 0.99 - 1
    )


def test_backtest_report_exposes_cost_gross_net_benchmark_periods_and_caveats():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    stock = pd.Series(np.linspace(100, 110, len(dates)), index=dates, name="Adj Close")
    benchmark = pd.Series(
        np.linspace(200, 208, len(dates)), index=dates, name="NIFTY 50 fixture"
    )
    evaluation = run_chronological_evaluation(
        stock,
        "buy_and_hold",
        cost_scenario="medium",
        benchmark=benchmark,
        benchmark_source="Offline synthetic benchmark",
    )

    report = build_backtest_report(evaluation["backtest"], periods=evaluation["periods"])

    assert report["cost_description"].startswith("Indian delivery scenario: medium;")
    assert report["frictionless"] is False
    assert report["gross_cumulative_return"] > report["net_cumulative_return"]
    assert report["benchmark"]["source"] == "Offline synthetic benchmark"
    assert report["benchmark"]["source_provenance"] == "caller_asserted"
    assert report["benchmark"]["source_independently_verified"] is False
    assert report["benchmark"]["price_basis"] == "Adj Close"
    assert report["benchmark"]["date_policy"] == "Exact common price dates; no filling"
    assert report["benchmark"]["period_start"] == dates[0]
    assert report["benchmark"]["period_end"] == dates[-1]
    assert report["periods"]["development"]["interpretation"].startswith("Chronological development")
    assert "untouched only if" in report["periods"]["final_evaluation"]["interpretation"]
    assert any("not a forecast" in item for item in report["limitations"])


def test_backtest_report_labels_frictionless_results_and_has_no_benchmark_claim():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 101.0, 102.0], index=dates, name="Adj Close")
    result = run_backtest(prices, "buy_and_hold")

    report = build_backtest_report(result)

    assert report["cost_description"] == "ZERO COST / FRICTIONLESS"
    assert report["gross_cumulative_return"] == pytest.approx(
        report["net_cumulative_return"]
    )
    assert report["benchmark"] is None
    assert report["periods"] is None


def _comparison_report(
    strategy,
    *,
    cost_scenario="medium",
    benchmark_source="Fixture benchmark",
    benchmark_values=None,
):
    dates = pd.date_range("2026-03-01", periods=8, freq="D")
    prices = pd.Series([100, 102, 101, 104, 103, 105, 106, 108], index=dates, name="Adj Close")
    benchmark = pd.Series(
        benchmark_values or [200, 202, 203, 205, 206, 207, 208, 210],
        index=dates,
        name="Fixture",
    )
    result = run_backtest(
        prices,
        strategy,
        cost_scenario=cost_scenario,
        benchmark=benchmark,
        benchmark_source=benchmark_source,
    )
    return build_backtest_report(result)


def test_strategy_report_comparison_uses_explicit_equal_period_and_metrics():
    reports = {
        "hold": _comparison_report("buy_and_hold"),
        "flat": _comparison_report("no_trade"),
    }
    result = compare_backtest_reports(
        reports,
        period_start=pd.Timestamp("2026-02-28 19:00", tz="UTC"),
        period_end=pd.Timestamp("2026-03-08 10:00", tz="UTC"),
    )

    assert result["comparable"] is True
    assert result["warnings"] == []
    assert result["benchmark"][0] == "Fixture benchmark"
    assert [row["name"] for row in result["strategies"]] == ["hold", "flat"]
    assert set(result["strategies"][0]["metrics"]) == {
        "net_cumulative_return", "cagr", "maximum_drawdown",
        "annualized_volatility", "turnover", "trade_count",
        "benchmark_relative_return",
    }
    assert result["ranking"] is None


def test_strategy_report_comparison_flags_mixed_frictionless_and_costed_results():
    reports = {
        "free": _comparison_report("buy_and_hold", cost_scenario=None),
        "charged": _comparison_report("buy_and_hold", cost_scenario="medium"),
    }

    result = compare_backtest_reports(
        reports, period_start="2026-03-01", period_end="2026-03-08"
    )

    assert result["comparable"] is False
    assert any("Frictionless and costed" in warning for warning in result["warnings"])
    assert result["strategies"][0]["frictionless"] is True
    assert result["strategies"][1]["frictionless"] is False


def test_strategy_report_comparison_rejects_different_periods_or_benchmarks():
    first = _comparison_report("buy_and_hold")
    other_benchmark = _comparison_report("no_trade", benchmark_source="Other benchmark")
    with pytest.raises(ValueError, match="same benchmark"):
        compare_backtest_reports(
            {"first": first, "second": other_benchmark},
            period_start="2026-03-01",
            period_end="2026-03-08",
        )
    same_endpoint_other_path = _comparison_report(
        "no_trade",
        benchmark_values=[200, 210, 195, 203, 204, 207, 208, 210],
    )
    with pytest.raises(ValueError, match="same benchmark"):
        compare_backtest_reports(
            {"first": first, "second": same_endpoint_other_path},
            period_start="2026-03-01",
            period_end="2026-03-08",
        )
    with pytest.raises(ValueError, match="not the requested common period"):
        compare_backtest_reports(
            {"first": first, "second": {**first, "evaluation_end": pd.Timestamp("2026-03-07")}},
            period_start="2026-03-01",
            period_end="2026-03-08",
        )


def test_strategy_report_comparison_rejects_different_internal_observation_dates():
    first = _comparison_report("buy_and_hold")
    dates = pd.date_range("2026-03-01", periods=8, freq="D")
    prices = pd.Series([100, 102, 101, 104, 103, 105, 106, 108], index=dates, name="Adj Close")
    sparse_dates = dates.delete(3)
    benchmark = pd.Series(
        [200, 202, 203, 206, 207, 208, 210], index=sparse_dates, name="Fixture"
    )
    sparse = run_backtest(
        prices.drop(index=dates[3]),
        "no_trade",
        cost_scenario="medium",
        benchmark=benchmark,
        benchmark_source="Fixture benchmark",
    )
    second = build_backtest_report(sparse)

    with pytest.raises(ValueError, match="exact evaluation observation dates"):
        compare_backtest_reports(
            {"first": first, "second": second},
            period_start=dates[0],
            period_end=dates[-1],
        )


def test_strategy_report_comparison_marks_missing_metrics_and_preserves_input_order():
    first = _comparison_report("buy_and_hold")
    first.pop("annualized_volatility")
    second = _comparison_report("no_trade")
    reports = {"zeta": first, "alpha": second}

    result = compare_backtest_reports(
        reports, period_start="2026-03-01", period_end="2026-03-08"
    )

    assert [row["name"] for row in result["strategies"]] == ["zeta", "alpha"]
    assert result["strategies"][0]["metrics"]["annualized_volatility"] is None
    assert result["strategies"][0]["missing_metrics"] == ["annualized_volatility"]


def test_strategy_report_comparison_detects_different_legacy_cost_rates():
    dates = pd.date_range("2026-03-01", periods=8, freq="D")
    prices = pd.Series([100, 102, 101, 104, 103, 105, 106, 108], index=dates, name="Adj Close")
    reports = {}
    for name, cost in (("small fee", 5), ("larger fee", 20)):
        result = run_backtest(prices, "buy_and_hold", transaction_cost_bps=cost)
        reports[name] = build_backtest_report(result)

    comparison = compare_backtest_reports(
        reports, period_start=dates[0], period_end=dates[-1]
    )

    assert comparison["comparable"] is False
    assert "Cost assumptions differ" in comparison["warnings"][0]
    assert reports["small fee"]["cost_description"].endswith(
        "symmetric buy/sell 5.00000 bps per full turnover"
    )
    assert comparison["strategies"][0]["buy_cost_bps_per_full_turnover"] == 5.0


def test_subperiod_summary_excludes_pre_period_price_move_but_keeps_boundary_cost():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series([100, 110, 121, 133.1], index=dates, name="Adj Close")
    result = run_backtest(prices, "buy_and_hold", transaction_cost_bps=100)
    daily = result.daily.copy()
    # Simulate an entry on the selected segment boundary, whose row's gross
    # return also contains a pre-boundary price move.
    daily.loc[dates[1], "Trading Cost Fraction"] = 0.01
    daily.loc[dates[1], "Gross Return"] = 0.10
    daily.loc[dates[1], "Net Return"] = (1.10 * 0.99) - 1

    summary = summarize_returns(daily, start=dates[1], end=dates[2])

    assert summary["gross_cumulative_return"] == pytest.approx(0.10)
    assert summary["net_cumulative_return"] == pytest.approx(0.089)


def test_adjusted_basis_required_and_missing_duplicate_or_invalid_prices_are_rejected():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    with pytest.raises(ValueError, match="Adj Close"):
        run_backtest(pd.DataFrame({"Close": [1, 2, 3]}, index=dates), "buy_and_hold")
    with pytest.raises(ValueError, match="Missing or nonnumeric"):
        run_backtest(pd.Series([1.0, np.nan, 3.0], index=dates), "buy_and_hold")
    for invalid_price in (np.inf, -np.inf, "bad"):
        with pytest.raises(ValueError, match="Missing or nonnumeric|finite and positive"):
            run_backtest(pd.Series([1.0, invalid_price], index=dates[:2]), "buy_and_hold")
    with pytest.raises(ValueError, match="duplicates"):
        run_backtest(pd.Series([1.0, 2.0], index=[dates[0], dates[0]]), "buy_and_hold")
    with pytest.raises(ValueError, match="finite and positive"):
        run_backtest(pd.Series([1.0, 0.0], index=dates[:2]), "buy_and_hold")
    with pytest.raises(ValueError, match="At least two"):
        run_backtest(pd.Series([1.0], index=dates[:1]), "buy_and_hold")


def test_strategy_cost_and_sma_parameter_edge_cases_are_rejected():
    prices = pd.Series([100, 101, 102], index=pd.date_range("2026-01-01", periods=3), dtype=float)
    with pytest.raises(ValueError, match="strategy must"):
        run_backtest(prices, "short")
    with pytest.raises(ValueError, match="SMA windows"):
        run_backtest(prices, "sma_crossover", fast_window=5, slow_window=3)
    with pytest.raises(ValueError, match="cannot be negative"):
        run_backtest(prices, "buy_and_hold", slippage_bps=-1)
    with pytest.raises(ValueError, match="finite numeric"):
        run_backtest(prices, "buy_and_hold", transaction_cost_bps=float("nan"))


def test_sma_crossover_rejects_history_shorter_than_its_slow_window():
    prices = pd.Series(
        np.arange(100.0, 149.0),
        index=pd.date_range("2026-01-01", periods=49),
        name="Adj Close",
    )
    with pytest.raises(ValueError, match="requires at least 50 valid observations"):
        run_backtest(prices, "sma_crossover")


def test_irregular_observed_dates_are_not_filled_or_counted_as_intermediate_sessions():
    dates = pd.to_datetime(["2026-01-02", "2026-01-06"])
    prices = pd.Series([100.0, 110.0], index=dates, name="Adj Close")

    result = run_backtest(prices, "buy_and_hold")

    assert len(result.daily) == 2
    assert result.daily.loc[dates[1], "Asset Return"] == pytest.approx(0.10)
    assert result.metrics["observations"] == 2


def test_chronological_periods_keep_date_order_and_final_segment_separate():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    prices = pd.Series(np.arange(100, 120), index=dates, dtype=float, name="Adj Close")

    periods = chronological_periods(prices)

    assert periods["development"].index.max() < periods["validation"].index.min()
    assert periods["validation"].index.max() < periods["final_evaluation"].index.min()
    assert periods["development"].iloc[-1] == prices.iloc[11]
    assert periods["validation"].iloc[0] == prices.iloc[12]

    evaluation = run_chronological_evaluation(prices, "buy_and_hold")
    reports = evaluation["periods"]
    assert reports["development"]["end_date"] < reports["validation"]["start_date"]
    assert reports["validation"]["end_date"] < reports["final_evaluation"]["start_date"]
    assert all("cumulative_return" in report for report in reports.values())


def test_benchmark_rejects_missing_dates_and_nan_observations_without_filling():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    stock = pd.Series([100, 101, 102, 103], index=dates, name="Adj Close")
    with pytest.raises(ValueError, match="Benchmark dates must exactly match"):
        run_backtest(stock, "buy_and_hold", benchmark=stock.drop(index=dates[1]))
    benchmark_with_gap = pd.Series([200, np.nan, 202, 203], index=dates, name="Adj Close")
    with pytest.raises(ValueError, match="Missing or nonnumeric prices are not filled"):
        run_backtest(stock, "buy_and_hold", benchmark=benchmark_with_gap)


def test_benchmark_price_basis_and_unverified_caller_label_are_explicit():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    stock = pd.Series([100, 101, 102], index=dates, name="Adj Close")
    close_benchmark = pd.Series([200, 201, 202], index=dates, name="Close")
    with pytest.raises(ValueError, match="price basis conflicts"):
        run_backtest(stock, "buy_and_hold", benchmark=close_benchmark)
    with pytest.raises(ValueError, match="price basis must match"):
        run_backtest(stock, "buy_and_hold", benchmark=stock, benchmark_price_basis="Close")

    labeled = run_backtest(
        stock,
        "buy_and_hold",
        benchmark=stock,
        benchmark_source="Official exchange data (caller label)",
    )
    report = build_backtest_report(labeled)
    assert report["benchmark"]["source"] == "Official exchange data (caller label)"
    assert report["benchmark"]["source_provenance"] == "caller_asserted"
    assert report["benchmark"]["source_independently_verified"] is False
    assert report["benchmark"]["price_basis"] == "Adj Close"
    assert report["benchmark"]["price_basis_status"] == "assumed_from_price_column"

    unlabeled = build_backtest_report(
        run_backtest(stock, "buy_and_hold", benchmark=stock.rename("Adj Close"))
    )["benchmark"]
    assert unlabeled["source"] == "Unknown (no source label supplied)"
    assert unlabeled["source_provenance"] == "unknown"
    assert unlabeled["source_independently_verified"] is False

