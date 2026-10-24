"""Streamlit rendering checks with a fake service; no provider or SQLite use."""

from pathlib import Path
import socket
import sqlite3

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import market_data_service
import market_data_storage
import backtesting


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_result(*, stale=False, missing_latest_adjusted=False):
    index = pd.date_range(end=pd.Timestamp.now(tz="Asia/Kolkata").normalize(), periods=30, freq="B")
    adjusted = pd.Series(range(100, 130), index=index, dtype=float)
    if missing_latest_adjusted:
        adjusted.iloc[-1] = float("nan")
    frame = pd.DataFrame(
        {
            "Open": adjusted * 0.99,
            "High": adjusted * 1.02,
            "Low": adjusted * 0.98,
            "Close": adjusted * 2,
            "Adj Close": adjusted,
            "Volume": [1000] * 29 + [4000],
        },
        index=index,
    )
    frame.attrs["quality_issues"] = ["synthetic data-quality notice"]
    return market_data_service.MarketDataResult(
        data=frame,
        source="Offline synthetic fixture",
        retrieved_at=pd.Timestamp.now(tz="Asia/Kolkata"),
        warnings=[],
        errors=["synthetic provider warning"] if stale else [],
        stale=stale,
    )


def synthetic_backtest_result(*, end=None, stale=False, missing_retrieval=False):
    end_date = pd.Timestamp.now(tz="Asia/Kolkata").normalize() if end is None else pd.Timestamp(end)
    if end_date.tzinfo is not None:
        end_date = end_date.tz_convert("Asia/Kolkata").tz_localize(None)
    index = pd.bdate_range(end=end_date, periods=65)
    values = pd.Series(150.0 + pd.Series(range(len(index)), index=index) * 0.35, index=index)
    frame = pd.DataFrame(
        {
            "Open": values * 0.995,
            "High": values * 1.02,
            "Low": values * 0.98,
            "Close": values * 1.01,
            "Adj Close": values,
            "Volume": 1000,
        },
        index=index,
    )
    frame.attrs["quality_issues"] = []
    frame.attrs["synthetic"] = True
    return market_data_service.MarketDataResult(
        data=frame,
        source="Offline synthetic NSE fixture",
        retrieved_at=None if missing_retrieval else pd.Timestamp.now(tz="Asia/Kolkata"),
        warnings=[],
        errors=["synthetic cached snapshot"] if stale else [],
        stale=stale,
    )


def empty_result():
    return market_data_service.MarketDataResult(
        data=None,
        source=None,
        retrieved_at=None,
        warnings=[],
        errors=["synthetic invalid ticker response"],
        stale=False,
    )


@pytest.fixture
def offline_app(monkeypatch):
    # Any accidental use of production retrieval or storage fails this test.
    monkeypatch.setattr(
        market_data_service.MarketDataService,
        "fetch",
        lambda *args, **kwargs: pytest.fail("Production provider fetch must not run in the offline UI test."),
    )
    monkeypatch.setattr(
        market_data_service,
        "fetch_openchart_data",
        lambda *args, **kwargs: pytest.fail("OpenChart must not be called in the offline UI test."),
    )
    monkeypatch.setattr(
        market_data_service,
        "fetch_stock_data",
        lambda *args, **kwargs: pytest.fail("Yahoo must not be called in the offline UI test."),
    )
    original_socket_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def block_external_socket(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if str(host).casefold() in {"127.0.0.1", "::1", "localhost"}:
            return original_socket_connect(self, address)
        pytest.fail(f"External network socket is blocked in the offline UI test: {host!r}.")

    def block_external_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if str(host).casefold() in {"127.0.0.1", "::1", "localhost"}:
            return original_create_connection(address, *args, **kwargs)
        pytest.fail(f"External network connection is blocked in the offline UI test: {host!r}.")

    monkeypatch.setattr(socket.socket, "connect", block_external_socket)
    monkeypatch.setattr(socket, "create_connection", block_external_connection)

    backtest_calls = []
    original_run_validated_backtest = backtesting.run_validated_backtest

    def observed_backtest(*args, **kwargs):
        backtest_calls.append((args, kwargs))
        return original_run_validated_backtest(*args, **kwargs)

    monkeypatch.setattr(backtesting, "run_validated_backtest", observed_backtest)
    monkeypatch.setattr(
        market_data_storage,
        "save_candles",
        lambda *args, **kwargs: pytest.fail("SQLite persistence must not run in the offline UI test."),
    )
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("SQLite connections must not run in the offline UI test."),
    )

    def run(result):
        # AppTest reruns a cached app function; clear only Streamlit's in-memory
        # test cache so each case receives its own synthetic result.
        st.cache_resource.clear()
        backtest_calls.clear()

        service_calls = []

        class FakeService:
            def fetch(self, *args, **kwargs):
                service_calls.append((args, kwargs))
                return result

        monkeypatch.setattr(market_data_service, "MarketDataService", FakeService)
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        app.backtest_calls = backtest_calls
        app.service_calls = service_calls
        return app

    return run


def test_dashboard_uses_adjusted_close_for_overview_and_shows_analysis(offline_app):
    app = offline_app(synthetic_result())

    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    latest_adjusted = 129.0
    assert any("Latest valid adjusted close" in label for label in metrics)
    latest_metric = next(value for label, value in metrics.items() if "Latest valid adjusted close" in label)
    assert latest_metric == f"{latest_adjusted:,.2f}"
    assert "30 candles" in " ".join(item.value for item in app.caption)
    assert any("Recent performance" in item.value for item in app.subheader)
    assert any("Risk overview" in item.value for item in app.subheader)
    assert any("Benchmark comparison" in item.value for item in app.subheader)
    assert any("synthetic data-quality notice" in item.value for item in app.warning)
    metric_values = list(metrics.values())
    assert any(value.endswith("%") for value in metric_values)
    assert any("20-session annualized volatility" in label for label in metrics)
    volume_metric = next(value for label, value in metrics.items() if "Volume / prior 20-session" in label)
    assert volume_metric == "4.00×"


def test_dashboard_marks_stale_snapshot_and_missing_adjusted_tail(offline_app):
    app = offline_app(synthetic_result(stale=True, missing_latest_adjusted=True))

    assert not app.exception
    messages = [item.value for item in app.error] + [item.value for item in app.warning]
    assert any("STALE SNAPSHOT" in message for message in messages)
    assert any("most recent row has no valid selected price" in message for message in messages)
    assert any("last observation has no usable price" in message for message in messages)
    metrics = {item.label: item.value for item in app.metric}
    assert any("STALE snapshot Adj Close" in label for label in metrics)
    assert any(value == "Insufficient clean data" for value in metrics.values())


def test_dashboard_stops_analysis_on_nonpositive_selected_price(offline_app):
    result = synthetic_result()
    result.data.iloc[-2, result.data.columns.get_loc("Adj Close")] = 0.0

    app = offline_app(result)

    assert any(
        "Selected price data is insufficient or invalid for analysis" in item.value
        and "nonpositive" in item.value
        for item in app.error
    )


def test_dashboard_backtest_is_opt_in_and_does_not_add_fetch_or_storage_calls(offline_app):
    app = offline_app(synthetic_backtest_result())
    app.selectbox[0].select("India")
    app.run()

    assert not app.exception
    assert any("Experimental daily backtest" in item.value for item in app.subheader)
    assert app.backtest_calls == []
    assert len(app.service_calls) == 2  # Initial run plus the existing market-selector rerun.
    assert any("Chronological model evaluation" in item.value for item in app.subheader)


def test_dashboard_backtest_runs_only_after_submit_and_reports_costed_daily_result(offline_app):
    app = offline_app(synthetic_backtest_result())
    app.selectbox[0].select("India")
    app = app.run()
    calls_before_submit = len(app.service_calls)
    next(button for button in app.button if button.label == "Run selected backtest").click().run()

    assert not app.exception
    assert len(app.backtest_calls) == 1
    assert any("Backtest report" in item.value for item in app.subheader)
    assert any("medium" in item.value.lower() for item in app.info)
    assert any("fresh_validated" in item.value for item in app.caption)
    assert len(app.service_calls) == calls_before_submit
    args, kwargs = app.backtest_calls[0]
    assert args[1] == "buy_and_hold"
    assert kwargs["input_metadata"].interval == "1d"
    assert kwargs["input_metadata"].source == "Offline synthetic NSE fixture"
    assert kwargs["input_metadata"].retrieved_at is not None
    assert kwargs["input_metadata"].synthetic is True
    assert str(kwargs["input_metadata"].evaluated_at.tzinfo) == "Asia/Kolkata"
    expected_latest = pd.Timestamp(app.backtest_calls[0][0][0].index[-1]).tz_localize("Asia/Kolkata")
    assert kwargs["input_metadata"].latest_observation_at == expected_latest
    assert kwargs["cost_scenario"] == "medium"


def test_dashboard_routes_selected_strategy_dates_basis_and_cost_to_engine(offline_app):
    result = synthetic_backtest_result()
    app = offline_app(result)
    app.selectbox[0].select("India")
    app.run()
    start = result.data.index[-55].date()
    end = result.data.index[-1].date()
    next(item for item in app.selectbox if item.label == "Backtest strategy").select(
        "sma_crossover"
    )
    next(item for item in app.date_input if item.label.startswith("Evaluation date range")).set_value(
        (start, end)
    )
    next(item for item in app.selectbox if item.label == "Backtest price basis").select("Close")
    next(item for item in app.selectbox if item.label == "Illustrative NSE delivery cost").select("High")
    next(button for button in app.button if button.label == "Run selected backtest").click().run()

    assert len(app.backtest_calls) == 1
    data, strategy = app.backtest_calls[0][0]
    options = app.backtest_calls[0][1]
    assert strategy == "sma_crossover"
    assert len(data) == 55
    assert data.index.min() == pd.Timestamp(start)
    assert data.index.max() == pd.Timestamp(end)
    assert options["price_column"] == "Close"
    assert options["cost_scenario"] == "high"
    assert "Backtest report" in [item.value for item in app.subheader]


def test_invalid_ticker_without_loaded_data_stops_before_backtest(offline_app):
    app = offline_app(empty_result())
    app.selectbox[0].select("India")
    app.run()
    next(item for item in app.text_input if item.label == "Or enter a ticker").set_value(
        "INVALID.NS"
    )
    app.run()

    assert any("No successful dataset" in item.value for item in app.error)
    assert app.backtest_calls == []
    assert not any("Run selected backtest" == item.label for item in app.button)
    assert app.service_calls[-1][0][0] == "INVALID.NS"


def test_sma_backtest_rejects_insufficient_selected_history(offline_app):
    result = synthetic_backtest_result()
    result.data = result.data.iloc[-40:].copy()
    app = offline_app(result)
    app.selectbox[0].select("India")
    app.run()
    next(item for item in app.selectbox if item.label == "Backtest strategy").select(
        "sma_crossover"
    )
    next(button for button in app.button if button.label == "Run selected backtest").click().run()

    assert len(app.backtest_calls) == 1
    assert any("requires at least 50 valid observations" in item.value for item in app.error)
    assert not any("Backtest report" in item.value for item in app.subheader)


def test_dashboard_labels_zero_cost_choice_as_frictionless(offline_app):
    app = offline_app(synthetic_backtest_result())
    app.selectbox[0].select("India")
    app.run()
    next(item for item in app.selectbox if item.label == "Illustrative NSE delivery cost").select(
        "ZERO COST / FRICTIONLESS"
    )
    next(button for button in app.button if button.label == "Run selected backtest").click().run()

    assert app.backtest_calls[0][1]["cost_scenario"] is None
    assert any("ZERO COST / FRICTIONLESS" in item.value for item in app.error)


def test_historical_backtest_requires_explicit_stale_override_and_keeps_provenance_required(offline_app):
    old_data = synthetic_backtest_result(end="2025-01-31")
    app = offline_app(old_data)
    app.selectbox[0].select("India")
    app = app.run()
    next(button for button in app.button if button.label == "Run selected backtest").click().run()

    assert len(app.backtest_calls) == 1
    assert any("stale_rejected" in item.value for item in app.error)

    next(item for item in app.checkbox if item.label.startswith("Allow stale data")).check()
    next(button for button in app.button if button.label == "Run selected backtest").click().run()
    assert len(app.backtest_calls) == 2
    assert app.backtest_calls[-1][1]["allow_stale_override"] is True
    assert any("stale_override" in item.value for item in [*app.caption, *app.warning, *app.info])

    unknown = synthetic_backtest_result(missing_retrieval=True)
    app_unknown = offline_app(unknown)
    app_unknown.selectbox[0].select("India")
    app_unknown = app_unknown.run()
    next(item for item in app_unknown.checkbox if item.label.startswith("Allow stale data")).check()
    next(button for button in app_unknown.button if button.label == "Run selected backtest").click().run()
    assert len(app_unknown.backtest_calls) == 1
    assert any(
        "unknown" in item.value.lower() or "retrieval" in item.value.lower()
        for item in [*app_unknown.error, *app_unknown.warning]
    )


def test_dashboard_does_not_offer_intraday_backtesting_controls(offline_app):
    app = offline_app(synthetic_backtest_result())
    app.selectbox[0].select("India")
    app.selectbox[2].select("5-minute (5m)")
    app.run()

    assert not app.exception
    assert any("only for daily data" in item.value for item in app.info)
    assert not any(button.label == "Run selected backtest" for button in app.button)
    assert app.backtest_calls == []


def test_stale_override_does_not_bypass_invalid_backtest_prices(offline_app):
    result = synthetic_backtest_result()
    result.data.iloc[10, result.data.columns.get_loc("Adj Close")] = float("nan")
    app = offline_app(result)
    app.selectbox[0].select("India")
    app.run()
    next(item for item in app.checkbox if item.label.startswith("Allow stale data")).check()
    next(button for button in app.button if button.label == "Run selected backtest").click().run()

    assert len(app.backtest_calls) == 1
    assert not any("Backtest report" in item.value for item in app.subheader)
    assert any("Missing or nonnumeric prices are not filled" in item.value for item in app.error)


def _upload_synthetic_benchmark(app, result, *, omit_one_date=False, backtest_basis=None):
    frame = result.data
    dates = list(frame.index)
    values = frame["Adj Close"].to_numpy()
    if omit_one_date:
        del dates[10]
        values = values[[index for index in range(len(values)) if index != 10]]
    csv_data = pd.DataFrame(
        {"Date": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates], "Adj Close": values}
    ).to_csv(index=False).encode("utf-8")
    app.file_uploader[0].set_value(("nifty_fixture.csv", csv_data, "text/csv"))
    app.run()
    if backtest_basis is not None:
        next(item for item in app.selectbox if item.label == "Backtest price basis").select(
            backtest_basis
        )
    next(
        item for item in app.checkbox
        if item.label.startswith("Compare with uploaded benchmark")
    ).check()
    next(button for button in app.button if button.label == "Run selected backtest").click().run()
    return app


def test_backtest_panel_reports_uploaded_benchmark_as_unverified_and_aligned(offline_app):
    result = synthetic_backtest_result()
    app = offline_app(result)
    app.selectbox[0].select("India")
    app.run()
    app = _upload_synthetic_benchmark(app, result)

    assert len(app.backtest_calls) == 1
    _, kwargs = app.backtest_calls[0]
    assert kwargs["benchmark"].index.equals(result.data.index)
    assert kwargs["benchmark_price_basis"] == "Adj Close"
    assert "caller-provided; unverified" in kwargs["benchmark_source"]
    assert any("does not authenticate" in item.value for item in app.warning)


def test_backtest_panel_rejects_uploaded_benchmark_with_missing_dates(offline_app):
    result = synthetic_backtest_result()
    app = offline_app(result)
    app.selectbox[0].select("India")
    app.run()
    app = _upload_synthetic_benchmark(app, result, omit_one_date=True)

    assert len(app.backtest_calls) == 1
    assert not any("Backtest report" in item.value for item in app.subheader)
    assert any("Benchmark dates must exactly match" in item.value for item in app.error)


def test_backtest_panel_rejects_benchmark_price_basis_mismatch(offline_app):
    result = synthetic_backtest_result()
    app = offline_app(result)
    app.selectbox[0].select("India")
    app.run()
    app = _upload_synthetic_benchmark(app, result, backtest_basis="Close")

    assert len(app.backtest_calls) == 1
    assert not any("Backtest report" in item.value for item in app.subheader)
    assert any("Benchmark price basis must match" in item.value for item in app.error)
