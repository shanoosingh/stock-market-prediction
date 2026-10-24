from types import ModuleType

import pandas as pd
import pytest

from market_data_service import (
    MarketDataService,
    fetch_openchart_data,
    map_nse_symbol,
    normalize_openchart_data,
)
from market_data_storage import get_stored_history, save_candles


@pytest.fixture(autouse=True)
def avoid_writing_test_market_data_to_the_project(monkeypatch):
    """Provider fakes must not populate the user's default persistent store."""
    monkeypatch.setattr("market_data_service.save_candles", lambda *args, **kwargs: 0)


def sample_data(index=None):
    index = index if index is not None else pd.date_range("2026-01-05 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1200],
        },
        index=index,
    )


def test_known_nse_tickers_map_to_exact_equity_symbols():
    assert map_nse_symbol("RELIANCE.NS") == "RELIANCE-EQ"
    assert map_nse_symbol("TCS.NS") == "TCS-EQ"
    assert map_nse_symbol("INFY.NS") == "INFY-EQ"
    assert map_nse_symbol("AAPL") is None


def test_openchart_normalization_localizes_naive_timestamps_as_ist():
    raw = sample_data(pd.DatetimeIndex(["2026-01-05 09:15", "2026-01-05 09:20"], name="Timestamp"))

    normalized = normalize_openchart_data(raw, "TCS-EQ")

    assert str(normalized.index.tz) == "Asia/Kolkata"
    assert normalized.index[0] == pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")
    assert list(normalized.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_openchart_normalization_rejects_missing_ohlcv_schema():
    with pytest.raises(ValueError, match="missing required OHLCV columns"):
        normalize_openchart_data(pd.DataFrame({"Close": [1]}, index=["2026-01-05"]), "TCS-EQ")


def test_openchart_fetch_uses_get_query_and_exact_nse_match(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "status": True,
                "data": [
                    {"symbol": "RELIANCE COMMUNICATIONS-EQ", "scripcode": "bad", "description": "Similar name", "type": "Equity", "exchange": "NSE"},
                    {"symbol": "RELIANCE-EQ", "scripcode": "2885", "description": "Reliance Industries", "type": "Equity", "exchange": "NSE"},
                    {"symbol": "RELIANCE-EQ", "scripcode": "wrong", "description": "Wrong exchange", "type": "Equity", "exchange": "BSE"},
                    {"symbol": "RELIANCE-EQ", "scripcode": "wrong", "description": "Index", "type": "Index", "exchange": "NSE"},
                ],
            }

    class FakeClient:
        def __init__(self):
            self.historical_args = None
            self.search_args = None
            self.search_url = "https://charting.nseindia.com/v1/exchanges/symbolsDynamic"
            self.session = self

        def _ensure_cookies(self):
            pass

        def get(self, url, **kwargs):
            self.search_args = (url, kwargs)
            return FakeResponse()

        def historical_direct(self, **kwargs):
            self.historical_args = kwargs
            return sample_data(pd.DatetimeIndex(["2026-01-05 09:15", "2026-01-05 09:20"], name="Timestamp"))

    client = FakeClient()
    fake_module = ModuleType("openchart")
    fake_module.NSEData = lambda: client
    monkeypatch.setitem(__import__("sys").modules, "openchart", fake_module)

    result = fetch_openchart_data("RELIANCE.NS", "5d", None, None, "5m")

    assert client.historical_args["token"] == "2885"
    assert client.historical_args["symbol"] == "RELIANCE-EQ"
    assert client.historical_args["symbol_type"] == "Equity"
    assert client.historical_args["interval"] == "5m"
    assert client.historical_args["start"].tzinfo is not None
    assert client.historical_args["end"].tzinfo is not None
    assert client.search_args == (
        "https://charting.nseindia.com/v1/exchanges/symbolsDynamic",
        {"params": {"symbol": "RELIANCE", "segment": "EQ"}, "timeout": 10},
    )
    assert str(result.index.tz) == "Asia/Kolkata"
    assert result.attrs["source"].startswith("OpenChart")


def test_openchart_ambiguous_or_non_exact_match_is_rejected(monkeypatch):
    class FakeClient:
        search_url = "https://charting.nseindia.com/v1/exchanges/symbolsDynamic"
        _ensure_cookies = lambda self: None

        class Session:
            def get(self, *args, **kwargs):
                class Response:
                    def raise_for_status(self): pass
                    def json(self):
                        return {"status": True, "data": [
                            {"symbol": "RELIANCE COMMUNICATIONS-EQ", "scripcode": "1", "description": "Similar name", "type": "Equity", "exchange": "NSE"}
                        ]}
                return Response()

        def __init__(self): self.session = self.Session()

    fake_module = ModuleType("openchart")
    fake_module.NSEData = FakeClient
    monkeypatch.setitem(__import__("sys").modules, "openchart", fake_module)

    with pytest.raises(ValueError, match="0 exact NSE equity matches"):
        fetch_openchart_data("RELIANCE.NS", "1mo", None, None, "1d")


def test_openchart_search_connection_error_is_not_reported_as_missing_symbol(monkeypatch):
    class FakeSession:
        def get(self, *args, **kwargs):
            raise OSError("HTTPSConnectionPool: [WinError 10013] access forbidden")

    class FakeClient:
        def __init__(self):
            self.session = FakeSession()
            self.search_url = "https://charting.nseindia.com/v1/exchanges/symbolsDynamic"

        def _ensure_cookies(self): pass

    fake_module = ModuleType("openchart")
    fake_module.NSEData = FakeClient
    monkeypatch.setitem(__import__("sys").modules, "openchart", fake_module)

    with pytest.raises(RuntimeError, match="search request failed.*WinError 10013"):
        fetch_openchart_data("RELIANCE.NS", "1d", None, None, "5m")

    result = MarketDataService(
        fetch_openchart_data,
        lambda ticker, **kwargs: sample_data(),
    ).fetch("RELIANCE.NS", "1d", interval="5m")
    assert result.source == "Yahoo Finance (yfinance)"
    assert result.data is not None
    assert any("search request failed" in error and "WinError 10013" in error for error in result.errors)


def test_openchart_empty_search_result_is_reported_as_no_exact_symbol(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"status": False, "data": []}

    class FakeClient:
        search_url = "https://charting.nseindia.com/v1/exchanges/symbolsDynamic"
        _ensure_cookies = lambda self: None

        class Session:
            def get(self, *args, **kwargs): return Response()

        def __init__(self): self.session = self.Session()

    fake_module = ModuleType("openchart")
    fake_module.NSEData = FakeClient
    monkeypatch.setitem(__import__("sys").modules, "openchart", fake_module)

    with pytest.raises(ValueError, match="did not find exact NSE equity symbol"):
        fetch_openchart_data("RELIANCE.NS", "1d", None, None, "5m")


def test_openchart_malformed_search_schema_is_reported(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"status": True, "data": [{"symbol": "RELIANCE-EQ", "scripcode": "2885"}]}

    class FakeClient:
        search_url = "https://charting.nseindia.com/v1/exchanges/symbolsDynamic"
        _ensure_cookies = lambda self: None

        class Session:
            def get(self, *args, **kwargs): return Response()

        def __init__(self): self.session = self.Session()

    fake_module = ModuleType("openchart")
    fake_module.NSEData = FakeClient
    monkeypatch.setitem(__import__("sys").modules, "openchart", fake_module)

    with pytest.raises(ValueError, match="malformed results; missing columns: description, exchange, type"):
        fetch_openchart_data("RELIANCE.NS", "1d", None, None, "5m")


@pytest.mark.parametrize("payload", [[], {"status": True}, {"status": True, "data": {"symbol": "RELIANCE-EQ"}}])
def test_openchart_malformed_search_payload_is_rejected(monkeypatch, payload):
    class Response:
        def raise_for_status(self): pass
        def json(self): return payload

    class FakeClient:
        search_url = "https://charting.nseindia.com/v1/exchanges/symbolsDynamic"
        _ensure_cookies = lambda self: None

        class Session:
            def get(self, *args, **kwargs): return Response()

        def __init__(self): self.session = self.Session()

    fake_module = ModuleType("openchart")
    fake_module.NSEData = FakeClient
    monkeypatch.setitem(__import__("sys").modules, "openchart", fake_module)

    with pytest.raises(ValueError, match="malformed response"):
        fetch_openchart_data("RELIANCE.NS", "1d", None, None, "5m")


def test_openchart_failure_falls_back_to_yahoo_and_keeps_provider_identity():
    def fail_openchart(*args):
        raise RuntimeError("offline")

    yahoo_calls = []

    def yahoo(ticker, **kwargs):
        yahoo_calls.append((ticker, kwargs))
        return sample_data()

    result = MarketDataService(fail_openchart, yahoo).fetch("TCS.NS", "5d", interval="5m")

    assert result.data is not None
    assert result.source == "Yahoo Finance (yfinance)"
    assert result.errors and "OpenChart" in result.errors[0]
    assert len(yahoo_calls) == 1


def test_empty_or_invalid_openchart_response_falls_back_to_yahoo():
    yahoo_calls = []

    def empty_openchart(*args):
        raise ValueError("OpenChart returned no rows.")

    def yahoo(ticker, **kwargs):
        yahoo_calls.append(ticker)
        return sample_data()

    result = MarketDataService(empty_openchart, yahoo).fetch("RELIANCE.NS", "5d", interval="5m")

    assert result.source == "Yahoo Finance (yfinance)"
    assert result.data is not None
    assert yahoo_calls == ["RELIANCE.NS"]
    assert any("OpenChart" in error for error in result.errors)


def test_both_providers_failing_returns_no_fabricated_dataset():
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise ConnectionError("offline")

    service = MarketDataService(fail, fail, cooldown_seconds=30, clock=lambda: 100.0)
    result = service.fetch("INFY.NS", "1d", interval="1d")

    assert result.data is None
    assert len(result.errors) == 2
    second = service.fetch("INFY.NS", "1d", interval="1d")
    assert second.data is None
    assert len(second.errors) == 2
    assert len(calls) == 2


def test_per_provider_cooldown_avoids_duplicate_requests_and_retains_last_good():
    now = [100.0]
    calls = {"openchart": 0, "yahoo": 0}

    def openchart(*args):
        calls["openchart"] += 1
        if calls["openchart"] == 1:
            data = sample_data()
            data.attrs["source"] = "OpenChart"
            data.attrs["retrieved_at"] = pd.Timestamp.now(tz="Asia/Kolkata")
            return data
        raise RuntimeError("OpenChart unavailable")

    def yahoo(*args, **kwargs):
        calls["yahoo"] += 1
        raise RuntimeError("Yahoo unavailable")

    service = MarketDataService(openchart, yahoo, cooldown_seconds=30, clock=lambda: now[0])
    first = service.fetch("INFY.NS", "5d", interval="5m")
    now[0] += 5
    within_cooldown = service.fetch("INFY.NS", "5d", interval="5m")
    assert calls == {"openchart": 1, "yahoo": 0}
    assert within_cooldown.data is not None
    assert any("cooldown active" in warning for warning in within_cooldown.warnings)

    now[0] += 30
    after_failure = service.fetch("INFY.NS", "5d", interval="5m")
    assert calls == {"openchart": 2, "yahoo": 1}
    assert after_failure.stale
    assert after_failure.source == first.source == "OpenChart"
    pd.testing.assert_frame_equal(after_failure.data, first.data)


def test_yahoo_last_good_keeps_prior_openchart_error_during_cooldown():
    now = [100.0]
    calls = {"openchart": 0, "yahoo": 0}

    def openchart(*args):
        calls["openchart"] += 1
        raise ValueError("OpenChart returned 0 exact NSE equity matches for RELIANCE-EQ")

    def yahoo(*args, **kwargs):
        calls["yahoo"] += 1
        return sample_data()

    service = MarketDataService(openchart, yahoo, cooldown_seconds=30, clock=lambda: now[0])
    initial = service.fetch("RELIANCE.NS", "1d", interval="1d")
    now[0] += 11
    during_cooldown = service.fetch("RELIANCE.NS", "1d", interval="1d")

    assert initial.source == during_cooldown.source == "Yahoo Finance (yfinance)"
    assert during_cooldown.stale
    assert any("OpenChart request cooldown active (19s remaining)" in warning for warning in during_cooldown.warnings)
    assert any("Yahoo Finance (yfinance) request cooldown active (19s remaining)" in warning for warning in during_cooldown.warnings)
    assert any("0 exact NSE equity matches" in error for error in during_cooldown.errors)
    assert calls == {"openchart": 1, "yahoo": 1}


def test_empty_openchart_response_is_a_provider_failure():
    with pytest.raises(ValueError, match="no rows"):
        normalize_openchart_data(pd.DataFrame(), "TCS-EQ")


def test_successful_fetch_persists_only_the_selected_provider_source(tmp_path, monkeypatch):
    database = tmp_path / "service.sqlite3"

    def openchart(*args):
        data = sample_data()
        data.attrs["source"] = "OpenChart (NSE historical chart data)"
        return data

    def persist(data, ticker, interval, source):
        return save_candles(data, ticker, interval, source, database)

    monkeypatch.setattr("market_data_service.save_candles", persist)
    result = MarketDataService(openchart, lambda *args, **kwargs: pytest.fail("fallback should not run")).fetch(
        "RELIANCE.NS", "5d", interval="5m"
    )

    restored = get_stored_history("RELIANCE.NS", "5m", "OpenChart (NSE historical chart data)", database)
    assert result.source == "OpenChart (NSE historical chart data)"
    pd.testing.assert_frame_equal(restored, result.data, check_freq=False, check_dtype=False)
    assert "retrieved_at" not in restored.attrs
