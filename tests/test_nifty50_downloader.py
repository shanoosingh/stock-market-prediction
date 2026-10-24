from datetime import datetime, timezone
import pandas as pd
import pytest

import nifty50_downloader as downloader


def make_roster(symbols=("RELIANCE", "TCS")):
    frame = pd.DataFrame(
        {
            "Company Name": [f"{symbol} Ltd." for symbol in symbols],
            "Industry": ["Test"] * len(symbols),
            "Symbol": list(symbols),
            "Series": ["EQ"] * len(symbols),
            "ISIN Code": [f"TEST{index}" for index, _ in enumerate(symbols)],
        }
    )
    raw = frame.to_csv(index=False).encode("utf-8")
    return downloader.OfficialRoster(
        raw_csv=raw,
        csv_url="https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
        retrieved_at_utc="2026-10-24T12:00:00+00:00",
        frame=frame,
    )


def valid_history():
    index = pd.date_range("2022-01-03", periods=3, freq="B")
    return pd.DataFrame(
        {
            "Open": [99.0, 100.0, 101.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [98.0, 99.0, 100.0],
            "Close": [100.0, 101.0, 102.0],
            "Adj Close": [98.0, 99.0, 100.0],
            "Volume": [1000, 1200, 1400],
        },
        index=index,
    )


def test_maps_nse_symbol_to_yahoo_suffix_and_records_override():
    assert downloader.map_nse_to_yahoo(" reliance ") == ("RELIANCE.NS", None)
    assert downloader.map_nse_to_yahoo("M&M") == ("M&M.NS", None)
    ticker, exception = downloader.map_nse_to_yahoo("SPECIAL", {"SPECIAL": "SPECIAL-NEW.NS"})
    assert ticker == "SPECIAL-NEW.NS"
    assert exception == "Explicit Yahoo ticker override: SPECIAL -> SPECIAL-NEW.NS"


def test_official_roster_is_validated_without_changing_original_bytes():
    roster = make_roster()
    validated = downloader.validate_official_roster(
        roster.raw_csv, roster.csv_url, roster.retrieved_at_utc, expected_count=2
    )
    assert validated.raw_csv == roster.raw_csv
    assert list(validated.frame["Symbol"]) == ["RELIANCE", "TCS"]


def test_official_roster_rejects_wrong_constituent_count():
    roster = make_roster()
    with pytest.raises(downloader.RosterError, match="expected 50"):
        downloader.validate_official_roster(roster.raw_csv, roster.csv_url, roster.retrieved_at_utc)


def test_manifest_generation_records_rows_columns_dates_and_status():
    frame, problems, first, last = downloader.validate_history(valid_history())
    row = downloader.build_manifest_row(
        "Reliance Industries Ltd.", "RELIANCE", "RELIANCE.NS", None,
        "2026-10-24T12:00:00+00:00", frame, "success", validation_problems=problems,
        first_date=first, last_date=last,
    )
    assert row["row_count"] == 3
    assert row["first_date"] == "2022-01-03"
    assert row["last_date"] == "2022-01-05"
    assert "Adj Close" in row["returned_columns"]
    assert row["status"] == "success"


@pytest.mark.parametrize(
    ("response", "expected_problem"),
    [
        (None, "no DataFrame"),
        (pd.DataFrame(), "empty history"),
    ],
)
def test_empty_response_is_reported(response, expected_problem):
    frame, problems, first, last = downloader.validate_history(response)
    assert expected_problem in problems[0]
    assert first is None and last is None
    assert frame is None or frame.empty


def test_validation_reports_missing_dates_close_duplicates_and_bad_prices():
    no_dates = pd.DataFrame({"Close": [10.0, 11.0]})
    _, problems, _, _ = downloader.validate_history(no_dates)
    assert any("Missing dates" in issue for issue in problems)

    missing_close = pd.DataFrame({"Open": [10.0]}, index=pd.date_range("2024-01-01", periods=1))
    _, problems, _, _ = downloader.validate_history(missing_close)
    assert any("Missing required Close" in issue for issue in problems)

    bad_prices = pd.DataFrame(
        {"Close": [10.0, 0.0, float("inf"), -1.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-04"]),
    )
    _, problems, first, last = downloader.validate_history(bad_prices)
    assert any("duplicate dates" in issue for issue in problems)
    assert any("nonpositive price" in issue for issue in problems)
    assert any("non-finite" in issue for issue in problems)
    assert first == "2024-01-01" and last == "2024-01-04"


def test_multilevel_yfinance_columns_are_flattened():
    columns = pd.MultiIndex.from_product([["Open", "Close", "Adj Close"], ["RELIANCE.NS"]])
    frame = pd.DataFrame([[10, 11, 10.5]], index=pd.date_range("2024-01-01", periods=1), columns=columns)
    normalized, problems, _, _ = downloader.validate_history(frame)
    assert list(normalized.columns) == ["Open", "Close", "Adj Close"]
    assert problems == []


def test_download_one_uses_five_year_daily_adjusted_columns_and_timeout(monkeypatch):
    captured = {}
    class FakeSession:
        trust_env = False

        def close(self):
            captured["session_closed"] = True

    session = FakeSession()

    def fake_download(ticker, **kwargs):
        captured["ticker"] = ticker
        captured.update(kwargs)
        return valid_history()

    monkeypatch.setattr(downloader.yf, "download", fake_download)
    monkeypatch.setattr(downloader, "_new_yfinance_session", lambda: session)
    result = downloader._download_one("RELIANCE.NS")
    assert result is not None
    assert captured == {
        "ticker": "RELIANCE.NS",
        "period": "5y",
        "interval": "1d",
        "auto_adjust": False,
        "multi_level_index": False,
        "progress": False,
        "threads": False,
        "timeout": 20,
        "session": session,
        "session_closed": True,
    }


def test_yfinance_session_ignores_environment_proxies():
    session = downloader._new_yfinance_session()
    try:
        assert session.trust_env is False
    finally:
        session.close()


def test_run_download_preserves_roster_and_success_when_another_ticker_fails(tmp_path):
    output = tmp_path / "nifty-run"
    roster = make_roster()
    fixed_clock = lambda: datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    calls = []

    def fake_download(ticker):
        calls.append(ticker)
        if ticker == "TCS.NS":
            raise RuntimeError("mocked provider failure")
        return valid_history()

    result = downloader.run_download(
        output_dir=output,
        roster_fetcher=lambda: roster,
        download_one=fake_download,
        clock=fixed_clock,
    )
    assert result == output
    assert calls == ["RELIANCE.NS", "TCS.NS"]
    assert (output / "nifty50_constituents_source.csv").read_bytes() == roster.raw_csv
    assert (output / "history" / "01_RELIANCE.NS.csv").exists()
    manifest = pd.read_csv(output / "manifest.csv")
    assert list(manifest["status"]) == ["success", "failed"]
    assert manifest.loc[1, "error_details"].startswith("RuntimeError: mocked provider failure")
    assert manifest.loc[0, "row_count"] == 3


def test_roster_failure_stops_before_output_or_ticker_requests(tmp_path):
    output = tmp_path / "must-not-exist"
    requested = []

    def fail_roster():
        raise downloader.RosterError("official roster unavailable")

    with pytest.raises(downloader.RosterError, match="official roster unavailable"):
        downloader.run_download(
            output_dir=output,
            roster_fetcher=fail_roster,
            download_one=lambda ticker: requested.append(ticker),
        )
    assert not output.exists()
    assert requested == []


def test_nonempty_invalid_history_is_preserved_but_not_marked_success(tmp_path):
    output = tmp_path / "nifty-run"
    roster = make_roster(("RELIANCE",))
    bad_history = pd.DataFrame(
        {"Close": [100.0, 0.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"]),
    )
    downloader.run_download(
        output_dir=output,
        roster_fetcher=lambda: roster,
        download_one=lambda ticker: bad_history,
        clock=lambda: datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc),
    )
    assert (output / "history" / "01_RELIANCE.NS.csv").exists()
    manifest = pd.read_csv(output / "manifest.csv")
    assert manifest.loc[0, "status"] == "validation_failed"
    assert "duplicate dates" in manifest.loc[0, "validation_problems"]
    assert "nonpositive price" in manifest.loc[0, "validation_problems"]
