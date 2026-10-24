"""Provider selection, normalization, cooldown, and last-good data retention."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from threading import RLock
import time as monotonic_time
from typing import Callable

import pandas as pd

from data_collection import INDIA_TZ, fetch_stock_data
from market_data_storage import MarketDataStorageError, save_candles
from preprocessing import assess_data_quality, preprocess_market_data

NSE_SYMBOL_MAP = {
    "RELIANCE.NS": "RELIANCE-EQ",
    "TCS.NS": "TCS-EQ",
    "INFY.NS": "INFY-EQ",
}
OPENCHART_INTERVALS = {"1d", "5m", "15m"}
OPENCHART_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
    "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
}


@dataclass
class MarketDataResult:
    data: pd.DataFrame | None
    source: str | None
    retrieved_at: pd.Timestamp | None
    warnings: list[str]
    errors: list[str]
    stale: bool = False
    cooldown_remaining: int = 0


def map_nse_symbol(ticker: str) -> str | None:
    """Map Yahoo-style NSE symbols to exact OpenChart EQ symbols.

    The known examples are explicit. Other .NS symbols are resolved by exact
    NSE equity matching at request time; substring/first-result matching is
    never accepted.
    """
    symbol = ticker.strip().upper()
    if not symbol.endswith(".NS"):
        return None
    return NSE_SYMBOL_MAP.get(symbol, f"{symbol[:-3]}-EQ")


def _as_ist_datetime(value: date | str | datetime | pd.Timestamp) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(INDIA_TZ)
    else:
        stamp = stamp.tz_convert(INDIA_TZ)
    return stamp.to_pydatetime()


def _period_start(period: str, end: datetime) -> datetime:
    if period == "ytd":
        start = pd.Timestamp(datetime(end.year, 1, 1), tz=INDIA_TZ)
        return start.to_pydatetime()
    days = OPENCHART_PERIOD_DAYS.get(period, 1825)
    return (pd.Timestamp(end) - pd.Timedelta(days=days)).to_pydatetime()


def normalize_openchart_data(data: pd.DataFrame, source_symbol: str) -> pd.DataFrame:
    """Normalize the OpenChart OHLCV frame while interpreting naive NSE times as IST."""
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("OpenChart returned no rows.")
    result = data.copy()
    if isinstance(result.columns, pd.MultiIndex):
        raise ValueError("OpenChart returned an unsupported multi-level column layout.")
    if "Timestamp" in result.columns:
        result = result.set_index("Timestamp")
    result.columns = [str(column).strip().title() for column in result.columns]
    # OpenChart 0.2.0 documents Open/High/Low/Close/Volume indexed by timestamp.
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError("OpenChart is missing required OHLCV columns: " + ", ".join(sorted(missing)))
    index = pd.to_datetime(result.index, errors="coerce")
    invalid = index.isna()
    invalid_count = int(invalid.sum())
    if invalid.any():
        result = result.loc[~invalid].copy()
        index = index[~invalid]
    if index.tz is None:
        index = index.tz_localize(INDIA_TZ)
    else:
        index = index.tz_convert(INDIA_TZ)
    result.index = index
    result.index.name = "Date"
    quality_issues = assess_data_quality(result)
    if invalid_count:
        quality_issues.append(f"{invalid_count} rows had invalid timestamps and were excluded.")
    result.attrs["quality_issues"] = quality_issues
    result.attrs["openchart_symbol"] = source_symbol
    return result.sort_index()


def _strict_openchart_symbol(client, exact_symbol: str) -> dict:
    """Resolve one exact NSE equity using the endpoint's GET query contract.

    OpenChart 0.2.0's ``NSEData.search`` posts JSON and can return default
    results that do not correspond to the requested equity. The current NSE
    dynamic-search contract uses GET query parameters. Keep using the
    package-created session for its headers/cookies, but parse and validate
    the response here so request errors cannot be mistaken for no matches.
    """
    session = getattr(client, "session", None)
    search_url = getattr(client, "search_url", None)
    if session is None or not search_url or not callable(getattr(session, "get", None)):
        raise ValueError("OpenChart client is missing its search session or endpoint.")

    search_query = exact_symbol[:-3] if exact_symbol.upper().endswith("-EQ") else exact_symbol
    try:
        ensure_cookies = getattr(client, "_ensure_cookies", None)
        if callable(ensure_cookies):
            ensure_cookies()
        response = session.get(
            search_url,
            params={"symbol": search_query, "segment": "EQ"},
            timeout=10,
        )
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"OpenChart symbol search request failed: {exc}") from exc

    try:
        payload = response.json()
    except Exception as exc:
        raise ValueError(f"OpenChart symbol search returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("OpenChart symbol search returned a malformed response; expected a JSON object.")
    raw_rows = payload.get("data")
    if raw_rows is None:
        if payload.get("status") is False:
            raw_rows = []
        else:
            raise ValueError("OpenChart symbol search returned a malformed response; missing data list.")
    if not isinstance(raw_rows, list):
        raise ValueError("OpenChart symbol search returned a malformed response; data must be a list.")
    matches = pd.DataFrame(raw_rows)
    required_columns = {"symbol", "scripcode", "description", "type", "exchange"}
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns and raw_rows:
        raise ValueError(
            "OpenChart symbol search returned malformed results; missing columns: "
            + ", ".join(sorted(missing_columns))
            + "."
        )
    if matches.empty:
        raise ValueError(f"OpenChart did not find exact NSE equity symbol {exact_symbol}.")
    rows = matches[
        matches["symbol"].astype(str).str.strip().str.upper().eq(exact_symbol.upper())
        & matches["exchange"].astype(str).str.strip().str.upper().eq("NSE")
        & matches["type"].astype(str).str.strip().str.casefold().eq("equity")
    ]
    if len(rows) != 1:
        raise ValueError(f"OpenChart returned {len(rows)} exact NSE equity matches for {exact_symbol}; refusing approximate symbol matches.")
    return rows.iloc[0].to_dict()


def fetch_openchart_data(ticker: str, period: str, start, end, interval: str) -> pd.DataFrame:
    """Fetch via the released OpenChart NSEData API, with exact symbol resolution."""
    if interval not in OPENCHART_INTERVALS:
        raise ValueError(f"OpenChart does not support this dashboard interval: {interval}.")
    exact_symbol = map_nse_symbol(ticker)
    if exact_symbol is None:
        raise ValueError("OpenChart is only used for NSE .NS equity tickers.")
    try:
        from openchart import NSEData
    except ImportError as exc:
        raise RuntimeError("OpenChart package is not installed in this Python environment.") from exc

    client = NSEData()
    instrument = _strict_openchart_symbol(client, exact_symbol)
    now = pd.Timestamp.now(tz=INDIA_TZ).to_pydatetime()
    request_end = _as_ist_datetime(end) if end is not None else now
    request_start = _as_ist_datetime(start) if start is not None else _period_start(period, request_end)
    if interval != "1d" and request_start < now - timedelta(days=60):
        request_start = now - timedelta(days=60)
    frame = client.historical_direct(
        token=str(instrument["scripcode"]),
        symbol=str(instrument["symbol"]),
        symbol_type=str(instrument["type"]),
        start=request_start,
        end=request_end,
        interval=interval,
    )
    normalized = normalize_openchart_data(frame, str(instrument["symbol"]))
    issues = list(normalized.attrs.get("quality_issues", []))
    for issue in assess_data_quality(normalized):
        if issue not in issues:
            issues.append(issue)
    # Structural and non-numeric failures make this provider response unusable;
    # ordinary gaps/ordering/duplicate notices remain visible after cleanup.
    if not normalized[["Open", "High", "Low", "Close", "Volume"]].notna().any().all():
        raise ValueError("OpenChart returned OHLCV columns without usable values.")
    if normalized[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
        raise ValueError("OpenChart returned missing OHLCV values; trying the Yahoo Finance fallback.")
    checked_values = {}
    for column in ("Open", "High", "Low", "Close", "Volume"):
        raw = normalized[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numeric.isna()).any() or (numeric.notna() & ~numeric.map(lambda v: abs(v) != float("inf"))).any():
            raise ValueError(f"OpenChart returned invalid numeric data in {column}.")
        checked_values[column] = numeric
    if (checked_values["High"] < checked_values["Low"]).any():
        raise ValueError("OpenChart returned rows where High is below Low.")
    if (checked_values["Volume"] < 0).any():
        raise ValueError("OpenChart returned negative volume values.")
    if (pd.DataFrame({c: checked_values[c] for c in ("Open", "High", "Low", "Close")}) <= 0).any().any():
        raise ValueError("OpenChart returned non-positive prices.")
    normalized.attrs["source"] = "OpenChart (NSE historical chart data)"
    normalized.attrs["retrieved_at"] = pd.Timestamp.now(tz=INDIA_TZ)
    normalized.attrs["quality_issues"] = issues
    return preprocess_market_data(normalized)


class MarketDataService:
    """Provider cooldown plus per-request last-good snapshot retention."""

    def __init__(
        self,
        openchart_fetcher: Callable = fetch_openchart_data,
        yahoo_fetcher: Callable = fetch_stock_data,
        cooldown_seconds: int = 30,
        clock: Callable[[], float] = monotonic_time.monotonic,
    ):
        self.openchart_fetcher = openchart_fetcher
        self.yahoo_fetcher = yahoo_fetcher
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.last_attempt: dict[tuple[str, str, str], float] = {}
        self.last_good: dict[tuple, MarketDataResult] = {}
        self.last_errors: dict[tuple, list[str]] = {}
        self.lock = RLock()

    @staticmethod
    def _request_key(ticker, period, start, end, interval):
        return (ticker.strip().upper(), period, str(start), str(end), interval)

    def _can_request(self, provider: str, ticker: str, interval: str, now: float) -> tuple[bool, int]:
        key = (provider, ticker.strip().upper(), interval)
        last = self.last_attempt.get(key)
        if last is None:
            return True, 0
        elapsed = now - last
        remaining = max(0, int(self.cooldown_seconds - elapsed + 0.999))
        return remaining == 0, remaining

    def fetch(self, ticker: str, period: str, start=None, end=None, interval="1d") -> MarketDataResult:
        key = self._request_key(ticker, period, start, end, interval)
        with self.lock:
            now = self.clock()
            previous = self.last_good.get(key)
            errors: list[str] = []
            warnings: list[str] = []
            if previous is not None and previous.source.startswith("OpenChart") and map_nse_symbol(ticker) is not None:
                allowed, remaining = self._can_request("OpenChart", ticker, interval, now)
                if not allowed:
                    warnings.append(f"OpenChart request cooldown active ({remaining}s remaining); using its cached snapshot.")
                    return MarketDataResult(
                        data=previous.data.copy(), source=previous.source,
                        retrieved_at=previous.retrieved_at, warnings=warnings + previous.warnings,
                        errors=previous.errors, stale=True, cooldown_remaining=remaining,
                    )
            providers = []
            if map_nse_symbol(ticker) is not None:
                providers.append(("OpenChart", self.openchart_fetcher))
            providers.append(("Yahoo Finance (yfinance)", self.yahoo_fetcher))

            for provider_name, fetcher in providers:
                provider_key = "OpenChart" if provider_name == "OpenChart" else "Yahoo Finance"
                allowed, remaining = self._can_request(provider_key, ticker, interval, now)
                if not allowed:
                    warnings.append(f"{provider_name} request cooldown active ({remaining}s remaining).")
                    continue
                self.last_attempt[(provider_key, ticker.strip().upper(), interval)] = now
                try:
                    if provider_key == "OpenChart":
                        data = fetcher(ticker, period, start, end, interval)
                    else:
                        data = fetcher(ticker, start=start, end=end, period=period, interval=interval)
                    if data is None or data.empty:
                        raise ValueError(f"{provider_name} returned empty data.")
                    if provider_key == "OpenChart":
                        source = data.attrs.get("source", "OpenChart")
                    else:
                        source = "Yahoo Finance (yfinance)"
                    data = data.copy()
                    data.attrs.update(getattr(data, "attrs", {}))
                    retrieved_at = data.attrs.get("retrieved_at", pd.Timestamp.now(tz=INDIA_TZ))
                    retrieved_at = pd.Timestamp(retrieved_at)
                    if retrieved_at.tzinfo is None:
                        retrieved_at = retrieved_at.tz_localize(INDIA_TZ)
                    else:
                        retrieved_at = retrieved_at.tz_convert(INDIA_TZ)
                    data.attrs["source"] = source
                    data.attrs["retrieved_at"] = retrieved_at
                    data = preprocess_market_data(data)
                    storage_warnings = []
                    try:
                        save_candles(data, ticker, interval, source)
                    except MarketDataStorageError as exc:
                        # Persistence must not turn a valid provider response into
                        # a failed dashboard fetch. Keep the exact source result.
                        storage_warnings.append(f"Local candle storage failed: {exc}")
                    result = MarketDataResult(
                        data=data, source=source, retrieved_at=retrieved_at,
                        warnings=list(data.attrs.get("quality_issues", [])) + storage_warnings,
                        errors=errors,
                    )
                    result.data.attrs["source"] = source
                    self.last_good[key] = result
                    self.last_errors[key] = list(errors)
                    return result
                except Exception as exc:
                    errors.append(f"{provider_name}: {type(exc).__name__}: {exc}")

            if errors:
                self.last_errors[key] = list(errors)
            if previous is not None:
                warnings.extend(previous.warnings)
                return MarketDataResult(
                    data=previous.data.copy(), source=previous.source,
                    retrieved_at=previous.retrieved_at, warnings=warnings,
                    errors=self.last_errors.get(key, previous.errors + errors), stale=True,
                )
            return MarketDataResult(
                data=None, source=None, retrieved_at=None, warnings=warnings,
                errors=self.last_errors.get(key, errors), stale=False,
            )
