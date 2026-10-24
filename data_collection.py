"""Download daily or recent intraday historical market data from Yahoo Finance."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf
from preprocessing import assess_data_quality

INDIA_TZ = "Asia/Kolkata"
SUPPORTED_INTERVALS = {"1d", "5m", "15m"}
SUPPORTED_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
INTRADAY_PERIODS = {"1d", "5d", "1mo"}

# yfinance stores cookies and timezone metadata in SQLite under the current
# user's cache directory by default. That location is not writable in every
# Windows/managed environment, so keep its cache alongside this project.
YFINANCE_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "yfinance"
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


def _exchange_time(value: date | str, timezone: str) -> pd.Timestamp:
    """Interpret date-only inputs in the exchange timezone for range checks."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def fetch_stock_data(
    ticker: str,
    start: date | str | None = None,
    end: date | str | None = None,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch daily or 5/15-minute historical OHLCV data.

    Explicit date ranges use Yahoo's exclusive end. Yahoo limits intraday
    history to the most recent 60 days; actual symbol/interval availability
    can be shorter. Intraday timestamps are retained and Indian symbols are
    converted/localized to Asia/Kolkata.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Enter a ticker symbol.")
    interval = interval.strip().lower()
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError("Choose a supported interval: 1d, 5m, or 15m.")
    if (start is None) != (end is None):
        raise ValueError("Provide both start and end dates, or neither.")
    if start is not None and pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("The start date must be earlier than the end date.")
    if start is None and period not in SUPPORTED_PERIODS:
        raise ValueError("Choose a supported historical period.")
    is_intraday = interval != "1d"
    if is_intraday:
        if start is None and period not in INTRADAY_PERIODS:
            raise ValueError("Yahoo Finance intraday history supports only 1d, 5d, or 1mo period requests in this app.")
        if start is not None:
            exchange_tz = INDIA_TZ if symbol.endswith((".NS", ".BO")) else "UTC"
            now = pd.Timestamp.now(tz=exchange_tz)
            range_start = _exchange_time(start, exchange_tz)
            range_end = _exchange_time(end, exchange_tz)
            if range_start < now - pd.Timedelta(days=60):
                raise ValueError("Yahoo Finance limits 5m/15m intraday history to the most recent 60 days. Choose a newer start date.")
            if range_start > now:
                raise ValueError("The intraday start date cannot be in the future.")
            # Date pickers commonly use tomorrow as the exclusive end to include today's bars.
            if range_end > now.normalize() + pd.Timedelta(days=1):
                raise ValueError("The intraday end date cannot be more than one day in the future.")
    options: dict[str, object] = {
        "tickers": symbol, "interval": interval, "auto_adjust": False,
        "progress": False, "threads": False,
    }
    if start is not None:
        options.update(start=start, end=end)
    else:
        options["period"] = period
    try:
        data = yf.download(**options)
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance could not retrieve {symbol}: {exc}") from exc
    retrieved_at = pd.Timestamp.now(tz=INDIA_TZ)
    if data is None or data.empty:
        kind = "intraday" if is_intraday else "daily"
        raise ValueError(
            f"Yahoo Finance returned no {kind} data for {symbol} at interval {interval}. "
            "The request may have failed because of a connection problem, or the symbol/history "
            "may be unavailable. Check the app terminal for Yahoo Finance's provider error."
        )
    metadata = dict(data.attrs)
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance may put ticker or field on either level depending on version.
        for level in range(data.columns.nlevels):
            values = data.columns.get_level_values(level)
            if symbol in values:
                data.columns = data.columns.get_level_values(1 - level if data.columns.nlevels == 2 else 0)
                break
        else:
            data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).strip() for c in data.columns]
    quality_issues = assess_data_quality(data)
    data.index = pd.to_datetime(data.index, errors="coerce")
    invalid_timestamps = int(data.index.isna().sum())
    if invalid_timestamps:
        quality_issues.append(f"{invalid_timestamps} rows had invalid timestamps and were excluded.")
    data = data.loc[~data.index.isna()].copy()
    if data.index.tz is not None and symbol.endswith((".NS", ".BO")):
        data.index = data.index.tz_convert(INDIA_TZ)
    elif is_intraday and data.index.tz is None and symbol.endswith((".NS", ".BO")):
        data.index = data.index.tz_localize(INDIA_TZ)
    elif is_intraday and data.index.tz is None:
        raise ValueError(f"Yahoo Finance returned timezone-naive intraday timestamps for {symbol}; refusing to guess their timezone.")
    data.index.name = "Date"
    data = data.sort_index()
    if data.empty or not any(c in data.columns for c in ("Close", "Adj Close")):
        raise ValueError(f"Yahoo Finance returned no usable closing prices for {symbol}.")
    data.attrs.update(metadata)
    data.attrs["retrieved_at"] = retrieved_at
    data.attrs["quality_issues"] = quality_issues
    return data
