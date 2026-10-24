"""SQLite persistence for provider-specific historical OHLCV candles.

Timestamps are stored as UTC instants plus the timezone used to interpret the
source index. Naive NSE timestamps are interpreted in Asia/Kolkata; other
naive indexes default to UTC unless a timezone is supplied explicitly.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from preprocessing import REQUIRED_OHLCV_COLUMNS, assess_data_quality

PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATABASE_PATH = PROJECT_DATA_DIR / "market_data.sqlite3"
SCHEMA_VERSION = 1


class MarketDataStorageError(RuntimeError):
    """Raised when a stored candle operation cannot be completed safely."""


def _timezone_for(ticker: str, explicit_timezone: str | None) -> str:
    timezone = explicit_timezone or ("Asia/Kolkata" if ticker.upper().endswith((".NS", ".BO")) else "UTC")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise MarketDataStorageError(f"Unknown timezone {timezone!r}.") from exc
    return timezone


def _to_utc(value: Any, timezone: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise MarketDataStorageError("Candle timestamps must be valid.")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone)
    return stamp.tz_convert("UTC")


def _connect(database_path: str | Path | None) -> sqlite3.Connection:
    path = Path(database_path) if database_path is not None else DEFAULT_DATABASE_PATH
    connection: sqlite3.Connection | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                ticker TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                timezone TEXT NOT NULL,
                source TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (ticker, interval, timestamp_utc, source)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_candles_lookup "
            "ON candles (ticker, interval, source, timestamp_utc)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS storage_metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO storage_metadata(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (str(SCHEMA_VERSION),),
        )
        connection.commit()
        return connection
    except (OSError, sqlite3.Error) as exc:
        if connection is not None:
            connection.close()
        raise MarketDataStorageError(f"Could not open candle database: {exc}") from exc


def save_candles(
    data: pd.DataFrame,
    ticker: str,
    interval: str,
    source: str,
    database_path: str | Path | None = None,
    timezone: str | None = None,
) -> int:
    """Insert or deterministically replace candles; return input row count.

    Return the number of unique timestamps written. A repeated key is updated
    with the latest saved OHLCV values (last write wins). Provider is part of
    the primary key, so sources never mix.
    """
    if not ticker or not interval or not source:
        raise MarketDataStorageError("Ticker, interval, and source are required.")
    if data is None or data.empty:
        raise MarketDataStorageError("Cannot store an empty candle dataset.")
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in data.columns]
    if missing:
        raise MarketDataStorageError("Cannot store candles; missing OHLCV columns: " + ", ".join(missing) + ".")

    if timezone is not None:
        market_timezone = _timezone_for(ticker, timezone)
    elif ticker.upper().endswith((".NS", ".BO")):
        market_timezone = "Asia/Kolkata"
    elif isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        market_timezone = _timezone_for(ticker, str(data.index.tz))
    else:
        market_timezone = "UTC"
    frame = data.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if frame.index.isna().any():
        raise MarketDataStorageError("Cannot store candles with invalid timestamps.")
    frame.index = pd.DatetimeIndex([_to_utc(value, market_timezone) for value in frame.index])
    frame.index.name = "timestamp_utc"
    # Resolve duplicates deterministically in input order, then query/store in
    # chronological order. Re-fetching an existing primary key remains upsert.
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    for column in REQUIRED_OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    problems = assess_data_quality(frame)
    if problems:
        raise MarketDataStorageError("Cannot store data with quality issues: " + " ".join(problems))
    if frame.loc[:, list(REQUIRED_OHLCV_COLUMNS)].isna().any().any():
        raise MarketDataStorageError("Cannot store candles with missing OHLCV values.")

    normalized_ticker = ticker.strip().upper()
    rows = []
    for timestamp, row in frame.iterrows():
        rows.append((
            normalized_ticker,
            interval,
            timestamp.isoformat(),
            market_timezone,
            source,
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"]),
        ))

    connection = _connect(database_path)
    try:
        connection.executemany(
            """
            INSERT INTO candles (
                ticker, interval, timestamp_utc, timezone, source,
                open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, interval, timestamp_utc, source) DO UPDATE SET
                timezone=excluded.timezone,
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume
            """,
            rows,
        )
        connection.commit()
        return len(rows)
    except sqlite3.Error as exc:
        connection.rollback()
        raise MarketDataStorageError(f"Could not save candles: {exc}") from exc
    finally:
        connection.close()


def query_date_range(
    ticker: str,
    interval: str,
    source: str,
    start: date | datetime | str | pd.Timestamp | None = None,
    end: date | datetime | str | pd.Timestamp | None = None,
    database_path: str | Path | None = None,
    timezone: str | None = None,
) -> pd.DataFrame:
    """Query one ticker/interval/source; start inclusive and end exclusive."""
    if not ticker or not interval or not source:
        raise MarketDataStorageError("Ticker, interval, and source are required.")
    clauses = ["ticker = ?", "interval = ?", "source = ?"]
    params: list[Any] = [ticker.strip().upper(), interval, source]
    connection = _connect(database_path)
    try:
        stored_timezone = None
        if timezone is None:
            timezone_rows = connection.execute(
                "SELECT DISTINCT timezone FROM candles WHERE ticker = ? AND interval = ? AND source = ?",
                (ticker.strip().upper(), interval, source),
            ).fetchmany(2)
            if len(timezone_rows) == 1:
                stored_timezone = timezone_rows[0][0]
        market_timezone = _timezone_for(ticker, timezone or stored_timezone)
        if start is not None:
            clauses.append("timestamp_utc >= ?")
            params.append(_to_utc(start, market_timezone).isoformat())
        if end is not None:
            clauses.append("timestamp_utc < ?")
            params.append(_to_utc(end, market_timezone).isoformat())
        rows = connection.execute(
            "SELECT timestamp_utc, timezone, open, high, low, close, volume FROM candles WHERE "
            + " AND ".join(clauses) + " ORDER BY timestamp_utc",
            params,
        ).fetchall()
    except sqlite3.Error as exc:
        raise MarketDataStorageError(f"Could not query candles: {exc}") from exc
    finally:
        connection.close()

    if not rows:
        empty = pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
        empty.index = pd.DatetimeIndex([], name="Date", tz=market_timezone)
        empty.attrs["source"] = source
        return empty

    timezones = {row[1] for row in rows}
    if len(timezones) != 1:
        raise MarketDataStorageError("Stored candles for this key contain inconsistent timezones.")
    stored_timezone = next(iter(timezones))
    try:
        index = pd.DatetimeIndex([pd.Timestamp(row[0]) for row in rows]).tz_convert(stored_timezone)
    except (ValueError, TypeError) as exc:
        raise MarketDataStorageError(f"Stored candle timestamps are invalid: {exc}") from exc
    frame = pd.DataFrame(
        [row[2:] for row in rows],
        columns=list(REQUIRED_OHLCV_COLUMNS),
        index=index,
    )
    frame.index.name = "Date"
    frame.attrs["source"] = source
    frame.attrs["stored_timezone"] = stored_timezone
    frame.attrs["stored_data"] = True
    return frame


def get_stored_history(
    ticker: str,
    interval: str,
    source: str,
    database_path: str | Path | None = None,
    timezone: str | None = None,
) -> pd.DataFrame:
    """Retrieve all stored history for one exact provider-specific key."""
    return query_date_range(
        ticker=ticker,
        interval=interval,
        source=source,
        database_path=database_path,
        timezone=timezone,
    )
