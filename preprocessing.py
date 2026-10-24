"""Validation and cleanup for daily market data."""

from __future__ import annotations

import pandas as pd

REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
INDIA_TZ = "Asia/Kolkata"


def assess_data_quality(data: pd.DataFrame | None) -> list[str]:
    """Describe source-data issues without correcting or filling the data."""
    if data is None or data.empty:
        return ["No rows were returned by the data provider."]

    issues: list[str] = []
    if data.index.has_duplicates:
        issues.append(f"{int(data.index.duplicated(keep=False).sum())} rows have duplicate timestamps.")
    if not data.index.is_monotonic_increasing:
        issues.append("Timestamps arrived out of chronological order.")
    missing_columns = [column for column in REQUIRED_OHLCV_COLUMNS if column not in data.columns]
    if missing_columns:
        issues.append("Missing OHLCV columns: " + ", ".join(missing_columns) + ".")

    numeric: dict[str, pd.Series] = {}
    numeric_columns = REQUIRED_OHLCV_COLUMNS + (("Adj Close",) if "Adj Close" in data.columns else ())
    for column in numeric_columns:
        if column not in data:
            continue
        original = data[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        non_finite = converted.notna() & ~converted.map(lambda value: pd.notna(value) and abs(value) != float("inf"))
        count_invalid = int(invalid.sum() + non_finite.sum())
        if count_invalid:
            issues.append(f"{column} contains {count_invalid} invalid numeric value(s).")
        missing = int(converted.isna().sum())
        if missing:
            issues.append(f"{column} is missing in {missing} row(s); values were not filled.")
        numeric[column] = converted

    def report_invalid_rows(label: str, invalid: pd.Series) -> None:
        count = int(invalid.sum())
        if count:
            timestamps = ", ".join(str(value) for value in data.index[invalid][:5])
            suffix = f" Affected timestamps include: {timestamps}." if timestamps else ""
            issues.append(f"{label} in {count} row(s).{suffix}")

    if all(column in numeric for column in ("Open", "High", "Low", "Close")):
        report_invalid_rows(
            "High is below Low",
            numeric["High"].notna() & numeric["Low"].notna() & (numeric["High"] < numeric["Low"]),
        )
        for column in ("Open", "Close"):
            report_invalid_rows(
                f"High is below {column}",
                numeric["High"].notna() & numeric[column].notna() & (numeric["High"] < numeric[column]),
            )
            report_invalid_rows(
                f"Low is above {column}",
                numeric["Low"].notna() & numeric[column].notna() & (numeric["Low"] > numeric[column]),
            )
    for column in ("Open", "High", "Low", "Close", "Adj Close"):
        if column in numeric:
            report_invalid_rows(
                f"{column} is nonpositive",
                numeric[column].notna() & (numeric[column] <= 0),
            )
    if "Volume" in numeric:
        report_invalid_rows("Volume is negative", numeric["Volume"].notna() & (numeric["Volume"] < 0))
    return issues


def get_retrieval_time(data: pd.DataFrame) -> pd.Timestamp | None:
    """Return provider retrieval time in IST when it is present in metadata."""
    value = data.attrs.get("retrieved_at")
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(INDIA_TZ)
    return timestamp.tz_convert(INDIA_TZ)


def format_retrieval_time(data: pd.DataFrame) -> str | None:
    """Format retrieval metadata for a dashboard caption, explicitly in IST."""
    timestamp = get_retrieval_time(data)
    return None if timestamp is None else f"{timestamp:%Y-%m-%d %H:%M:%S} IST (Asia/Kolkata)"


def get_latest_candle(data: pd.DataFrame, price_column: str = "Close") -> tuple[pd.Timestamp | None, float | None]:
    """Return the row at the latest timestamp and its price when available."""
    if data is None or data.empty:
        return None, None
    timestamp = pd.Timestamp(data.index.max())
    row = data.loc[timestamp]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    if price_column not in row or pd.isna(row[price_column]):
        return timestamp, None
    price = float(row[price_column])
    return timestamp, price if pd.notna(price) and abs(price) != float("inf") else None


def preprocess_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return validated, sorted market data without fabricating missing prices.

    Duplicate dates retain the last observation. Numeric columns are coerced;
    rows with missing close prices are retained as gaps. A dataset with no
    valid close in any row is rejected. Other gaps stay NaN.
    Adjusted prices account for distributions and splits when supplied by the
    provider; they are useful for returns but differ from historical traded closes.
    """
    if data is None or data.empty:
        raise ValueError("Market data is empty.")
    metadata = dict(data.attrs)
    issues = list(metadata.get("quality_issues", []))
    for issue in assess_data_quality(data):
        if issue not in issues:
            issues.append(issue)
    result = data.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()].copy()
    # Keep timezone-aware intraday timestamps intact; data_collection normalizes
    # Indian Yahoo symbols to Asia/Kolkata before they reach preprocessing.
    result.index.name = "Date"
    result.columns = [str(c).strip() for c in result.columns]
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    close_cols = [c for c in ("Close", "Adj Close") if c in result]
    if not close_cols:
        raise ValueError("Market data must contain Close or Adj Close.")
    if not result[close_cols].notna().to_numpy().any():
        raise ValueError("Market data contains no valid closing prices.")
    result.attrs.update(metadata)
    result.attrs["quality_issues"] = issues
    return result
