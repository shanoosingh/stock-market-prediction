"""Offline investing-analysis metrics for already available market data.

These helpers never fetch data or persist it. Price returns should use adjusted
close when the provider supplies it, because adjusted prices account for
distributions and splits while raw close represents the historical traded
price.
"""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO

import numpy as np
import pandas as pd

INDIA_TZ = "Asia/Kolkata"
TRADING_SESSIONS_PER_YEAR = 252
VOLATILITY_WINDOW = 20
DRAWDOWN_MOVE_WARNING = 0.20
VOLUME_SPIKE_MULTIPLE = 3.0
VOLUME_LOW_MULTIPLE = 1 / 3


def _price_series(
    data: pd.DataFrame | pd.Series,
    price_column: str = "Adj Close",
    *,
    daily_dates: bool = False,
) -> pd.Series:
    if isinstance(data, pd.Series):
        raw = data.copy()
    elif isinstance(data, pd.DataFrame) and price_column in data.columns:
        raw = data[price_column].copy()
    else:
        raise ValueError(f"A {price_column!r} price series is required.")
    index = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="coerce"))
    if index.isna().any():
        raise ValueError("Price dates contain invalid timestamps.")
    if daily_dates:
        if index.tz is not None:
            index = index.tz_convert(INDIA_TZ).tz_localize(None)
        index = index.normalize()
    if index.has_duplicates:
        raise ValueError("Price dates contain duplicates.")
    values = pd.to_numeric(raw, errors="coerce").astype(float)
    invalid = values.notna() & (~np.isfinite(values) | (values <= 0))
    if invalid.any():
        raise ValueError("Prices must be finite and positive when present.")
    result = pd.Series(values.to_numpy(), index=index, name=price_column).sort_index()
    return result


def _invalid_numeric_price_count(data: pd.DataFrame, price_column: str) -> int:
    """Count supplied price values that cannot be parsed as finite numbers."""
    if price_column not in data:
        return 0
    raw = data[price_column]
    numeric = pd.to_numeric(raw, errors="coerce")
    return int((raw.notna() & (numeric.isna() | ~np.isfinite(numeric))).sum())


def calculate_investing_metrics(
    data: pd.DataFrame,
    price_column: str = "Adj Close",
    *,
    recent_sessions: Iterable[int] = (1, 5, 20),
    volatility_window: int = VOLATILITY_WINDOW,
) -> dict[str, object]:
    """Calculate recent returns, 20-session volatility, drawdown, and warnings.

    Recent returns compound the last N observed one-session returns and are
    unavailable if any required return is missing. Volatility is sample
    standard deviation of the last 20 observed daily returns times sqrt(252).
    Maximum drawdown is computed over all valid prices in the selected history.
    """
    if data is None or data.empty or price_column not in data:
        raise ValueError(f"Valid data with {price_column!r} is required.")
    if volatility_window < 2:
        raise ValueError("volatility_window must be at least 2.")
    prices = _price_series(data, price_column)
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be chronological.")
    valid = prices.dropna()
    if valid.empty:
        raise ValueError("No valid prices are available.")

    returns = prices.pct_change(fill_method=None)
    recent: dict[int, float | None] = {}
    for sessions in recent_sessions:
        if sessions < 1:
            raise ValueError("Recent-performance windows must be positive.")
        tail = returns.tail(sessions)
        recent[int(sessions)] = (
            float((1 + tail).prod() - 1)
            if len(tail) == sessions and tail.notna().all() and np.isfinite(tail).all()
            else None
        )

    vol_returns = returns.tail(volatility_window)
    annualized_volatility = (
        float(vol_returns.std(ddof=1) * np.sqrt(TRADING_SESSIONS_PER_YEAR))
        if len(vol_returns) == volatility_window and vol_returns.notna().all()
        else None
    )

    drawdowns = valid / valid.cummax() - 1
    max_drawdown = float(drawdowns.min()) if len(valid) >= 2 else None
    warnings: list[str] = []
    invalid_numeric_count = _invalid_numeric_price_count(data, price_column)
    if invalid_numeric_count:
        warnings.append(
            f"{invalid_numeric_count} supplied {price_column} value(s) are nonnumeric or nonfinite and were excluded."
        )
    if prices.iloc[-1:].isna().any():
        warnings.append("The last observation has no usable price; the displayed latest valid price is earlier.")
    if prices.tail(volatility_window).isna().any():
        warnings.append(f"Missing prices occur in the latest {volatility_window} observations; recent returns or volatility may be unavailable.")

    recent_move = returns.tail(volatility_window).dropna()
    large_moves = recent_move[recent_move.abs() >= DRAWDOWN_MOVE_WARNING]
    if not large_moves.empty:
        worst_date = large_moves.abs().idxmax()
        warnings.append(
            f"A one-session {price_column} move of at least {DRAWDOWN_MOVE_WARNING:.0%} occurred in the latest "
            f"{volatility_window} observed returns (largest: {large_moves.loc[worst_date]:+.1%} on {worst_date:%Y-%m-%d}); verify corporate actions and source data."
        )

    volume_ratio: float | None = None
    if "Volume" in data.columns:
        volume = pd.to_numeric(data["Volume"], errors="coerce").reindex(prices.index)
        latest_price_position = int(prices.index.get_loc(valid.index[-1]))
        latest_volume = volume.iloc[latest_price_position]
        previous_volumes = volume.iloc[
            max(0, latest_price_position - volatility_window):latest_price_position
        ]
        if (
            len(previous_volumes) == volatility_window
            and previous_volumes.notna().all()
            and pd.notna(latest_volume)
            and np.isfinite(previous_volumes).all()
            and (previous_volumes >= 0).all()
            and latest_volume >= 0
        ):
            median_volume = float(previous_volumes.median())
            if median_volume > 0:
                volume_ratio = float(latest_volume / median_volume)
                if volume_ratio >= VOLUME_SPIKE_MULTIPLE:
                    warnings.append(f"Latest reported volume is {volume_ratio:.1f}x the prior {volatility_window}-observation median.")
                elif volume_ratio <= VOLUME_LOW_MULTIPLE:
                    warnings.append(f"Latest reported volume is {volume_ratio:.2f}x the prior {volatility_window}-observation median; check for incomplete or unusual data.")

    return {
        "latest_valid_price": float(valid.iloc[-1]),
        "latest_valid_price_date": valid.index[-1],
        "recent_returns": recent,
        "annualized_volatility": annualized_volatility,
        "volatility_window": volatility_window,
        "volatility_observations": int(vol_returns.notna().sum()),
        "max_drawdown": max_drawdown,
        "drawdown_start": valid.index[0],
        "drawdown_end": valid.index[-1],
        "drawdown_observations": int(len(valid)),
        "latest_volume_vs_prior_median": volume_ratio,
        "volume_observation_date": valid.index[-1],
        "warnings": warnings,
    }


def assess_data_freshness(
    data: pd.DataFrame,
    price_column: str,
    *,
    retrieved_at=None,
    stale: bool = False,
    now=None,
    retrieval_warning_hours: int = 24,
    observation_warning_days: int = 7,
) -> dict[str, object]:
    """Describe provider retrieval age separately from the latest price date."""
    prices = _price_series(data, price_column)
    valid = prices.dropna()
    if valid.empty:
        raise ValueError("No valid price observation is available to assess freshness.")
    current = pd.Timestamp.now(tz=INDIA_TZ) if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize(INDIA_TZ)
    else:
        current = current.tz_convert(INDIA_TZ)
    retrieved = None
    retrieval_age_hours = None
    warnings: list[str] = []
    invalid_numeric_count = _invalid_numeric_price_count(data, price_column)
    if invalid_numeric_count:
        warnings.append(
            f"{invalid_numeric_count} supplied {price_column} value(s) are nonnumeric or nonfinite and were excluded."
        )
    if retrieved_at is not None:
        retrieved = pd.Timestamp(retrieved_at)
        if retrieved.tzinfo is None:
            retrieved = retrieved.tz_localize(INDIA_TZ)
        else:
            retrieved = retrieved.tz_convert(INDIA_TZ)
        retrieval_age_hours = float((current - retrieved).total_seconds() / 3600)
    else:
        warnings.append("Provider retrieval time is unavailable; data freshness cannot be verified from retrieval metadata.")

    observation_date = pd.Timestamp(valid.index[-1])
    if observation_date.tzinfo is None:
        observation_date = observation_date.tz_localize(INDIA_TZ)
    else:
        observation_date = observation_date.tz_convert(INDIA_TZ)
    observation_age_days = int((current.normalize() - observation_date.normalize()).days)
    if stale:
        status = "Stale cached snapshot"
        warnings.insert(0, "Provider checks failed or were cooling down; this is the last successful snapshot, not a confirmed current quote.")
    elif retrieval_age_hours is not None and retrieval_age_hours < 0:
        status = "Retrieval timestamp is in the future"
        warnings.append("Provider retrieval time is ahead of the current time; verify its timezone and clock source.")
    elif retrieval_age_hours is not None and retrieval_age_hours > retrieval_warning_hours:
        status = "Retrieval metadata is old"
        warnings.append(f"The last retrieval was {retrieval_age_hours:.1f} hours ago; verify before relying on current conditions.")
    elif retrieved_at is None:
        status = "Retrieval time unavailable"
    else:
        status = "Provider response received"
    if observation_age_days < 0:
        warnings.append("Latest price observation is dated in the future relative to the local clock; verify timestamps.")
    elif observation_age_days > observation_warning_days:
        warnings.append(f"Latest valid price is {observation_age_days} calendar days old; check the market calendar and provider coverage.")
    if prices.iloc[-1:].isna().any():
        warnings.append("The most recent row has no valid selected price; the displayed price is from an earlier observation.")
    return {
        "status": status,
        "latest_valid_price_date": observation_date,
        "observation_age_days": observation_age_days,
        "retrieved_at": retrieved,
        "retrieval_age_hours": retrieval_age_hours,
        "warnings": warnings,
    }


def read_benchmark_csv(source, price_column: str = "Adj Close") -> pd.Series:
    """Read a user-supplied daily benchmark CSV without filling or writing data."""
    try:
        frame = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not read benchmark CSV: {exc}") from exc
    date_column = next((column for column in frame.columns if str(column).strip().casefold() == "date"), None)
    if date_column is None:
        raise ValueError("Benchmark CSV must include a Date column.")
    if price_column not in frame.columns:
        raise ValueError(f"Benchmark CSV must include {price_column!r} to match the selected stock price basis.")
    dates = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("Benchmark CSV contains missing or invalid dates.")
    prices = pd.to_numeric(frame[price_column], errors="coerce")
    if prices.isna().any() or (~np.isfinite(prices)).any() or (prices <= 0).any():
        raise ValueError(f"Benchmark {price_column} contains missing, invalid, or nonpositive prices.")
    return _price_series(pd.Series(prices.to_numpy(), index=pd.DatetimeIndex(dates), name=price_column), price_column, daily_dates=True)


def compare_benchmark_returns(
    stock_prices: pd.Series,
    benchmark_prices: pd.Series,
) -> dict[str, object]:
    """Compare compounded returns only over matching dates; never forward-fill."""
    stock = _price_series(stock_prices, daily_dates=True).rename("stock")
    benchmark = _price_series(benchmark_prices, daily_dates=True).rename("benchmark")
    aligned = pd.concat([stock, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        raise ValueError("At least two matching price dates are required for a benchmark comparison.")
    stock_return = float(aligned.stock.iloc[-1] / aligned.stock.iloc[0] - 1)
    benchmark_return = float(aligned.benchmark.iloc[-1] / aligned.benchmark.iloc[0] - 1)
    relative_return = float((1 + stock_return) / (1 + benchmark_return) - 1)
    return {
        "matched_price_dates": int(len(aligned)),
        "matched_return_intervals": int(len(aligned) - 1),
        "start_date": aligned.index[0],
        "end_date": aligned.index[-1],
        "stock_absolute_return": stock_return,
        "benchmark_absolute_return": benchmark_return,
        "benchmark_relative_return": relative_return,
    }
