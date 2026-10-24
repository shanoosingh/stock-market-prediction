"""Small offline long/flat backtester for adjusted daily prices.

Signals are computed after a session close. A new position is assumed filled
at the next observed session's close; it earns returns only from the following
close-to-close interval. This conservative one-session execution delay avoids
using a signal to earn a return that happened before the assumed fill. Since
adjusted OHLC data is not generally available, adjusted close is used as a
proxy execution price; this is an approximation, not a fill simulation.

Adjusted close is appropriate for total-return accounting, but vendors can
back-adjust past observations for corporate actions that occurred later.
Moving-average signals computed from a retrospectively adjusted series can
therefore differ from signals that were knowable in real time around an action.

The engine never fetches or persists data. Missing price rows are rejected
rather than forward-filled or skipped silently.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR = 252
COST_RATE_VERIFIED_DATE = "2026-10-25"
DEFAULT_MAX_DAILY_OBSERVATION_AGE_DAYS = 7
DEFAULT_MAX_DAILY_RETRIEVAL_AGE_HOURS = 24


@dataclass
class BacktestInputMetadata:
    """Provenance and freshness context supplied with backtest prices.

    Naive timestamps are interpreted as Asia/Kolkata. Set ``synthetic=True``
    for generated fixtures; this identifies their provenance but does not
    make unknown freshness appear validated.
    """

    source: str | None
    latest_observation_at: object | None
    retrieved_at: object | None
    evaluated_at: object | None
    interval: str = "1d"
    freshness_status: str | None = None
    warnings: tuple[str, ...] = ()
    synthetic: bool = False


class BacktestInputPolicyError(ValueError):
    """Backtest request blocked by input freshness policy, with inspectable assessment."""

    def __init__(self, message: str, assessment: dict[str, object]):
        super().__init__(message)
        self.assessment = assessment


def _as_ist_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("timestamp is missing or invalid")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("Asia/Kolkata")
    return timestamp.tz_convert("Asia/Kolkata")


def assess_backtest_input_freshness(
    metadata: BacktestInputMetadata | None,
    *,
    max_daily_observation_age_days: float = DEFAULT_MAX_DAILY_OBSERVATION_AGE_DAYS,
    max_daily_retrieval_age_hours: float = DEFAULT_MAX_DAILY_RETRIEVAL_AGE_HOURS,
    allow_stale_override: bool = False,
    require_known: bool = False,
) -> dict[str, object]:
    """Assess provenance/freshness without fetching or persisting data.

    Daily limits use Asia/Kolkata calendar dates for observation age and
    elapsed hours for retrieval age. Intraday inputs stay unknown because this
    project has no bar-completion policy or intraday freshness limits.
    """
    try:
        limits = tuple(float(value) for value in (
            max_daily_observation_age_days, max_daily_retrieval_age_hours
        ))
    except (TypeError, ValueError) as exc:
        raise ValueError("Freshness limits must be finite, nonnegative values.") from exc
    if any(not np.isfinite(value) or value < 0 for value in limits):
        raise ValueError("Freshness limits must be finite, nonnegative values.")
    max_daily_observation_age_days, max_daily_retrieval_age_hours = limits
    if metadata is not None and not isinstance(metadata, BacktestInputMetadata):
        raise ValueError("metadata must be BacktestInputMetadata or None.")
    warnings = list(metadata.warnings) if isinstance(metadata, BacktestInputMetadata) else []
    if metadata is None:
        warnings.append("Backtest input provenance and freshness metadata are unavailable.")
        return {
            "status": "unknown", "allowed": not require_known,
            "source": None, "interval": None,
            "latest_observation_at": None, "retrieved_at": None, "evaluated_at": None,
            "observation_age_days": None, "retrieval_age_hours": None,
            "warnings": warnings, "stale_override": False, "synthetic": False,
        }

    source = str(metadata.source).strip() if metadata.source is not None else ""
    if source.casefold() in {"unknown", "unspecified", "none", "n/a"}:
        source = ""
    if not source:
        warnings.append("Data source/provider is unknown.")
    explicit_status = (metadata.freshness_status or "").strip().casefold()
    explicit_stale = any(
        token in explicit_status for token in ("stale", "old", "future", "failed", "cooldown")
    )
    if metadata.interval != "1d":
        warnings.append(
            f"Daily freshness limits were not applied to interval {metadata.interval!r}; intraday bar completeness is unverified."
        )
        intraday_timestamps = {}
        intraday_invalid = not bool(source)
        for field in ("latest_observation_at", "retrieved_at", "evaluated_at"):
            try:
                if getattr(metadata, field) is None:
                    raise ValueError("missing")
                intraday_timestamps[field] = _as_ist_timestamp(getattr(metadata, field))
            except (TypeError, ValueError, OverflowError):
                intraday_invalid = True
                intraday_timestamps[field] = None
                warnings.append(f"{field.replace('_', ' ').capitalize()} is unavailable or invalid.")
        if not intraday_invalid:
            evaluated = intraday_timestamps["evaluated_at"]
            if intraday_timestamps["latest_observation_at"] > evaluated:
                explicit_stale = True
                warnings.append("Latest observation timestamp is in the future relative to evaluation time.")
            if intraday_timestamps["retrieved_at"] > evaluated:
                explicit_stale = True
                warnings.append("Retrieval timestamp is in the future relative to evaluation time.")
        if explicit_stale:
            warnings.append(f"Provider freshness status is {metadata.freshness_status!r}.")
        status = "unknown"
        allowed = not require_known
        return {
            "status": status, "allowed": allowed,
            "source": source or None, "interval": metadata.interval,
            "latest_observation_at": intraday_timestamps.get("latest_observation_at"),
            "retrieved_at": intraday_timestamps.get("retrieved_at"),
            "evaluated_at": intraday_timestamps.get("evaluated_at"),
            "observation_age_days": None, "retrieval_age_hours": None,
            "warnings": warnings, "stale_override": status == "stale_override",
            "synthetic": bool(metadata.synthetic),
        }

    timestamps = {}
    invalid_timestamp = False
    for field in ("latest_observation_at", "retrieved_at", "evaluated_at"):
        value = getattr(metadata, field)
        if value is None:
            warnings.append(f"{field.replace('_', ' ').capitalize()} is unavailable.")
            invalid_timestamp = True
            timestamps[field] = None
            continue
        try:
            timestamps[field] = _as_ist_timestamp(value)
        except (TypeError, ValueError, OverflowError):
            warnings.append(f"{field.replace('_', ' ').capitalize()} is invalid.")
            invalid_timestamp = True
            timestamps[field] = None

    observation_age_days = None
    retrieval_age_hours = None
    stale_reasons = []
    future_timestamp = False
    if not invalid_timestamp:
        observation = timestamps["latest_observation_at"]
        retrieved = timestamps["retrieved_at"]
        evaluated = timestamps["evaluated_at"]
        observation_age_days = int((evaluated.normalize() - observation.normalize()).days)
        retrieval_age_hours = float((evaluated - retrieved).total_seconds() / 3600)
        if observation > evaluated:
            future_timestamp = True
            stale_reasons.append("Latest observation timestamp is in the future relative to evaluation time.")
        elif observation_age_days > max_daily_observation_age_days:
            stale_reasons.append(
                f"Latest observation is {observation_age_days} calendar days old; limit is {max_daily_observation_age_days:g}."
            )
        if retrieved > evaluated:
            future_timestamp = True
            stale_reasons.append("Retrieval timestamp is in the future relative to evaluation time.")
        elif retrieval_age_hours > max_daily_retrieval_age_hours:
            stale_reasons.append(
                f"Retrieval is {retrieval_age_hours:.1f} hours old; limit is {max_daily_retrieval_age_hours:g}."
            )
    if explicit_stale:
        stale_reasons.append(f"Provider freshness status is {metadata.freshness_status!r}.")
    warnings.extend(stale_reasons)

    if stale_reasons:
        override_complete = not invalid_timestamp and not future_timestamp and bool(source)
        status = (
            "stale_override"
            if allow_stale_override and override_complete
            else "stale_rejected"
        )
        allowed = status == "stale_override"
        if allow_stale_override and not override_complete:
            warnings.append(
                "Stale override is unavailable because provenance/timestamps are incomplete or a timestamp is in the future."
            )
        if status == "stale_override":
            warnings.append("Stale input was used under an explicit diagnostic override.")
    elif invalid_timestamp or not source or explicit_status in {"unknown", "unavailable"}:
        status = "unknown"
        allowed = not require_known
        if explicit_status in {"unknown", "unavailable"}:
            warnings.append(f"Provider freshness status is {metadata.freshness_status!r}.")
    else:
        status = "fresh_validated"
        allowed = True
    return {
        "status": status, "allowed": allowed, "source": source or None,
        "interval": metadata.interval,
        "latest_observation_at": timestamps["latest_observation_at"],
        "retrieved_at": timestamps["retrieved_at"],
        "evaluated_at": timestamps["evaluated_at"],
        "observation_age_days": observation_age_days,
        "retrieval_age_hours": retrieval_age_hours,
        "warnings": warnings, "stale_override": status == "stale_override",
        "synthetic": bool(metadata.synthetic),
    }


@dataclass(frozen=True)
class IndianDeliveryCostScenario:
    """Per-side bps assumptions for an NSE cash-equity delivery order.

    Statutory/exchange rates reflect official material checked on
    ``COST_RATE_VERIFIED_DATE``. Brokerage and slippage are user-editable
    scenario assumptions, not official rates or execution estimates.
    GST is modeled on brokerage + exchange + SEBI service charges; STT and
    stamp duty are excluded from the GST base (pure-agent treatment assumed).
    """

    name: str
    brokerage_bps_buy: float
    brokerage_bps_sell: float
    slippage_bps_buy: float
    slippage_bps_sell: float
    stt_bps_buy: float = 10.0
    stt_bps_sell: float = 10.0
    exchange_bps_buy: float = 0.307
    exchange_bps_sell: float = 0.307
    sebi_bps_buy: float = 0.01
    sebi_bps_sell: float = 0.01
    stamp_duty_bps_buy: float = 1.5
    stamp_duty_bps_sell: float = 0.0
    gst_rate: float = 0.18

    def __post_init__(self):
        numeric_fields = (
            "brokerage_bps_buy", "brokerage_bps_sell",
            "slippage_bps_buy", "slippage_bps_sell",
            "stt_bps_buy", "stt_bps_sell",
            "exchange_bps_buy", "exchange_bps_sell",
            "sebi_bps_buy", "sebi_bps_sell",
            "stamp_duty_bps_buy", "stamp_duty_bps_sell",
            "gst_rate",
        )
        for field in numeric_fields:
            value = getattr(self, field)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be a finite, nonnegative number.")
        if self.gst_rate > 1:
            raise ValueError("gst_rate must be a fraction between 0 and 1.")
        if not self.name.strip():
            raise ValueError("Scenario name must not be empty.")

    def side_cost_bps(self, side: Literal["buy", "sell"]) -> float:
        if side == "buy":
            brokerage = self.brokerage_bps_buy
            slippage = self.slippage_bps_buy
            exchange = self.exchange_bps_buy
            sebi = self.sebi_bps_buy
            stt = self.stt_bps_buy
            stamp = self.stamp_duty_bps_buy
        elif side == "sell":
            brokerage = self.brokerage_bps_sell
            slippage = self.slippage_bps_sell
            exchange = self.exchange_bps_sell
            sebi = self.sebi_bps_sell
            stt = self.stt_bps_sell
            stamp = self.stamp_duty_bps_sell
        else:
            raise ValueError("side must be 'buy' or 'sell'.")
        gst = self.gst_rate * (brokerage + exchange + sebi)
        return float(brokerage + slippage + exchange + sebi + stt + stamp + gst)


INDIAN_DELIVERY_COST_SCENARIOS = MappingProxyType(
    {
        "low": IndianDeliveryCostScenario(
            name="low",
            brokerage_bps_buy=0.0,
            brokerage_bps_sell=0.0,
            slippage_bps_buy=1.0,
            slippage_bps_sell=1.0,
        ),
        "medium": IndianDeliveryCostScenario(
            name="medium",
            brokerage_bps_buy=2.0,
            brokerage_bps_sell=2.0,
            slippage_bps_buy=5.0,
            slippage_bps_sell=5.0,
        ),
        "high": IndianDeliveryCostScenario(
            name="high",
            brokerage_bps_buy=5.0,
            brokerage_bps_sell=5.0,
            slippage_bps_buy=15.0,
            slippage_bps_sell=15.0,
        ),
    }
)


def get_indian_delivery_cost_scenario(name: str) -> IndianDeliveryCostScenario:
    """Return a named low/medium/high illustrative NSE delivery scenario."""
    normalized = str(name).strip().lower()
    try:
        return INDIAN_DELIVERY_COST_SCENARIOS[normalized]
    except KeyError as exc:
        raise ValueError("Cost scenario must be one of: low, medium, high.") from exc


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    price_column: str
    daily: pd.DataFrame
    metrics: dict[str, float | int | str | bool | None]


def _prices(data: pd.Series | pd.DataFrame, price_column: str) -> pd.Series:
    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, pd.DataFrame) and price_column in data.columns:
        series = data[price_column].copy()
    else:
        raise ValueError(f"Input must contain the selected {price_column!r} price basis.")
    dates = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce"))
    if dates.isna().any():
        raise ValueError("Price dates contain invalid timestamps.")
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Kolkata").tz_localize(None)
    dates = dates.normalize()
    if dates.has_duplicates:
        raise ValueError("Daily price dates contain duplicates after date normalization.")
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.isna().any():
        raise ValueError("Missing or nonnumeric prices are not filled; clean or segment the data explicitly.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive.")
    result = pd.Series(values.to_numpy(), index=dates, name=price_column).sort_index()
    if len(result) < 2:
        raise ValueError("At least two valid adjusted-price observations are required.")
    return result


def summarize_returns(
    daily: pd.DataFrame,
    start=None,
    end=None,
) -> dict[str, float | int | None]:
    """Summarize daily net returns and executed-position changes."""
    selected = daily if start is None and end is None else daily.loc[start:end]
    if selected.empty:
        raise ValueError("The requested evaluation period contains no observations.")
    returns = selected["Net Return"].astype(float).copy()
    gross_returns = selected["Gross Return"].astype(float).copy()
    if start is not None:
        # A segment begins at the session close: price movement up to that
        # close belongs to the preceding period. Boundary execution cost stays.
        first_cost = float(selected["Trading Cost Fraction"].iloc[0])
        returns.iloc[0] = -first_cost
        gross_returns.iloc[0] = 0.0
    equity = (1 + returns).cumprod()
    cumulative = float(equity.iloc[-1] - 1)
    elapsed_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    cagr = (
        float(equity.iloc[-1] ** (365.25 / elapsed_days) - 1)
        if elapsed_days > 0 and equity.iloc[-1] > 0
        else None
    )
    full_equity = pd.concat(
        [pd.Series([1.0], index=[equity.index[0] - pd.Timedelta(nanoseconds=1)]), equity]
    )
    drawdown = full_equity / full_equity.cummax() - 1
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(TRADING_SESSIONS_PER_YEAR))
        if len(returns) >= 2
        else None
    )
    turnover = selected["Turnover"].astype(float)
    return {
        "observations": int(len(selected)),
        "cumulative_return": cumulative,
        "net_cumulative_return": cumulative,
        "gross_cumulative_return": float((1 + gross_returns).prod() - 1),
        "cagr": cagr,
        "maximum_drawdown": float(drawdown.min()),
        "annualized_volatility": volatility,
        "turnover": float(turnover.sum()),
        "trade_count": int((turnover > 0).sum()),
        "total_transaction_cost_fraction": float(
            selected["Trading Cost Fraction"].astype(float).sum()
        ),
    }


def run_backtest(
    data: pd.Series | pd.DataFrame,
    strategy: str,
    *,
    benchmark: pd.Series | pd.DataFrame | None = None,
    price_column: str = "Adj Close",
    fast_window: int = 20,
    slow_window: int = 50,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    cost_scenario: str | IndianDeliveryCostScenario | None = None,
    benchmark_source: str | None = None,
    benchmark_price_basis: str | None = None,
    input_metadata: BacktestInputMetadata | None = None,
    allow_stale_override: bool = False,
    max_daily_observation_age_days: float = DEFAULT_MAX_DAILY_OBSERVATION_AGE_DAYS,
    max_daily_retrieval_age_hours: float = DEFAULT_MAX_DAILY_RETRIEVAL_AGE_HOURS,
) -> BacktestResult:
    """Run buy-and-hold, no-trade, or a long/flat SMA crossover strategy.

    Without ``cost_scenario``, legacy symmetric ``transaction_cost_bps`` and
    ``slippage_bps`` remain supported. With a named/custom scenario, buy and
    sell costs are computed separately. Turnover is the sum of absolute
    position changes (0 to 1 for each entry/exit); trade_count counts orders.
    """
    if strategy not in {"buy_and_hold", "no_trade", "sma_crossover"}:
        raise ValueError("strategy must be buy_and_hold, no_trade, or sma_crossover.")
    input_validation = assess_backtest_input_freshness(
        input_metadata,
        max_daily_observation_age_days=max_daily_observation_age_days,
        max_daily_retrieval_age_hours=max_daily_retrieval_age_hours,
        allow_stale_override=allow_stale_override,
    )
    if input_validation["status"] == "stale_rejected":
        raise BacktestInputPolicyError(
            "Backtest input rejected by freshness policy: "
            + " ".join(input_validation["warnings"]),
            input_validation,
        )
    if strategy == "sma_crossover" and (fast_window < 1 or slow_window < 2 or fast_window >= slow_window):
        raise ValueError("SMA windows must satisfy 1 <= fast_window < slow_window.")
    try:
        transaction_cost_bps = float(transaction_cost_bps)
        slippage_bps = float(slippage_bps)
    except (TypeError, ValueError) as exc:
        raise ValueError("Transaction cost and slippage must be finite numeric basis-point values.") from exc
    if not np.isfinite(transaction_cost_bps) or not np.isfinite(slippage_bps):
        raise ValueError("Transaction cost and slippage must be finite numeric basis-point values.")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("Transaction cost and slippage assumptions cannot be negative.")
    if cost_scenario is not None and (transaction_cost_bps != 0 or slippage_bps != 0):
        raise ValueError("Use either cost_scenario or legacy transaction_cost_bps/slippage_bps, not both.")
    if isinstance(cost_scenario, str):
        scenario = get_indian_delivery_cost_scenario(cost_scenario)
    elif cost_scenario is None:
        scenario = None
    elif isinstance(cost_scenario, IndianDeliveryCostScenario):
        scenario = cost_scenario
    else:
        raise ValueError("cost_scenario must be a preset name or IndianDeliveryCostScenario.")

    legacy_cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    if legacy_cost_rate >= 1:
        raise ValueError("Combined per-trade costs must be below 100%.")

    prices = _prices(data, price_column)
    if strategy == "sma_crossover" and len(prices) < slow_window:
        raise ValueError(
            f"SMA crossover requires at least {slow_window} valid observations for its slow window; "
            f"received {len(prices)}."
        )
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    if not np.isfinite(asset_returns).all():
        raise ValueError("Adjusted-price returns overflowed or are nonfinite.")

    if strategy == "buy_and_hold":
        executed_position = pd.Series(1.0, index=prices.index)
        signal_at_close = executed_position.copy()
    elif strategy == "no_trade":
        executed_position = pd.Series(0.0, index=prices.index)
        signal_at_close = executed_position.copy()
    else:
        fast = prices.rolling(fast_window, min_periods=fast_window).mean()
        slow = prices.rolling(slow_window, min_periods=slow_window).mean()
        signal_at_close = (fast > slow).astype(float)
        # Signal at close t can execute at close t+1.
        executed_position = signal_at_close.shift(1).fillna(0.0)

    signed_position_changes = executed_position.diff()
    if strategy == "buy_and_hold":
        signed_position_changes.iloc[0] = executed_position.iloc[0]
    else:
        signed_position_changes.iloc[0] = 0.0
    position_changes = signed_position_changes.abs()

    # A position filled at close t earns the next close-to-close interval,
    # whose return is indexed t+1.
    held_position = executed_position.shift(1).fillna(0.0)
    gross_returns = held_position * asset_returns
    if scenario is None:
        cost_rate_bps = pd.Series(
            transaction_cost_bps + slippage_bps,
            index=prices.index,
            dtype=float,
        )
        cost_mode = (
            "ZERO COST / FRICTIONLESS"
            if transaction_cost_bps == 0 and slippage_bps == 0
            else "legacy symmetric bps"
        )
        scenario_name = None
        buy_cost_bps = transaction_cost_bps + slippage_bps
        sell_cost_bps = transaction_cost_bps + slippage_bps
    else:
        buy_cost_bps = scenario.side_cost_bps("buy")
        sell_cost_bps = scenario.side_cost_bps("sell")
        cost_rate_bps = pd.Series(
            np.where(signed_position_changes > 0, buy_cost_bps,
                     np.where(signed_position_changes < 0, sell_cost_bps, 0.0)),
            index=prices.index,
            dtype=float,
        )
        cost_mode = f"Indian delivery scenario: {scenario.name}"
        scenario_name = scenario.name
    if buy_cost_bps >= 10_000 or sell_cost_bps >= 10_000:
        raise ValueError("Combined per-trade costs must be below 100% on each side.")
    cost_fractions = position_changes * cost_rate_bps / 10_000
    if (cost_fractions >= 1).any():
        raise ValueError("Combined per-trade costs must be below 100%.")
    trading_costs = cost_fractions
    net_returns = (1 + gross_returns) * (1 - trading_costs) - 1
    daily = pd.DataFrame(
        {
            "Adjusted Close": prices,
            "Asset Return": asset_returns,
            "Signal Position": signal_at_close,
            "Executed Position": executed_position,
            "Held Position": held_position,
            "Turnover": position_changes,
            "Cost Rate Bps": cost_rate_bps,
            "Trading Cost Fraction": trading_costs,
            "Gross Return": gross_returns,
            "Net Return": net_returns,
        },
        index=prices.index,
    )
    metrics = summarize_returns(daily)
    metrics.update(
        {
            "cost_mode": cost_mode,
            "cost_scenario": scenario_name,
            "frictionless": bool(buy_cost_bps == 0 and sell_cost_bps == 0),
            "buy_cost_bps_per_full_turnover": float(buy_cost_bps),
            "sell_cost_bps_per_full_turnover": float(sell_cost_bps),
            "cost_rates_verified_as_of": COST_RATE_VERIFIED_DATE if scenario is not None else None,
            "input_validation": input_validation,
        }
    )
    if benchmark is not None:
        if benchmark_price_basis is not None and benchmark_price_basis != price_column:
            raise ValueError(
                "Benchmark price basis must match the strategy price basis; "
                "convert inputs explicitly before comparison."
            )
        if isinstance(benchmark, pd.DataFrame):
            if price_column not in benchmark.columns:
                raise ValueError(f"Input must contain the selected {price_column!r} price basis.")
            benchmark_basis_status = "selected_column"
        else:
            benchmark_series_name = str(getattr(benchmark, "name", "") or "").strip()
            if benchmark_series_name.lower() in {"adj close", "close"} and benchmark_series_name != price_column:
                raise ValueError(
                    "Benchmark series price basis conflicts with the selected strategy price basis."
                )
            benchmark_basis_status = (
                "caller_asserted" if benchmark_price_basis is not None else "assumed_from_price_column"
            )
        benchmark_prices = _prices(benchmark, price_column)
        if not benchmark_prices.index.equals(prices.index):
            missing_from_benchmark = len(prices.index.difference(benchmark_prices.index))
            outside_stock_period = len(benchmark_prices.index.difference(prices.index))
            raise ValueError(
                "Benchmark dates must exactly match strategy dates; "
                f"{missing_from_benchmark} strategy dates are absent from the benchmark and "
                f"{outside_stock_period} benchmark dates are outside the strategy period. "
                "No dates are filled or interpolated."
            )
        aligned_prices = pd.concat(
            [prices.rename("Stock Price"), benchmark_prices.rename("Benchmark Price")],
            axis=1,
            join="inner",
        )
        if len(aligned_prices) < 2:
            raise ValueError("Benchmark has fewer than two matching price dates with the stock.")
        strategy_intervals = []
        benchmark_intervals = []
        interval_dates = []
        interval_start_dates = []
        for interval_i, (previous_date, current_date) in enumerate(zip(aligned_prices.index[:-1], aligned_prices.index[1:])):
            daily_slice = daily.loc[(daily.index > previous_date) & (daily.index <= current_date), "Net Return"]
            if daily_slice.empty:
                continue
            interval_growth = float((1 + daily_slice).prod())
            if interval_i == 0:
                # Include any boundary-date execution cost without attributing
                # the preceding close-to-close price move to this interval.
                interval_growth *= 1 - float(daily.loc[previous_date, "Trading Cost Fraction"])
            strategy_intervals.append(interval_growth - 1)
            benchmark_intervals.append(
                float(aligned_prices.loc[current_date, "Benchmark Price"]
                      / aligned_prices.loc[previous_date, "Benchmark Price"] - 1)
            )
            interval_start_dates.append(previous_date)
            interval_dates.append(current_date)
        if not strategy_intervals:
            raise ValueError("Benchmark has no matching return intervals with the strategy.")
        strategy_matched = float(np.prod(1 + np.asarray(strategy_intervals)) - 1)
        benchmark_matched = float(np.prod(1 + np.asarray(benchmark_intervals)) - 1)
        benchmark_fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(aligned_prices["Benchmark Price"], index=True)
            .to_numpy()
            .tobytes()
        ).hexdigest()
        benchmark_label = benchmark_source
        if benchmark_label is None:
            candidate_name = getattr(benchmark, "name", None)
            if candidate_name and str(candidate_name).lower() not in {"adj close", "close", "date"}:
                benchmark_label = str(candidate_name)
        metrics.update(
            {
                "benchmark_matched_observations": int(len(interval_dates)),
                "benchmark_period_start": interval_start_dates[0],
                "benchmark_period_end": interval_dates[-1],
                "benchmark_source": benchmark_label or "Unknown (no source label supplied)",
                "benchmark_source_provenance": "caller_asserted" if benchmark_label else "unknown",
                "benchmark_source_independently_verified": False,
                "benchmark_price_basis": price_column,
                "benchmark_price_basis_status": benchmark_basis_status,
                "benchmark_data_fingerprint": benchmark_fingerprint,
                "benchmark_input_observations": int(len(benchmark_prices)),
                "benchmark_matched_price_dates": int(len(aligned_prices)),
                "benchmark_unmatched_stock_dates": int(len(prices.index.difference(benchmark_prices.index))),
                "benchmark_unmatched_input_dates": int(len(benchmark_prices.index.difference(prices.index))),
                "strategy_return_on_benchmark_dates": strategy_matched,
                "benchmark_cumulative_return": benchmark_matched,
                "benchmark_relative_return": float(
                    (1 + strategy_matched) / (1 + benchmark_matched) - 1
                ),
            }
        )
    else:
        metrics.update(
            {
                "benchmark_matched_observations": 0,
                "benchmark_period_start": None,
                "benchmark_period_end": None,
                "benchmark_source": None,
                "benchmark_source_provenance": "unknown",
                "benchmark_source_independently_verified": False,
                "benchmark_price_basis": None,
                "benchmark_price_basis_status": "unknown",
                "benchmark_data_fingerprint": None,
                "benchmark_input_observations": 0,
                "benchmark_matched_price_dates": 0,
                "benchmark_unmatched_stock_dates": 0,
                "benchmark_unmatched_input_dates": 0,
                "strategy_return_on_benchmark_dates": None,
                "benchmark_cumulative_return": None,
                "benchmark_relative_return": None,
            }
        )

    return BacktestResult(strategy, price_column, daily, metrics)


def run_validated_backtest(
    data: pd.Series | pd.DataFrame,
    strategy: str,
    *,
    input_metadata: BacktestInputMetadata,
    allow_stale_override: bool = False,
    max_daily_observation_age_days: float = DEFAULT_MAX_DAILY_OBSERVATION_AGE_DAYS,
    max_daily_retrieval_age_hours: float = DEFAULT_MAX_DAILY_RETRIEVAL_AGE_HOURS,
    **backtest_options,
) -> BacktestResult:
    """Production-style request: require known provenance and fresh daily data.

    Stale data is rejected unless ``allow_stale_override=True`` is explicitly
    supplied for diagnostic/historical review. Unknown provenance and all
    intraday data remain rejected because no intraday freshness policy exists.
    """
    validation = assess_backtest_input_freshness(
        input_metadata,
        max_daily_observation_age_days=max_daily_observation_age_days,
        max_daily_retrieval_age_hours=max_daily_retrieval_age_hours,
        allow_stale_override=allow_stale_override,
        require_known=True,
    )
    if not validation["allowed"]:
        raise BacktestInputPolicyError(
            f"Backtest input policy status {validation['status']!r}: "
            + " ".join(validation["warnings"]),
            validation,
        )
    price_column = backtest_options.get("price_column", "Adj Close")
    actual_latest_date = _prices(data, price_column).index[-1]
    metadata_latest_date = _as_ist_timestamp(
        input_metadata.latest_observation_at
    ).tz_localize(None).normalize()
    if metadata_latest_date != actual_latest_date:
        raise ValueError(
            "Backtest input metadata latest observation date does not match the supplied price data."
        )
    return run_backtest(
        data,
        strategy,
        input_metadata=input_metadata,
        allow_stale_override=allow_stale_override,
        max_daily_observation_age_days=max_daily_observation_age_days,
        max_daily_retrieval_age_hours=max_daily_retrieval_age_hours,
        **backtest_options,
    )


def build_backtest_report(
    result: BacktestResult,
    *,
    periods: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a presentation-ready, factual summary without making a forecast.

    ``periods`` may be supplied from ``run_chronological_evaluation``. The
    helper deliberately calls its last segment a final evaluation period;
    whether it is an untouched holdout depends on research history outside
    the backtest engine.
    """
    metrics = result.metrics
    input_validation = metrics.get("input_validation") or assess_backtest_input_freshness(None)
    date_fingerprint = hashlib.sha256(
        "\n".join(pd.Timestamp(value).isoformat() for value in result.daily.index).encode("utf-8")
    ).hexdigest()
    cost_mode = str(metrics["cost_mode"])
    cost_description = cost_mode
    if metrics.get("cost_scenario"):
        cost_description += (
            f"; buy {metrics['buy_cost_bps_per_full_turnover']:.5f} bps, "
            f"sell {metrics['sell_cost_bps_per_full_turnover']:.5f} bps per full turnover"
        )
    elif metrics.get("frictionless"):
        cost_description = "ZERO COST / FRICTIONLESS"
    else:
        cost_description += (
            f"; symmetric buy/sell {metrics['buy_cost_bps_per_full_turnover']:.5f} bps "
            "per full turnover"
        )
    benchmark_report = None
    if metrics.get("benchmark_matched_observations", 0):
        benchmark_report = {
            "source": metrics["benchmark_source"],
            "source_provenance": metrics["benchmark_source_provenance"],
            "source_independently_verified": metrics["benchmark_source_independently_verified"],
            "price_basis": metrics["benchmark_price_basis"],
            "price_basis_status": metrics["benchmark_price_basis_status"],
            "period_start": metrics["benchmark_period_start"],
            "period_end": metrics["benchmark_period_end"],
            "matched_return_intervals": metrics["benchmark_matched_observations"],
            "strategy_net_return": metrics["strategy_return_on_benchmark_dates"],
            "benchmark_return": metrics["benchmark_cumulative_return"],
            "data_fingerprint": metrics["benchmark_data_fingerprint"],
            "input_observations": metrics["benchmark_input_observations"],
            "matched_price_dates": metrics["benchmark_matched_price_dates"],
            "unmatched_stock_dates": metrics["benchmark_unmatched_stock_dates"],
            "unmatched_input_dates": metrics["benchmark_unmatched_input_dates"],
            "relative_return": metrics["benchmark_relative_return"],
            "date_policy": "Exact common price dates; no filling",
        }
    period_report = None
    if periods is not None:
        period_report = {
            name: {
                **values,
                "interpretation": (
                    "Retrospective final evaluation; untouched only if no one previously examined these dates or outcomes."
                    if name == "final_evaluation"
                    else f"Chronological {name.replace('_', ' ')} segment."
                ),
            }
            for name, values in periods.items()
        }
    return {
        "strategy": result.strategy,
        "price_basis": result.price_column,
        "evaluation_start": result.daily.index.min(),
        "evaluation_end": result.daily.index.max(),
        "evaluation_observations": int(len(result.daily)),
        "evaluation_dates_fingerprint": date_fingerprint,
        "cost_mode": cost_mode,
        "cost_description": cost_description,
        "frictionless": bool(metrics.get("frictionless", False)),
        "input_validation": input_validation,
        "input_freshness_status": input_validation["status"],
        "buy_cost_bps_per_full_turnover": metrics["buy_cost_bps_per_full_turnover"],
        "sell_cost_bps_per_full_turnover": metrics["sell_cost_bps_per_full_turnover"],
        "gross_cumulative_return": metrics["gross_cumulative_return"],
        "net_cumulative_return": metrics["net_cumulative_return"],
        "cagr": metrics["cagr"],
        "maximum_drawdown": metrics["maximum_drawdown"],
        "annualized_volatility": metrics["annualized_volatility"],
        "turnover": metrics["turnover"],
        "trade_count": metrics["trade_count"],
        "benchmark_relative_return": metrics["benchmark_relative_return"],
        "total_transaction_cost_fraction": metrics["total_transaction_cost_fraction"],
        "benchmark": benchmark_report,
        "periods": period_report,
        "limitations": [
            "Historical simulation is not a forecast or guarantee of future returns.",
            "Transaction costs are configured assumptions and may differ from actual charges and fills.",
            "Adjusted close is used as a proxy execution price, not an executable historical quote.",
            "The engine has no complete exchange calendar or broker-specific execution model.",
        ],
    }


def compare_backtest_reports(
    reports: dict[str, dict[str, object]],
    *,
    period_start,
    period_end,
) -> dict[str, object]:
    """Compare report summaries on one explicitly specified common period.

    Input order is preserved and no strategy is ranked. Reports with different
    cost assumptions are returned with an explicit comparability warning.
    Benchmark presence, source, dates, and return must match exactly, otherwise
    a ValueError prevents an unlike comparison.
    """
    if not isinstance(reports, dict) or len(reports) < 2:
        raise ValueError("Provide reports for at least two named strategies.")
    if period_start is None or period_end is None:
        raise ValueError("An ordered, explicit evaluation period is required.")

    def local_date(value) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("Asia/Kolkata").tz_localize(None)
        return stamp.normalize()

    requested_start = local_date(period_start)
    requested_end = local_date(period_end)
    if requested_start > requested_end:
        raise ValueError("An ordered, explicit evaluation period is required.")
    metric_names = (
        "net_cumulative_return",
        "cagr",
        "maximum_drawdown",
        "annualized_volatility",
        "turnover",
        "trade_count",
        "benchmark_relative_return",
    )
    benchmark_signature = None
    evaluation_signature = None
    input_signature = None
    cost_signatures = []
    rows = []
    for name, report in reports.items():
        if not isinstance(report, dict):
            raise ValueError(f"Report for {name!r} must be a mapping.")
        start = local_date(report.get("evaluation_start"))
        end = local_date(report.get("evaluation_end"))
        if start != requested_start or end != requested_end:
            raise ValueError(
                f"Report for {name!r} covers {start.date()} to {end.date()}, "
                f"not the requested common period {requested_start.date()} to {requested_end.date()}."
            )
        current_evaluation = (
            report.get("price_basis"),
            report.get("evaluation_observations"),
            report.get("evaluation_dates_fingerprint"),
        )
        if any(value is None for value in current_evaluation):
            raise ValueError(f"Report for {name!r} lacks price-basis or evaluation-date identity metadata.")
        if evaluation_signature is None:
            evaluation_signature = current_evaluation
        elif current_evaluation != evaluation_signature:
            raise ValueError(
                "Strategies must use the same price basis and exact evaluation observation dates."
            )
        input_validation = report.get("input_validation", {})
        current_input = (
            input_validation.get("source"),
            input_validation.get("interval"),
            input_validation.get("latest_observation_at"),
            input_validation.get("retrieved_at"),
            input_validation.get("evaluated_at"),
            input_validation.get("status", "unknown"),
            input_validation.get("synthetic", False),
        )
        if input_signature is None:
            input_signature = current_input
        elif current_input != input_signature:
            raise ValueError("Strategies must use the same input provenance and freshness metadata.")

        benchmark = report.get("benchmark")
        current_benchmark = None if benchmark is None else (
            benchmark.get("source"),
            benchmark.get("source_provenance", "unknown"),
            benchmark.get("source_independently_verified", False),
            benchmark.get("price_basis"),
            benchmark.get("price_basis_status", "unknown"),
            pd.Timestamp(benchmark.get("period_start")),
            pd.Timestamp(benchmark.get("period_end")),
            benchmark.get("benchmark_return"),
            benchmark.get("matched_return_intervals"),
            benchmark.get("data_fingerprint"),
            benchmark.get("input_observations"),
            benchmark.get("matched_price_dates"),
            benchmark.get("unmatched_stock_dates"),
            benchmark.get("unmatched_input_dates"),
        )
        if benchmark_signature is None and not rows:
            benchmark_signature = current_benchmark
        elif current_benchmark != benchmark_signature:
            raise ValueError(
                "Strategies must use the same benchmark presence, source, aligned period, and return."
            )

        cost_signature = (
            report.get("cost_mode"),
            report.get("buy_cost_bps_per_full_turnover"),
            report.get("sell_cost_bps_per_full_turnover"),
            bool(report.get("frictionless", False)),
        )
        cost_signatures.append(cost_signature)
        unavailable = [metric for metric in metric_names if metric not in report]
        rows.append(
            {
                "name": name,
                "strategy": report.get("strategy"),
                "cost_mode": report.get("cost_mode", "unspecified"),
                "frictionless": bool(report.get("frictionless", False)),
                "input_freshness_status": input_validation.get("status", "unknown"),
                "metrics": {metric: report.get(metric) for metric in metric_names},
                "missing_metrics": unavailable,
                "unavailable_metrics": [
                    metric for metric in metric_names
                    if metric in report and report[metric] is None
                ],
                "cost_description": report.get("cost_description", "unspecified"),
                "buy_cost_bps_per_full_turnover": report.get("buy_cost_bps_per_full_turnover"),
                "sell_cost_bps_per_full_turnover": report.get("sell_cost_bps_per_full_turnover"),
            }
        )

    same_costs = all(signature == cost_signatures[0] for signature in cost_signatures[1:])
    warnings = []
    if not same_costs:
        frictionless_flags = [signature[2] for signature in cost_signatures]
        if any(frictionless_flags) and not all(frictionless_flags):
            warnings.append(
                "Frictionless and costed results are mixed; they are not directly comparable."
            )
        else:
            warnings.append("Cost assumptions differ; results are not directly comparable.")
    input_freshness_statuses = [row["input_freshness_status"] for row in rows]
    freshness_validated = all(status == "fresh_validated" for status in input_freshness_statuses)
    if not freshness_validated:
        warnings.append(
            "Input freshness/provenance is not fully validated for every strategy; comparison is diagnostic only."
        )
    return {
        "period_start": requested_start,
        "period_end": requested_end,
        "comparable": same_costs and freshness_validated,
        "warnings": warnings,
        "benchmark": benchmark_signature,
        "strategies": rows,
        "ranking": None,
        "interpretation": "Historical comparison only; no strategy is selected and past results do not predict future performance.",
    }


def chronological_periods(
    data: pd.Series | pd.DataFrame,
    *,
    development_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    price_column: str = "Adj Close",
) -> dict[str, pd.Series]:
    """Split observations in time order into development/validation/final periods.

    The final segment is only a genuine holdout when its dates and outcomes
    have not previously informed model or strategy choices. Old project
    experiments already examined much of the supplied historical dataset.
    """
    prices = _prices(data, price_column)
    if not 0 < development_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Development and validation fractions must be between 0 and 1.")
    if development_fraction + validation_fraction >= 1:
        raise ValueError("Development and validation fractions must leave a final evaluation period.")
    development_end = int(len(prices) * development_fraction)
    validation_end = int(len(prices) * (development_fraction + validation_fraction))
    if development_end < 2 or validation_end <= development_end or validation_end >= len(prices):
        raise ValueError("Not enough observations for nonempty chronological periods.")
    return {
        "development": prices.iloc[:development_end],
        "validation": prices.iloc[development_end:validation_end],
        "final_evaluation": prices.iloc[validation_end:],
    }


def run_chronological_evaluation(
    data: pd.Series | pd.DataFrame,
    strategy: str,
    *,
    development_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    **backtest_options,
) -> dict[str, object]:
    """Backtest a fixed strategy once and report chronological subperiods.

    Pass strategy windows and cost assumptions before inspecting the final
    segment. This helper reports all periods but cannot prevent a caller from
    tuning against them. Prior-examined dates are retrospective regardless of
    how this function labels their segments.
    """
    price_column = backtest_options.get("price_column", "Adj Close")
    periods = chronological_periods(
        data,
        development_fraction=development_fraction,
        validation_fraction=validation_fraction,
        price_column=price_column,
    )
    result = run_backtest(data, strategy, **backtest_options)
    period_results = {}
    for name, prices in periods.items():
        period_results[name] = {
            "start_date": prices.index[0],
            "end_date": prices.index[-1],
            **summarize_returns(
                result.daily,
                start=prices.index[0],
                end=prices.index[-1],
            ),
        }
    return {"backtest": result, "periods": period_results}
