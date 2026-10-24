"""Interactive educational dashboard for historical market data and models."""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from exploratory_analysis import price_volume_figure, summarize_market_data
from forecasting import recursive_forecast
from model_training import train_and_evaluate
from preprocessing import format_retrieval_time, get_latest_candle
from technical_indicators import add_technical_indicators
from market_data_service import MarketDataService, map_nse_symbol
from investment_analysis import (
    assess_data_freshness,
    calculate_investing_metrics,
    compare_benchmark_returns,
    read_benchmark_csv,
)
from backtesting import BacktestInputMetadata, build_backtest_report, run_validated_backtest

st.set_page_config(page_title="Stock Market Analysis", page_icon="📈", layout="wide")
st.title("Stock Market Data Analysis & Experimental Forecasting")
st.caption("Educational analysis of historical market data. Forecasts are uncertain experiments, not investment advice.")


@st.cache_resource(show_spinner=False)
def get_market_data_service() -> MarketDataService:
    return MarketDataService()


auto_refresh = st.sidebar.toggle(
    "Automatic display refresh every 5 seconds",
    value=True,
    help="Display checks run every 5 seconds. Each provider is queried at most once per ticker and interval every 30 seconds.",
)


@st.fragment(run_every="5s" if auto_refresh else None)
def render_dashboard() -> None:
    with st.sidebar:
        st.header("Data settings")
        market = st.selectbox("Market", ["US", "India"])
        examples = ["AAPL", "MSFT", "GOOGL"] if market == "US" else ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        ticker = st.selectbox("Example ticker", examples)
        ticker = st.text_input("Or enter a ticker", value=ticker, max_chars=20).strip().upper()
        interval_labels = {"Daily (1d)": "1d", "5-minute (5m)": "5m", "15-minute (15m)": "15m"}
        interval_label = st.selectbox("Data interval", list(interval_labels), index=0)
        interval = interval_labels[interval_label]
        daily_ranges = ["1 month", "3 months", "6 months", "1 year", "2 years", "5 years", "Custom"]
        intraday_ranges = ["1 day", "5 days", "1 month", "Custom"]
        range_choices = daily_ranges if interval == "1d" else intraday_ranges
        range_choice = st.selectbox("Historical period", range_choices)
        today = date.today()
        if range_choice == "Custom":
            lookback_days = 365 if interval == "1d" else 5
            default_end = today if interval == "1d" else today + timedelta(days=1)
            start_date = st.date_input("Start date", value=today - timedelta(days=lookback_days), max_value=today)
            end_date = st.date_input("End date (exclusive)", value=default_end, max_value=today + timedelta(days=1))
        else:
            start_date = end_date = None
        use_adjusted = st.checkbox(
            "Use adjusted close when available",
            value=True,
            help="Adjusted Close accounts for distributions and splits when supplied. Uncheck to analyze historical traded Close instead.",
        )
        horizon = st.selectbox("Forecast horizon (trading days)", [1, 5, 10], index=1)
        refresh = st.button("Refresh downloaded data")
        if refresh:
            st.caption("Refresh requested. Provider cooldowns still apply.")

    periods = {
        "1 month": "1mo", "3 months": "3mo", "6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y",
        "1 day": "1d", "5 days": "5d",
    }
    if range_choice == "Custom" and start_date >= end_date:
        st.error("Choose a start date earlier than the exclusive end date.")
        st.stop()

    period = periods.get(range_choice, "1mo" if interval != "1d" else "1y")
    request_key = (ticker, period, str(start_date), str(end_date), interval)
    cached_market_result = st.session_state.get("_last_dashboard_market_result")
    submitted_backtest = bool(st.session_state.get("backtest_run_submit", False))
    if (
        submitted_backtest
        and isinstance(cached_market_result, dict)
        and cached_market_result.get("request_key") == request_key
    ):
        result = cached_market_result["result"]
    else:
        with st.spinner(f"Checking historical {interval} data for {ticker}..."):
            result = get_market_data_service().fetch(ticker, period, start_date, end_date, interval)
    st.session_state["_last_dashboard_market_result"] = {
        "request_key": request_key,
        "result": result,
    }
    if result.data is None:
        for error in result.errors:
            st.error(error)
        st.error("No successful dataset is available for this selection yet.")
        st.stop()
    history = result.data
    if result.stale:
        st.error("STALE SNAPSHOT: provider checks failed or were cooling down. The last successful dataset is shown and is not confirmed current.")
    if result.errors:
        st.warning("Provider status: " + " | ".join(result.errors))

    price_col = "Adj Close" if use_adjusted and "Adj Close" in history else "Close"
    if price_col not in history:
        st.error("The downloaded data does not contain a usable closing price column.")
        st.stop()
    try:
        summary = summarize_market_data(history, price_col)
        investment_metrics = calculate_investing_metrics(history, price_col) if interval == "1d" else None
    except ValueError as exc:
        st.error(f"Selected price data is insufficient or invalid for analysis: {exc}")
        st.stop()
    freshness = assess_data_freshness(
        history, price_col, retrieved_at=result.retrieved_at, stale=result.stale
    )
    currency_note = "Currency depends on the exchange; Yahoo Finance data does not provide a reliable currency label here."
    st.caption(f"{ticker} · downloaded candles {history.index.min():%Y-%m-%d} to {history.index.max():%Y-%m-%d} · {currency_note}")
    retrieval_label = format_retrieval_time(history)
    retrieval_text = retrieval_label or "retrieval time unavailable"
    st.caption(f"Selected ticker: {ticker} | interval: {interval} | actual source for displayed dataset: {result.source} | last successful retrieval: {retrieval_text}")
    st.caption(
        f"Data status: {freshness['status']} · last valid {price_col} observation "
        f"{freshness['latest_valid_price_date']:%Y-%m-%d} · observation age {freshness['observation_age_days']} calendar day(s)."
    )
    for freshness_warning in freshness["warnings"]:
        if not (result.stale and "last successful snapshot" in freshness_warning):
            st.warning(f"Freshness: {freshness_warning}")
    latest_candle_time, latest_candle_close = get_latest_candle(history, "Close")
    if latest_candle_time is not None:
        if latest_candle_time.tzinfo is not None:
            candle_time_text = latest_candle_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            candle_time_text = f"{latest_candle_time:%Y-%m-%d %H:%M:%S} (timezone not supplied)"
        close_text = f"{latest_candle_close:,.2f}" if latest_candle_close is not None else "unavailable"
        st.caption(f"Latest available candle: {candle_time_text} | raw close: {close_text} | completion is unverified; it may still be forming.")
        if latest_candle_time.tzinfo is not None and interval != "1d":
            candle_age = pd.Timestamp.now(tz=latest_candle_time.tz).tz_convert("UTC") - latest_candle_time.tz_convert("UTC")
            if candle_age.total_seconds() >= 0:
                st.caption(f"Approximate candle age: {candle_age} (timestamp estimate; freshness is unverified).")
            else:
                st.caption("Candle age unavailable because the timestamp is ahead of the local clock.")
    quality_issues = list(history.attrs.get("quality_issues", []))
    for quality_issue in quality_issues:
        st.warning(f"Data quality warning: {quality_issue} No missing values or candles were filled.")
    for service_warning in result.warnings:
        if service_warning not in quality_issues:
            st.warning(service_warning)
    if interval == "1d":
        st.caption("Daily historical data. Retrieval time does not imply a live or real-time feed.")
    else:
        st.info(f"Historical intraday data from {result.source} ({interval}). Freshness is unverified; data is not guaranteed real-time.")
    if map_nse_symbol(ticker) is not None and result.source == "Yahoo Finance (yfinance)":
        st.warning("OpenChart was unavailable or failed validation; this dataset is the separate Yahoo Finance fallback.")
    latest_volume = (
        history.loc[summary["latest_close_date"], "Volume"]
        if "Volume" in history and summary["latest_close_date"] in history.index
        else None
    )
    if latest_volume is not None and pd.isna(latest_volume):
        latest_volume = None
    latest_return = summary["latest_return"]
    cols = st.columns(5)
    close_label = "Latest valid adjusted close" if price_col == "Adj Close" else "Latest valid close"
    if result.stale:
        close_label = f"STALE snapshot {price_col} · {summary['latest_close_date']:%Y-%m-%d}"
    cols[0].metric(f"{close_label} · {summary['latest_close_date']:%Y-%m-%d}", f"{summary['latest_close']:,.2f}")
    cols[1].metric("Daily return" if interval == "1d" else "Latest bar return", f"{latest_return * 100:+.2f}%" if pd.notna(latest_return) else "—")
    cols[2].metric("Period high", f"{summary['highest_close']:,.2f}")
    cols[3].metric("Period low", f"{summary['lowest_close']:,.2f}")
    cols[4].metric("Latest volume (shares)", f"{latest_volume:,.0f}" if latest_volume is not None else "—")
    st.caption(f"{summary['candles']} candles · {summary['observations']} valid closes · average close {summary['average_close']:,.2f} · mean daily reported volume {summary['average_volume']:,.0f} shares · total reported volume {summary['total_reported_volume']:,.0f} shares")
    st.caption(
        f"Analysis price basis: {price_col}. Adjusted Close reflects distributions/splits when supplied; "
        "it can differ from the historical traded Close."
    )

    if interval == "1d":
        st.subheader("Recent performance")
        recent_columns = st.columns(3)
        for column, sessions in zip(recent_columns, (1, 5, 20)):
            value = investment_metrics["recent_returns"].get(sessions)
            column.metric(
                f"{sessions}-session return",
                f"{value:+.2%}" if value is not None else "Insufficient clean data",
            )
        st.caption("Returns compound consecutive observed-session returns. Missing prices are not filled; windows containing missing observations are reported as unavailable.")

        st.subheader("Risk overview")
        risk_columns = st.columns(3)
        annualized_volatility = investment_metrics["annualized_volatility"]
        risk_columns[0].metric(
            "20-session annualized volatility",
            f"{annualized_volatility:.2%}" if annualized_volatility is not None else "Insufficient clean data",
        )
        max_drawdown = investment_metrics["max_drawdown"]
        risk_columns[1].metric(
            "Maximum drawdown · selected period",
            f"{max_drawdown:.2%}" if max_drawdown is not None else "Insufficient observations",
        )
        volume_ratio = investment_metrics["latest_volume_vs_prior_median"]
        risk_columns[2].metric(
            "Volume / prior 20-session median at latest price date",
            f"{volume_ratio:.2f}×" if volume_ratio is not None else "Unavailable",
        )
        st.caption(
            "Volatility = sample standard deviation of the last 20 daily returns × √252. "
            "Maximum drawdown = minimum (price / running peak − 1) across valid prices in the selected period. "
            "Volume compares the latest reported value with the prior 20 observations. These are historical descriptions, not forecasts."
        )
        for risk_warning in investment_metrics["warnings"]:
            st.warning(f"Risk/data warning: {risk_warning}")

        st.subheader("Benchmark comparison")
        st.caption(
            "Optional, offline comparison: upload a daily benchmark CSV (for example, NIFTY 50 for an Indian equity) "
            f"with Date and {price_col} columns. Only matching dates are compared; dates are never forward-filled."
        )
        benchmark_file = st.file_uploader(
            "Benchmark daily CSV · no provider request is made",
            type=["csv"],
            key="benchmark_csv",
        )
        benchmark_prices = None
        benchmark_file_name = None
        if benchmark_file is not None:
            try:
                benchmark_prices = read_benchmark_csv(benchmark_file, price_col)
                benchmark_file_name = benchmark_file.name
                comparison = compare_benchmark_returns(history[price_col], benchmark_prices)
                benchmark_columns = st.columns(3)
                benchmark_columns[0].metric("Stock absolute return", f"{comparison['stock_absolute_return']:+.2%}")
                benchmark_columns[1].metric("Benchmark absolute return", f"{comparison['benchmark_absolute_return']:+.2%}")
                benchmark_columns[2].metric("Stock relative return", f"{comparison['benchmark_relative_return']:+.2%}")
                st.caption(
                    f"Matched price dates: {comparison['start_date']:%Y-%m-%d} to {comparison['end_date']:%Y-%m-%d} "
                    f"({comparison['matched_price_dates']} dates, {comparison['matched_return_intervals']} return intervals). "
                    "Relative return = (1 + stock return) / (1 + benchmark return) − 1. Uploaded benchmark freshness and source are not verified."
                )
            except ValueError as exc:
                st.warning(f"Benchmark comparison unavailable: {exc}")

    st.subheader("Experimental daily backtest")
    st.caption(
        "Opt-in historical simulation using the data already loaded above. It does not make a new provider request. "
        "Results are descriptive research, not forecasts, recommendations, or evidence of profitability."
    )
    if market != "India" or not ticker.endswith(".NS"):
        st.info("The initial backtest panel is limited to NSE cash-equity tickers ending in .NS.")
    elif interval != "1d":
        st.info("Backtesting is available only for daily data; intraday bar completeness is not validated.")
    else:
        backtest_index = pd.DatetimeIndex(pd.to_datetime(history.index, errors="coerce"))
        if backtest_index.isna().any():
            st.error("Backtest unavailable: the loaded data contains invalid dates.")
        else:
            if backtest_index.tz is not None:
                backtest_index = backtest_index.tz_convert("Asia/Kolkata").tz_localize(None)
            normalized_history = history.copy()
            normalized_history.index = backtest_index.normalize()
            first_available_date = normalized_history.index.min().date()
            last_available_date = normalized_history.index.max().date()
            with st.form("experimental_daily_backtest_form"):
                strategy = st.selectbox(
                    "Backtest strategy",
                    ["buy_and_hold", "no_trade", "sma_crossover"],
                    format_func=lambda value: {
                        "buy_and_hold": "Buy and hold",
                        "no_trade": "No trade (cash baseline)",
                        "sma_crossover": "SMA crossover (20 / 50 sessions)",
                    }[value],
                    key="backtest_strategy",
                )
                selected_dates = st.date_input(
                    "Evaluation date range (inclusive)",
                    value=(first_available_date, last_available_date),
                    min_value=first_available_date,
                    max_value=last_available_date,
                    key="backtest_date_range",
                )
                basis_options = [column for column in ("Adj Close", "Close") if column in normalized_history]
                if not basis_options:
                    st.error("Backtest unavailable: neither Adj Close nor Close is present.")
                    backtest_basis = None
                else:
                    default_basis = price_col if price_col in basis_options else basis_options[0]
                    backtest_basis = st.selectbox(
                        "Backtest price basis",
                        basis_options,
                        index=basis_options.index(default_basis),
                        help="Adjusted Close is preferred for return accounting. Close is the historical unadjusted close.",
                        key="backtest_price_basis",
                    )
                cost_choice = st.selectbox(
                    "Illustrative NSE delivery cost",
                    ["Medium", "Low", "High", "ZERO COST / FRICTIONLESS"],
                    help="Low/medium/high are configurable assumptions, not a personal cost quote.",
                    key="backtest_cost_choice",
                )
                include_benchmark = st.checkbox(
                    "Compare with uploaded benchmark (caller-provided and unverified)",
                    value=False,
                    disabled=benchmark_prices is None,
                    key="backtest_include_benchmark",
                )
                allow_stale_override = st.checkbox(
                    "Allow stale data for diagnostic/historical analysis",
                    value=False,
                    help="This explicit override is recorded in the report. It cannot bypass missing provenance, invalid data, or the daily-only rule.",
                    key="backtest_allow_stale_override",
                )
                submitted = st.form_submit_button(
                    "Run selected backtest",
                    key="backtest_run_submit",
                )

            if submitted:
                if backtest_basis is None:
                    st.error("Backtest unavailable: select a usable price basis.")
                elif not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
                    st.error("Choose both the start and end dates for the evaluation period.")
                else:
                    start_date, end_date = selected_dates
                    start_stamp = pd.Timestamp(start_date)
                    end_stamp = pd.Timestamp(end_date)
                    if start_stamp > end_stamp:
                        st.error("The evaluation start date must not be after its end date.")
                    else:
                        evaluation_data = normalized_history.loc[
                            (normalized_history.index >= start_stamp)
                            & (normalized_history.index <= end_stamp)
                        ]
                        evaluation_time = pd.Timestamp.now(tz="Asia/Kolkata")
                        selected_freshness = None
                        input_setup_error = None
                        if not evaluation_data.empty and backtest_basis in evaluation_data:
                            try:
                                selected_freshness = assess_data_freshness(
                                    evaluation_data,
                                    backtest_basis,
                                    retrieved_at=result.retrieved_at,
                                    stale=result.stale,
                                    now=evaluation_time,
                                )
                            except ValueError as exc:
                                input_setup_error = str(exc)
                        latest_observation = (
                            selected_freshness["latest_valid_price_date"]
                            if selected_freshness is not None
                            else None
                        )
                        freshness_status = (
                            "Stale cached snapshot"
                            if result.stale
                            else selected_freshness["status"] if selected_freshness else "unknown"
                        )
                        metadata_warnings = (
                            list(selected_freshness["warnings"])
                            if selected_freshness is not None
                            else ["Selected evaluation range has no usable observations."]
                        )
                        metadata_warnings.extend(result.warnings)
                        metadata_warnings.extend(result.errors)
                        metadata_warnings.extend(quality_issues)
                        input_metadata = BacktestInputMetadata(
                            source=result.source,
                            latest_observation_at=latest_observation,
                            retrieved_at=result.retrieved_at,
                            evaluated_at=evaluation_time,
                            interval=interval,
                            freshness_status=freshness_status,
                            warnings=tuple(dict.fromkeys(str(item) for item in metadata_warnings if item)),
                            synthetic=bool(history.attrs.get("synthetic", False)),
                        )
                        selected_benchmark = None
                        selected_benchmark_label = None
                        benchmark_setup_error = None
                        if include_benchmark:
                            if benchmark_prices is None:
                                benchmark_setup_error = "Benchmark comparison unavailable: no benchmark CSV is loaded."
                            if evaluation_data.empty:
                                benchmark_setup_error = "Backtest unavailable: the selected period contains no observations."
                            if benchmark_prices is not None and not evaluation_data.empty:
                                selected_benchmark = benchmark_prices.loc[
                                    (benchmark_prices.index >= evaluation_data.index.min())
                                    & (benchmark_prices.index <= evaluation_data.index.max())
                                ]
                                selected_benchmark_label = (
                                    f"Uploaded CSV {benchmark_file_name!r} (caller-provided; unverified)"
                                )
                        cost_scenario = {
                            "Low": "low",
                            "Medium": "medium",
                            "High": "high",
                            "ZERO COST / FRICTIONLESS": None,
                        }[cost_choice]
                        try:
                            if input_setup_error is not None:
                                raise ValueError(f"Selected price data is invalid: {input_setup_error}")
                            if benchmark_setup_error is not None:
                                raise ValueError(benchmark_setup_error)
                            backtest_result = run_validated_backtest(
                                evaluation_data,
                                strategy,
                                input_metadata=input_metadata,
                                allow_stale_override=allow_stale_override,
                                price_column=backtest_basis,
                                cost_scenario=cost_scenario,
                                benchmark=selected_benchmark,
                                benchmark_source=selected_benchmark_label,
                                benchmark_price_basis=price_col if selected_benchmark is not None else None,
                            )
                            backtest_report = build_backtest_report(backtest_result)
                            st.subheader("Backtest report")
                            st.caption(
                                f"Evaluation period: {backtest_report['evaluation_start']:%Y-%m-%d} to "
                                f"{backtest_report['evaluation_end']:%Y-%m-%d} · "
                                f"price basis: {backtest_report['price_basis']} · "
                                f"freshness: {backtest_report['input_freshness_status']} · "
                                f"{backtest_report['evaluation_observations']} observations"
                            )
                            validation = backtest_report["input_validation"]
                            retrieval_time = validation["retrieved_at"]
                            retrieval_text = (
                                retrieval_time.isoformat() if retrieval_time is not None else "unavailable"
                            )
                            source_text = validation["source"] or "unknown"
                            st.caption(
                                f"Reported provider label: {source_text} (not independently authenticated) · "
                                f"latest observation: {validation['latest_observation_at']} · "
                                f"retrieved at: {retrieval_text} · "
                                f"evaluation timestamp: {validation['evaluated_at']}"
                            )
                            if backtest_report["frictionless"]:
                                st.error("ZERO COST / FRICTIONLESS — costs are disabled for this diagnostic comparison.")
                            else:
                                st.info(f"Transaction-cost assumptions: {backtest_report['cost_description']}")
                            for warning in backtest_report["input_validation"]["warnings"]:
                                st.warning(f"Backtest input: {warning}")
                            metric_columns = st.columns(7)
                            report_metrics = (
                                ("Gross cumulative return", "gross_cumulative_return"),
                                ("Net cumulative return", "net_cumulative_return"),
                                ("CAGR", "cagr"),
                                ("Maximum drawdown", "maximum_drawdown"),
                                ("Annualized volatility", "annualized_volatility"),
                                ("Turnover", "turnover"),
                                ("Executed orders", "trade_count"),
                            )
                            for column, (label, metric_key) in zip(metric_columns, report_metrics):
                                value = backtest_report[metric_key]
                                if metric_key in {"turnover", "trade_count"}:
                                    value_text = f"{value:.2f}" if value is not None else "Unavailable"
                                else:
                                    value_text = f"{value:.2%}" if value is not None else "Unavailable"
                                column.metric(label, value_text)
                            if backtest_report["benchmark"] is not None:
                                benchmark_report = backtest_report["benchmark"]
                                st.write(
                                    f"Benchmark relative return: {benchmark_report['relative_return']:+.2%} · "
                                    f"dates: {benchmark_report['period_start']:%Y-%m-%d} to "
                                    f"{benchmark_report['period_end']:%Y-%m-%d} · "
                                    f"source label: {benchmark_report['source']} · "
                                    f"provenance: {benchmark_report['source_provenance']} · "
                                    f"independently verified: {benchmark_report['source_independently_verified']}"
                                )
                                st.warning(
                                    "The benchmark label and uploaded data are caller-provided; the app does not authenticate their origin or freshness."
                                )
                            st.caption("Limitations: " + " ".join(backtest_report["limitations"]))
                            st.caption(
                                "This is a historical simulation, not a forecast or recommendation. "
                                "No strategy is ranked; costs, adjusted-close execution, and observed-session timing are assumptions."
                            )
                        except ValueError as exc:
                            policy_assessment = getattr(exc, "assessment", None)
                            if policy_assessment:
                                st.error(
                                    f"Backtest blocked by input policy ({policy_assessment['status']}): {exc}"
                                )
                                for warning in policy_assessment.get("warnings", []):
                                    st.warning(warning)
                            else:
                                st.error(f"Backtest unavailable: {exc}")

    st.subheader("Historical analysis")
    st.plotly_chart(price_volume_figure(history, price_col), use_container_width=True)
    if interval != "1d":
        st.info("Technical indicators, model evaluation, and forecasts remain on the existing daily workflow. Intraday forecasting is disabled.")
        with st.expander("View cleaned historical intraday data"):
            st.dataframe(history.sort_index(ascending=False), use_container_width=True)
        st.stop()

    analysis = add_technical_indicators(history, price_col)
    price_fig = go.Figure()
    for column in [price_col, "SMA20", "SMA50", "EMA20"]:
        price_fig.add_trace(go.Scatter(x=analysis.index, y=analysis[column], name=column))
    price_fig.update_layout(title="Closing price and trailing moving averages", yaxis_title="Price (exchange currency)", hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)
    st.caption(
        "SMA20/SMA50 and EMA20 are trailing averages that summarize recent price levels. RSI14 summarizes recent gains/losses on a 0–100 scale; MACD compares short and long exponential averages. "
        "These indicators describe past prices and are not guaranteed buy or sell signals."
    )
    indicator_left, indicator_right = st.columns(2)
    with indicator_left:
        st.plotly_chart(go.Figure(go.Scatter(x=analysis.index, y=analysis["Daily Return"] * 100, name="Daily return")).update_layout(title="Daily returns (%)", yaxis_title="Percent"), use_container_width=True)
        st.plotly_chart(go.Figure(go.Scatter(x=analysis.index, y=analysis["RSI14"], name="RSI14")).update_layout(title="RSI (14 sessions, Wilder smoothing)", yaxis_title="RSI", yaxis_range=[0, 100]), use_container_width=True)
    with indicator_right:
        st.plotly_chart(go.Figure(go.Scatter(x=analysis.index, y=analysis["Rolling Volatility"] * 100, name="Annualized volatility")).update_layout(title="20-session rolling volatility (annualized, %)", yaxis_title="Percent"), use_container_width=True)
        macd_fig = go.Figure()
        for col in ["MACD", "MACD Signal", "MACD Histogram"]:
            macd_fig.add_trace(go.Scatter(x=analysis.index, y=analysis[col], name=col))
        st.plotly_chart(macd_fig.update_layout(title="MACD (12, 26, 9)"), use_container_width=True)
    with st.expander("Return distribution and data definitions"):
        st.write(f"Daily return mean: {summary['daily_returns'].mean():.4%}; standard deviation: {summary['daily_returns'].std():.4%}. Average volume is the mean per observed session; total reported volume sums the period. Adjusted prices account for distributions/splits when supplied and may differ from traded close.")
        st.plotly_chart(go.Figure(go.Histogram(x=summary["daily_returns"] * 100, nbinsx=40)).update_layout(title="Daily return distribution", xaxis_title="Daily return (%)", yaxis_title="Sessions"), use_container_width=True)

    st.subheader("Chronological model evaluation")
    st.info("The NIFTY 50 research to date has not shown a reliable predictive improvement over simple baselines. Models shown here are experimental comparisons for this selected ticker and period, not validated investment signals.")
    st.write("Target: next trading session's closing price. Earliest 70% fits models, the next 15% is reserved for validation, and the final 15% is held out for reported metrics. No rows are shuffled. R-squared can be negative and is not a measure of investment return.")
    try:
        result = train_and_evaluate(history, price_col)
        st.dataframe(result.metrics.style.format("{:.4f}"), use_container_width=True)
        st.caption(
            f"Models: Linear Regression and Random Forest, predicting next-session {price_col}. "
            f"They fit on the earliest 70% of complete rows, with training labels through {result.train_end:%Y-%m-%d}; "
            f"the final 15% test features span {result.predictions.index.min():%Y-%m-%d} to {result.predictions.index.max():%Y-%m-%d}. "
            f"Recursive forecast horizon: {horizon} trading sessions. The middle 15% is reserved and not used for tuning."
        )
        st.caption("Directional-accuracy caveat: Naive Persistence predicts no direction, so its sign-based score is near zero by construction. That score should not be read as a fair directional-skill comparison; the NIFTY research uses training-only directional baselines and found no reliable edge.")
        selected_model = st.selectbox("Model for future forecast", ["Linear Regression", "Random Forest"], index=0)
        test_fig = go.Figure()
        for col in ["Actual", "Naive Persistence", "Linear Regression", "Random Forest"]:
            test_fig.add_trace(go.Scatter(x=result.predictions.index, y=result.predictions[col], name=col))
        st.plotly_chart(test_fig.update_layout(title="Held-out test: next-session actual vs predicted", xaxis_title="Feature date (forecast made after close)", yaxis_title="Close (exchange currency)"), use_container_width=True)
        st.subheader("Recursive experimental forecast")
        st.write("Each predicted close is fed into the next step's lag and indicator calculations. Forecast dates are business days; exchange holidays are not modeled.")
        forecast = recursive_forecast(history, result.models[selected_model], int(horizon), price_col)
        st.dataframe(forecast.style.format("{:.2f}"), use_container_width=True)
        future_fig = go.Figure()
        future_fig.add_trace(go.Scatter(x=history.index[-60:], y=history[price_col].iloc[-60:], name="Historical close"))
        future_fig.add_trace(go.Scatter(x=forecast.index, y=forecast["Predicted Close"], name="Experimental forecast", mode="lines+markers"))
        st.plotly_chart(future_fig.update_layout(title=f"{horizon}-session recursive forecast · {selected_model}", yaxis_title="Price (exchange currency)"), use_container_width=True)
        st.caption(f"Forecast details: {selected_model}; {horizon} recursive session(s); {price_col} basis; model training labels end {result.train_end:%Y-%m-%d}. No confidence interval is estimated, and this model has not established a reliable predictive edge.")
    except ValueError as exc:
        st.info(f"Modeling is unavailable for this date range: {exc}")

    st.warning("Forecasts are statistical experiments based on historical patterns. Markets are affected by new information, regime changes, liquidity, corporate actions, and other factors. These outputs are not financial advice and do not guarantee future prices or profitable decisions.")
    with st.expander("View cleaned daily data"):
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)


render_dashboard()
