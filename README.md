# Stock Market Data Analysis and Trend Prediction

An educational Python application for inspecting daily historical stock data, calculating common technical indicators, and comparing experimental next-session price models. Yahoo Finance data is historical and may be delayed or revised. Nothing in this project is investment advice.

## Objectives and features

- Retrieve historical NSE data through OpenChart when available, with Yahoo Finance as a separate fallback; Yahoo Finance remains available for US tickers.
- Review date coverage, closing prices, returns, volume, and price trends interactively.
- Calculate trailing moving averages, Wilder RSI, MACD, daily returns, and annualized rolling volatility.
- Compare naive persistence, Linear Regression, and Random Forest on a chronological holdout.
- Produce optional recursive 1, 5, or 10 trading-session experimental forecasts.

## Technologies and architecture

Python 3.11 or 3.12, Pandas, NumPy, OpenChart, yfinance, Plotly, Streamlit, scikit-learn, and pytest. The dashboard calls reusable modules: provider service → data collection → preprocessing → indicators/EDA → modeling → recursive forecasting. Models are fitted at runtime and are not saved.

## Project structure

```text
app.py                     Streamlit user interface
data_collection.py         Yahoo Finance daily and short intraday downloads
market_data_service.py     OpenChart-first NSE retrieval, Yahoo fallback, cooldown, last-good retention
market_data_storage.py    Provider-separated SQLite candle persistence
preprocessing.py           Validation and cleanup
exploratory_analysis.py    Summary statistics and Plotly charts
technical_indicators.py   Causal indicator calculations
model_training.py         Chronological evaluation
forecasting.py             Recursive experimental forecasts
backtesting.py             Offline long/flat strategy backtest engine
tests/                     Offline synthetic-data tests, including mocked downloads
data/                      Reserved for local, ignored datasets
requirements.txt           Runtime and test dependencies, including OpenChart 0.2.0
```

## Install and run (Windows PowerShell)

From the project directory:

```powershell
cd X:\stock-market-predicton
```

Use the existing `.venv` if it is healthy. Check its Python version; the project targets Python 3.11 or 3.12.

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `.venv` was created with an unsupported Python version, use an installed 3.11/3.12 launcher to make a new environment in another project-local folder, preserving the existing environment:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Press Ctrl+C in the same terminal to stop it. Internet access is needed for either market-data provider. If a managed Windows environment denies Streamlit's default configuration path, use the project-local temporary configuration workaround you already have.

## Usage

Pick a US or Indian market example, or type another ticker. NSE `.NS` symbols are mapped to exact NSE equity symbols (for example, `RELIANCE.NS` → `RELIANCE-EQ`) and accepted only when OpenChart returns one exact NSE equity match. Examples include `AAPL`, `MSFT`, `GOOGL`, `RELIANCE.NS`, `TCS.NS`, and `INFY.NS`. Choose a preset period or an explicit custom date range (end is exclusive), then inspect the history, indicators, test metrics, and optional forecast. Automatic display checks default to every 5 seconds. Each provider is requested at most once per ticker and interval every 30 seconds; manual refresh does not bypass that cooldown.

### Historical Indian intraday bars

The dashboard offers historical `5m` and `15m` intervals. For NSE `.NS` equities, OpenChart is tried first when installed and usable; Yahoo Finance is tried as a separate fallback. US tickers use Yahoo Finance. Availability varies by symbol and interval. Intraday timestamps are normalized to `Asia/Kolkata` for NSE symbols; OpenChart's naive timestamps are interpreted as Indian market time, not UTC. Yahoo's existing intraday range restrictions still apply on its fallback path.

The dashboard shows the actual provider, local retrieval time, latest candle timestamp, and approximate candle age when available. A five-second display check does not mean a provider is queried that often, nor does it make the data live. The 30-second provider cooldown is separate from display refresh, and failed checks retain the last successful dataset with a warning. Candle completion and freshness are unverified. OpenChart is an unofficial client for NSE's charting endpoint and is not affiliated with or endorsed by NSE. Yahoo Finance is accessed through the community `yfinance` package and is not affiliated with Yahoo. Provider availability, rate limits, historical coverage, and data-use terms can change; do not redistribute downloaded data. The daily indicators, model evaluation, and forecasts remain available only in daily mode; intraday forecasting has not been enabled.

## Local candle storage

Successful provider downloads are stored in `data/market_data.sqlite3`, created automatically the first time a valid dataset is saved. This uses Python's built-in SQLite library and adds no package dependency. The database is local and excluded from version control by the existing `data/*` ignore rule. Back it up by stopping the dashboard and copying `data/market_data.sqlite3` to a separate safe location; never place the only backup inside the project folder. SQLite journals may exist briefly while the application is writing, so copy after stopping the app for a consistent simple file backup.

Stored keys include ticker, interval, UTC timestamp, and exact provider/source. OpenChart and Yahoo Finance candles remain separate. Saving a repeated candle updates that source's prior row deterministically (last write wins). The service persists the successful response but does not merge stored rows into the dashboard result or substitute stored rows after failure. The existing in-memory cooldown and last-successful snapshot remain distinct: a stored candle has no new retrieval time, and loading stored history does not label it as freshly fetched. The storage module offers source-specific range and full-history queries for later offline analysis.

Database timestamps are stored as UTC instants with a timezone label for reconstruction. Naive NSE (`.NS`/`.BO`) timestamps are interpreted as `Asia/Kolkata`; other naive timestamps default to UTC. Timezone-aware input is converted as an instant and read back in the stored market timezone. The store rejects datasets with missing or invalid OHLCV data or impossible OHLC relationships; it does not fabricate prices or missing candles. If saving fails, the dashboard can still show the provider response and reports a storage warning.

This local dataset grows only from successful downloads requested by the app; it is not a complete exchange archive. Free-provider intraday coverage, delayed updates, rate limits, symbol availability, market holidays, and terms of use can vary. Persistence does not make data live, complete, or suitable for trading. Provider sources are never combined by the storage layer.

The collector deliberately supports only `1d`, `5m`, and `15m` in this stage. A 60-day maximum is enforced for custom intraday requests; individual ticker/interval availability can be shorter. No missing candles are fabricated. Market holidays and closed sessions naturally have no candles, so data gaps should be interpreted against exchange trading sessions and official holiday calendars.

Currency is exchange-dependent and is not inferred by the application. Reported average volume means the mean volume per session; total reported volume is the sum across available sessions.

## Indicators

- SMA20/SMA50: trailing means over 20/50 observed sessions; initial values are unavailable.
- EMA20: exponentially weighted trailing mean.
- RSI14: Wilder-smoothed gains and losses over 14 sessions, scaled 0–100.
- MACD: EMA12 minus EMA26, with EMA9 signal and histogram.
- Daily return: close-to-close percentage change.
- Rolling volatility: 20-session sample standard deviation of daily returns, annualized by √252.

Calculations use only values known through the displayed date. Adjusted Close reflects splits/distributions when supplied and is not the same as the historical traded close. Missing prices remain missing rather than being invented.

## Investing research view

For daily data, the dashboard defaults to `Adj Close` when the provider supplies it; the checkbox can switch to traded `Close`. Recent 1/5/20-session performance uses consecutive observed sessions and is unavailable when the requested tail contains missing prices. Risk cards show the sample standard deviation of the last 20 daily returns annualized by √252, maximum drawdown over the selected history (`price / running peak − 1`), and latest volume relative to the prior 20-observation median. These historical measures and price/volume warnings are descriptive, not signals.

Data freshness separates retrieval time from observation date. A cached response after provider failure is explicitly marked stale. The dashboard does not fill missing prices or claim that the newest observation is a live quote. An optional benchmark comparison accepts a user-supplied daily CSV with `Date` and the same selected price basis (`Adj Close` or `Close`); it compares only matching dates, without forward filling or making a provider request. The uploaded benchmark's source and freshness are not verified.

### Market-data quality and freshness safeguards

Provider timestamps that cannot be parsed are excluded with a quality warning; duplicate timestamps and out-of-order input are reported before preprocessing sorts the data and retains the last duplicate. Missing numeric values remain gaps. Quality checks report missing OHLCV columns/values, nonnumeric or non-finite values, nonpositive OHLC/adjusted prices, negative volume, and impossible High/Low relationships. The dashboard now stops selected-price summaries when supplied values are nonnumeric, non-finite, zero, or negative. Other field inconsistencies remain visible as quality warnings, and the SQLite storage layer rejects quality-flagged rows if persistence is attempted.

Freshness uses the latest valid selected-price observation, converts its timestamp to Asia/Kolkata for the reported observation date and age, and separately normalizes retrieval metadata to that timezone. Provider fallback/cooldown snapshots are marked stale; old retrieval metadata, old observations, future observations, and future retrieval timestamps are warned about. The observation-age threshold is calendar days, so weekends and holidays can contribute to it.

No exchange calendar is bundled: the app cannot distinguish a missing trading session from a market holiday or provider omission, and it does not infer missing dates from calendar-day gaps. Daily data gaps are not forward-filled. A 20% one-session move in recent daily data triggers a review warning only; it is not rejected because splits, dividends, and legitimate market moves can cause large changes. Yahoo responses may still be displayed with quality warnings for non-selected OHLCV fields; invalid selected prices fail the dashboard summary. Intraday data freshness and bar completion remain unverified.

The model panel identifies the selected price basis, training period, chronological split, and recursive horizon. Historical NIFTY 50 research has not established a reliable predictive edge; forecasts remain experimental and are not approved investment signals.

## Experimental backtesting engine

`backtesting.py` is an offline, independently testable engine exposed through an opt-in dashboard panel. It requires `Adj Close` by default; a caller can explicitly select another price column. Buy-and-hold and no-trade baselines and a long/flat simple-moving-average crossover are available. SMA crossover requires enough observations for its configured slow window (50 sessions by default). Daily signals use data through that session's close, execute at the next observed session's close, and begin earning close-to-close returns only after that assumed fill. This deliberately delays signal returns by one more close-to-close interval. Adjusted close is only a proxy execution price because adjusted OHLC fill prices are not generally supplied.

`build_backtest_report(result, periods=...)` provides presentation-ready fields for cost mode/preset, gross and net cumulative return, benchmark dates and provenance status, chronological segment descriptions, and limitations. Benchmark and strategy inputs must have exactly matching normalized daily dates; absent observations or NaN prices are rejected, with no filling or interpolation. A caller-supplied `benchmark_source` or series name is only a label: reports mark it `caller_asserted`, set `source_independently_verified=False`, and retain a fingerprint that identifies the supplied aligned values but does not authenticate them. Unlabeled data is marked `unknown`. No current API independently verifies benchmark origin. For a Series, `price_column` is the assumed price basis; a recognized conflicting Series name or an explicit conflicting `benchmark_price_basis` is rejected. DataFrame inputs select the requested price column. `run_chronological_evaluation()` returns development, validation, and final-evaluation summaries with gross and net returns. Prior-examined periods are retrospective; the final segment is an untouched holdout only if nobody previously examined its dates or outcomes.

### Dashboard backtest panel

The **Experimental daily backtest** panel appears for NSE `.NS` tickers on the daily (`1d`) workflow. It reuses the data result already loaded by the dashboard and runs calculations only after **Run selected backtest** is clicked. The form-submit rerun reuses the matching session-local data result; it does not call the market-data service again. Normal dashboard refresh/periodic data behavior is unchanged. This panel supports buy-and-hold, no-trade, and the fixed 20/50-session SMA crossover; it does not expose parameter tuning or strategy ranking. The user chooses an inclusive evaluation date range, `Adj Close`/`Close`, an illustrative low/medium/high NSE delivery-cost preset, or a separate `ZERO COST / FRICTIONLESS` mode.

The existing optional benchmark CSV upload can be included in the backtest. Its filename is a caller-provided label, not verified provenance. The selected price basis and every normalized daily observation date in the evaluation slice must match exactly; missing/extra dates and invalid values stop the benchmark comparison. Benchmark freshness and authenticity are not independently verified. The backtest report shows evaluation dates, price basis, reported provider label, latest observation/retrieval/evaluation timestamps, freshness status, gross/net return, available CAGR/drawdown/volatility, turnover, orders, costs, benchmark provenance status, and limitations. The report is historical analysis, not a forecast, recommendation, or profitability claim.

Daily inputs use the validated backtest policy: unknown source/timestamps, future timestamps, invalid prices, and intraday data are not overrideable. Data older than the daily freshness limits or marked stale is rejected unless **Allow stale data for diagnostic/historical analysis** is explicitly checked; the report then shows `stale_override` and its warnings. This permits retrospective slices without calling them fresh. A stale override does not authenticate provider or benchmark provenance. The engine has no exchange calendar, so dates missing from both supplied series cannot be classified as holidays or omitted sessions; its execution assumption is based on observed rows.

To preview the report without using the dashboard, run:

```powershell
.\.venv\Scripts\python.exe offline_backtest_preview.py
```

The preview prints frictionless and medium-cost examples from a deterministic generated price series and benchmark. Every example is labeled **synthetic demonstration data — not real market data or investment advice**. It does not access a market-data provider or SQLite and does not save result files. The production dashboard is a separate application (`streamlit run app.py`) with its existing market-data workflow; the offline preview is not a live dashboard mode and should not be interpreted as evidence of a profitable strategy.

`compare_backtest_reports(reports, period_start=..., period_end=...)` compares at least two `build_backtest_report()` outputs in their supplied order. The requested period must match every report's evaluation dates after conversion to the exchange's Asia/Kolkata calendar date; price basis and exact stock observation dates must match as well. Benchmark presence, caller label and provenance status, price basis, exact date/value fingerprint, return, and interval count must also match. Different cost modes or buy/sell rates are returned with `comparable=False` and a warning, with a specific warning when frictionless and costed results are mixed. The helper reports net cumulative return, CAGR, maximum drawdown, annualized volatility, turnover, order count, and benchmark-relative return where available. Missing and unavailable report metrics are listed; no strategy is ranked or selected. The synthetic preview includes a medium-cost comparison of the moving-average example with buy-and-hold, using the same benchmark and dates.

Subperiod summaries exclude the price return ending at the first date of that segment, because that move began in the preceding period, while retaining any execution cost charged on the segment's first date. Benchmark-relative strategy returns use the same boundary convention and include transaction cost at the first matched benchmark date.

### Backtest input freshness policy

Supply `BacktestInputMetadata` with the provider/source, latest valid observation timestamp, retrieval timestamp, evaluation timestamp, interval, provider freshness status/warnings, and a `synthetic` marker when applicable. Naive timestamps are interpreted as Asia/Kolkata; aware timestamps are converted to Asia/Kolkata. Daily defaults are a maximum observation age of 7 calendar days and maximum retrieval age of 24 elapsed hours. Future observation/retrieval timestamps and provider statuses indicating stale, old, failed, cooldown, or future data are rejected.

`run_validated_backtest(...)` is the production-style entry point: it requires known source/timestamps, daily interval policy, and agreement between the metadata observation date and the final price date. Stale data is rejected by default. `allow_stale_override=True` permits stale data only for explicit diagnostic/historical review; the report labels it `stale_override` and carries the warning. A blocked request raises `BacktestInputPolicyError`, whose `.assessment` contains the `stale_rejected` or `unknown` policy report. No freshness state is inferred from absent metadata.

The lower-level `run_backtest(...)` remains available for compatibility and numerical unit tests. Without metadata it may calculate a result, but the report must say `unknown` and must not claim that input freshness was validated. With known stale metadata it rejects by default unless `allow_stale_override=True`. Synthetic tests and the offline preview provide explicit synthetic metadata and are labeled as synthetic; they are not treated as real provider data. Daily age limits are not applied to intraday inputs because bar completion and intraday freshness are not modeled; production-style validation reports these inputs as unknown and rejects them.

Backtest summaries prominently retain `ZERO COST / FRICTIONLESS` when no costs are configured. Results are historical simulations, not forecasts or guarantees of future returns. Cost presets are assumptions; adjusted-close execution is a proxy; and the engine lacks a complete exchange calendar and broker-specific execution model.

Adjusted close supports total-return accounting, but providers may revise historical values for later corporate actions. As a result, crossover signals near splits or distributions may not match the signals that could have been computed at the time. The engine does not have point-in-time corporate-action adjustment data to eliminate this limitation.

Missing, nonnumeric, nonpositive, duplicate, or fewer-than-two daily prices are rejected; the engine never fills or silently skips them. A local-only check found no repository exchange-calendar files and no installed `exchange_calendars`, `pandas_market_calendars`, or `holidays` package in `.venv`. A missing calendar date still cannot be identified as a missing trading session, holiday, or provider omission; exact stock/benchmark date matching only detects disagreement between the supplied series. No holiday classification is inferred.

### Illustrative NSE cash-equity delivery costs

Call `run_backtest(..., cost_scenario="low" | "medium" | "high")` to use a reusable side-specific preset. Presets apply estimated costs to each executed buy or sell position change. `transaction_cost_bps` and `slippage_bps` remain supported for backward compatibility as symmetric costs; do not combine them with `cost_scenario`. Omitting all cost arguments still runs a zero-cost comparison, prominently marked `ZERO COST / FRICTIONLESS` in the returned metrics.

| Preset | Brokerage assumption, each side | Slippage assumption, each side | Total modeled buy cost | Total modeled sell cost |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.0 bps | 1.0 bps | 12.87406 bps | 11.37406 bps |
| Medium | 2.0 bps | 5.0 bps | 19.23406 bps | 17.73406 bps |
| High | 5.0 bps | 15.0 bps | 32.77406 bps | 31.27406 bps |

All presets include the same dated NSE statutory/exchange assumptions: STT 10 bps (0.1%) on both buy and sell; stamp duty 1.5 bps (0.015%) on buy only; SEBI turnover fee 0.01 bps (0.0001%) on both sides; and an NSE cash-market transaction charge plus IPFT contribution of 0.307 bps (₹307/crore) on each side. The conversion is `₹307 / ₹10,000,000 × 10,000 = 0.307 bps`. GST is modeled at 18% on brokerage, exchange, and SEBI fee amounts, excluding STT and stamp duty only as a simplifying pure-agent assumption. That exclusion is conditional: the actual GST treatment depends on the broker's invoice and whether the statutory recovery qualifies for pure-agent treatment. Check the contract note for an actual transaction. Rates were checked against official materials on 2026-10-25. The NSE rates are exchange-specific and can change.

Brokerage and slippage in the table are illustrative inputs, not official rates. Brokerage can be fixed per order, capped, or otherwise tariff-specific; a bps approximation will not represent every broker or order size. Slippage is not a statutory charge and has no universal published rate. The scenarios do not include DP/demat delivery charges, investor-specific taxes, impact from order size, or broker-specific rounding/minimum fees. Charges vary by broker, exchange, transaction type, turnover, account, and current rules. Do not treat these presets as a personal fee quote.

The preset values live in `INDIAN_DELIVERY_COST_SCENARIOS` and can be customized by constructing `IndianDeliveryCostScenario` with per-side fields. Its component formula is:

```text
buy bps = brokerage + slippage + delivery STT + NSE transaction/IPFT
          + SEBI turnover fee + stamp duty
          + GST_rate × (brokerage + NSE transaction/IPFT + SEBI fee)
sell bps = brokerage + slippage + delivery STT + NSE transaction/IPFT
           + SEBI turnover fee
           + GST_rate × (brokerage + NSE transaction/IPFT + SEBI fee)
cost fraction on a trade = absolute position change × side bps / 10,000
```

Official references (rates verified 2026-10-25): [NSE statutory levy table](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies) (STT, stamp duty, SEBI fee, GST); [NSE transaction-charge circular](https://nsearchives.nseindia.com/content/circulars/FA73061.pdf) (₹307/crore/side effective 2026-03-01); [CBIC stock-broking GST FAQ](https://cbic-gst.gov.in/sectoral-faq.html) (brokerage services and pure-agent treatment for STT/stamp recoveries); [SEBI investor FAQ](https://www.sebi.gov.in/sebi_data/faqfiles/sep-2024/1727418208774.pdf) (broker tariff sheet and brokerage variability). These sources do not set the preset brokerage or slippage assumptions.

Reported metrics are net cumulative return, CAGR when elapsed time and ending equity permit it, maximum drawdown (including initial capital), annualized sample volatility (√252), turnover (sum of absolute position changes), and executed order count. Optional benchmark results use common daily price dates and compound strategy returns over each exact common-date interval; dates are not forward-filled. Relative return is `(1 + strategy return) / (1 + benchmark return) − 1`.

`chronological_periods()` defines 60% development, 20% validation, and 20% final evaluation segments in date order. Parameters should be fixed using development and validation only. A final segment is a genuine holdout only if nobody previously examined its dates or outcomes. The NIFTY research dataset and prior experiment periods are retrospective and must not be called untouched. The engine has not established profitability or suitability for investment.

## Machine-learning method

The target is the next observed trading session's close. Features are trailing close, lags, returns, and indicators available at the feature date. Complete rows are ordered chronologically: earliest 70% fits the models, the next 15% is reserved, and final 15% is held out for reported results. The fixed models are not tuned on the validation segment. Naive persistence predicts that the next close equals the latest close. MAE and RMSE report price error in the ticker's exchange currency; R-squared can be negative and is sensitive to the sample. Directional accuracy compares predicted and actual next-session direction relative to the feature-date close. These scores do not represent trading returns.

Future predictions are recursive: model predictions are fed into later lag and indicator values. Future exchange holidays are approximated with business days, and confidence intervals are not estimated. Market regimes, corporate actions, liquidity, news, and other external factors limit all forecasts. There is no guaranteed superior model.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py data_collection.py preprocessing.py exploratory_analysis.py investment_analysis.py technical_indicators.py model_training.py forecasting.py backtesting.py offline_backtest_preview.py tests
```

Tests use synthetic local data, mocked provider responses, and isolated temporary SQLite databases; they do not require a network connection or write mocked candles to the project's default database. `tests/test_dashboard_offline.py` runs the Streamlit app with a fake data service and explicitly fails if production fetching or SQLite persistence is attempted.

For a focused audit run that excludes the service test which writes a temporary SQLite database:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q tests\test_backtesting.py tests\test_backtesting_freshness.py tests\test_offline_backtest_preview.py tests\test_data_quality.py tests\test_exploratory_analysis.py tests\test_investment_analysis.py tests\test_preprocessing.py tests\test_market_data_service.py tests\test_dashboard_offline.py -k 'not successful_fetch_persists_only_the_selected_provider_source'
```

## Troubleshooting

- Empty results can mean an invalid symbol, unsupported dates, no exchange sessions in the range, a Yahoo Finance outage, or throttling; verify the symbol and try refresh later.
- Yahoo Finance does not provide a reliability guarantee or stable API contract. The dashboard displays a concise error instead of a traceback.
- If package installation fails, confirm the environment's Python version and install into its explicit Python executable.
- Some short date ranges do not contain enough history for model evaluation; use at least several months (80 complete feature rows are required).

## Academic and portfolio disclaimer

This repository demonstrates data handling, visualization, and time-series evaluation for learning and portfolio review. Historical patterns do not guarantee future outcomes. The forecasts are experimental and are not recommendations to buy or sell securities.
