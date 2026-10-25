# Indian Stock Market Analysis & Experimental Backtesting

An educational Streamlit application for exploring Indian equity history, technical indicators, investment-risk measures, and simple backtests. Yahoo Finance data may be delayed or revised. This project is not investment advice.

> **Research status:** The machine-learning experiments have not shown reliable improvement over simple baselines. Forecasts and backtests are experimental historical analysis—not trading signals or promises of returns.

## Project overview

- Explore historical data for the saved NIFTY 50 roster or another NSE ticker ending in `.NS`.
- Review price trends, returns, volume, moving averages, RSI, MACD, volatility, and drawdown.
- Compare a stock with an optional benchmark over matching dates.
- Run an opt-in daily backtest for buy-and-hold, no-trade, or a fixed SMA crossover, with illustrative transaction costs.
- Inspect chronological Linear Regression and Random Forest experiments and optional recursive forecasts. These models have not been validated as predictors.

Yahoo Finance is the default provider. Its data may be delayed, revised, unavailable, or rate-limited. The dashboard identifies stale cached data instead of presenting it as a confirmed current quote. It does not place trades or connect to a brokerage account.

## Contents

- [Run the dashboard](#install-and-run-windows-powershell)
- [College demonstration](#college-demonstration)
- [Data-source behavior](#usage)
- [Indicators and risk analysis](#indicators)
- [Backtesting and cost assumptions](#experimental-backtesting-engine)
- [Machine-learning method and limitations](#machine-learning-method)
- [Safe offline tests](#tests)
- [Troubleshooting](#troubleshooting)

## Technologies and architecture

Python 3.11 or 3.12, Pandas, NumPy, Requests, OpenChart, yfinance, Plotly, Streamlit, scikit-learn, and pytest. The dashboard calls reusable modules: provider service → data collection → preprocessing → indicators/EDA → modeling → recursive forecasting. Models are fitted at runtime and are not saved.

## Project structure

```text
app.py                     Streamlit user interface
data_collection.py         Yahoo Finance daily and short intraday downloads
market_data_service.py     OpenChart-first auto route, isolated explicit provider routes, cooldown and last-good retention
market_data_storage.py    Provider-separated SQLite candle persistence
tejhq_provider.py          Explicit keyless NSE daily adapter
smartapi_provider.py       Explicit daily adapter; live HTTP disabled pending header-contract confirmation
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

Use the existing `.venv` if it is healthy. The project targets Python 3.11 or 3.12. Check the environment before the demonstration; do not reinstall requirements just to launch an already prepared environment.

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit --version
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Only for a new environment, use an installed 3.11/3.12 launcher and install the declared dependencies. Preserve an existing environment when creating a replacement:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Press Ctrl+C in the same terminal to stop it. Internet access is needed for either market-data provider. If a managed Windows environment denies Streamlit's default configuration path, use the project-local temporary configuration workaround you already have.

### College demonstration

The dashboard defaults to Yahoo Finance only. When an archived NIFTY 50 dataset is found under the user's TEMP folder, **Local NIFTY 50 CSV archive** is available as a read-only option that does not call a provider or write SQLite. **Automatic provider fallback** can be selected to try OpenChart before Yahoo Finance. Provider modes use network access and the normal SQLite persistence path. Automatic display refresh is off by default; when enabled, the display reruns every five seconds while provider cooldowns still apply. Before presenting, check the selected stock's date range and data-source label. Provider data availability and timing are not guaranteed; OpenChart historical data has known availability issues, and Yahoo Finance may be delayed. The local archive is a fixed historical snapshot, not current live data.

A five-minute walkthrough: (1) select an Indian company from the NIFTY 50 list, choose a daily interval, and confirm that a result loaded; (2) point out the latest observed price/date, adjusted-close choice, provider label, and freshness warnings; (3) show the price chart and existing moving averages, RSI, MACD, returns, and rolling volatility as descriptive historical measures; (4) explain recent return, volatility, drawdown, and optional exact-date benchmark comparison; (5) open **Experimental daily backtest**, choose a strategy and medium delivery-cost preset, choose a date range, and click the run button. Show gross/net returns, costs, freshness, session-coverage warnings, and benchmark provenance if supplied. Explain that the benchmark label is not verified and gaps are not holiday-classified. Do not rank strategies or present results as forecasts or evidence of profitability.

If provider data is unavailable, `offline_backtest_preview.py` demonstrates the report and comparison helpers with deterministic synthetic prices. Run `.\.venv\Scripts\python.exe offline_backtest_preview.py` in a separate terminal and identify it as synthetic; it does not demonstrate the dashboard charts or technical-indicator screens.

## Usage

The dashboard is India-only. Choose one of the 50 NIFTY 50 constituents in the archived official NSE roster snapshot, or enter another NSE ticker ending in `.NS`. The bundled `nifty50_constituents_source.csv` and `.json` provenance record were retrieved from NSE on 2026-10-24; this is a fixed snapshot and is not automatically updated when index membership changes. Each selector entry shows the company name and NSE symbol. **Yahoo Finance** is the default provider and selects Yahoo only; it does not try OpenChart first. **Automatic provider fallback** explicitly tries OpenChart before Yahoo. When a local dataset is present, **Local NIFTY 50 CSV archive** reads each ticker's own CSV without providers or database writes. `NIFTY50_DATASET_DIR` can point to a particular run directory or the parent folder containing `run_*` directories. The local archive is daily only and does not fall back to providers for missing tickers. For provider mode, symbols are mapped to exact NSE equity symbols (for example, `RELIANCE.NS` → `RELIANCE-EQ`) and accepted by OpenChart only when it returns one exact NSE equity match. Choose a preset period or an explicit custom date range (end is exclusive), then inspect the history, indicators, test metrics, and optional forecast. Automatic display refresh is off by default. If enabled, the page reruns every 5 seconds, but each provider is requested at most once per ticker and interval every 30 seconds; manual refresh does not bypass that cooldown.

### Historical Indian intraday bars

The dashboard offers historical `5m` and `15m` intervals for NSE `.NS` equities in provider modes. Yahoo-only mode is the default and requests Yahoo directly; select automatic fallback to try OpenChart first and Yahoo second. Availability varies by symbol and interval. Intraday timestamps are normalized to `Asia/Kolkata` for NSE symbols. OpenChart 0.2.0 parses epoch-millisecond timestamps as UTC and then drops timezone metadata; the adapter restores UTC before converting those timestamps to IST. Yahoo's existing intraday range restrictions apply. OpenChart historical availability remains unresolved: a controlled `RELIANCE.NS` daily request returned HTTP 200 with `status=true` and an empty candle list, so automatic mode may fall back to Yahoo or report no data.

The dashboard shows the provider reported for the displayed dataset, its retrieval time, and the latest observation. For daily bars it shows the calendar date only; a timezone-naive midnight is not presented as the market-close time, and missing timezone information remains explicit. Intraday timestamps and approximate age are shown when available. When providers fail or are cooling down, the service may return its last successful in-memory snapshot marked stale; the displayed retrieval time remains that snapshot's original successful retrieval, not the failed refresh time. The dashboard prominently labels stale snapshots and does not describe them as current or live. Automatic refresh is off by default; if enabled, its five-second page reruns do not bypass the 30-second provider cooldown or guarantee new data. Candle completion and freshness are unverified. OpenChart is an unofficial client for NSE's charting endpoint and is not affiliated with or endorsed by NSE. Yahoo Finance is accessed through the community `yfinance` package and is not affiliated with Yahoo. Provider availability, rate limits, historical coverage, and data-use terms can change; do not redistribute downloaded data. The daily indicators, model evaluation, and forecasts remain available only in daily mode; intraday forecasting has not been enabled.

### Explicit TejHQ daily provider

`MarketDataService.fetch(..., provider="yahoo")` selects Yahoo Finance only and does not call OpenChart. The dashboard uses this explicit Yahoo-only mode by default. `provider="auto"` remains the separate OpenChart-first, Yahoo-fallback mode. TejHQ and SmartAPI remain separate explicit modes and are not added to automatic fallback. Pass `persist=False` for a diagnostic fetch that must skip the service's SQLite writes; the dashboard's provider mode retains normal persistence.

`MarketDataService.fetch(..., provider="tejhq")` selects TejHQ only; it does not fall back to OpenChart or Yahoo Finance. This adapter supports NSE `.NS` symbols and daily (`1d`) candles only. It uses the provider-published keyless `GET /v1/ohlcv/{exchange}/{symbol}` interface with inclusive `from` and `to` ISO dates, a 10-second request timeout, no automatic retries, and no redirect following. The normal `provider="auto"` route remains unchanged and does not include TejHQ.

The provider describes the public endpoint as free, keyless EOD data, sourced from exchange bhavcopy, with an advertised NSE history beginning 2010-01-04. Its published documentation states that refreshes run on trading days after market close and describes rate limits of 100 requests per 10 seconds per IP at the edge and 120 per minute at origin. These are provider statements, not independently verified guarantees; availability, coverage, rate limits, licensing, and freshness can change. The regular OHLCV route supplies unadjusted close; this adapter never renames `Close` to `Adj Close`. It preserves an `adj_close` field only if a response actually supplies it. Empty or malformed responses and invalid candles are errors. Unsorted valid rows are sorted with a warning; duplicates are rejected. One controlled live validation succeeded for `RELIANCE.NS` with 62 daily candles from 2026-07-27 through 2026-10-23 using `persist=False` and no provider fallback or database write. That single symbol/range does not establish general coverage or reliability.

### Angel One SmartAPI status

`MarketDataService.fetch(..., provider="smartapi")` explicitly selects Angel One SmartAPI and suppresses OpenChart, TejHQ, and Yahoo fallback. It supports NSE `.NS` daily candles only; `provider="auto"` remains unchanged and does not include SmartAPI. The provider reads configuration only when called and does not require credentials at import time. **Live HTTP is disabled in `smartapi_provider.py`** until Angel One confirms the endpoint-specific authentication-header contract. Mocked tests temporarily enable the adapter with fake sessions; no live SmartAPI request has been made.

The adapter uses the official `loginByPassword` flow (client code, PIN, current six-digit TOTP) and `POST /rest/secure/angelbroking/historical/v1/getCandleData`. The endpoint-specific [SmartAPI guide](https://smartapi.angelone.in/docs/MarketData) shows login with `Content-Type`, `Accept`, `X-UserType`, `X-SourceID`, local/public IP, MAC, and `X-PrivateKey`; the official [Python SDK](https://github.com/angel-one/smartapi-python/blob/main/SmartApi/smartConnect.py) constructs the same common header set and adds `Authorization: Bearer <JWT>` after login for authenticated requests. Its historical/logout methods use the shared request builder, which does not add `x-api-key`, `x-client-code`, or `x-feed-token`. The endpoint guide's historical and logout examples likewise show the common headers plus bearer JWT.

There is still a material conflict: the general [SmartAPI User guide](https://smartapi.angelone.in/docs/User) separately marks `Authorization`, `x-api-key`, `x-client-code`, and `x-feed-token` mandatory in its authentication-header table. The endpoint examples and SDK conflict with that table about the three additional headers, and the SDK is implementation evidence rather than a guarantee of the current server contract. The adapter retains its pre-existing combined headers on secured requests and remains **live-disabled** pending explicit Angel One confirmation. Do not infer that sending the union is correct. Offline tests verify the documented common headers and JWT stage transition only; they do not validate the disputed server contract.

| Stage | Method and endpoint | Endpoint-specific documented headers | Authentication value available then | Evidence / status |
|---|---|---|---|---|
| Login | `POST /rest/auth/angelbroking/user/v1/loginByPassword` | JSON content/accept, `X-PrivateKey`, `X-UserType`, `X-SourceID`, client local/public IP, MAC | No session token yet; client code, PIN, and current TOTP are in the JSON body | Login examples in the endpoint guide and SDK common-header builder agree. |
| Historical candles | `POST /rest/secure/angelbroking/historical/v1/getCandleData` | Same common headers plus `Authorization: Bearer <JWT>` in endpoint examples and SDK behavior | JWT from successful login; feed token is also returned but examples/SDK do not use it for this call | General auth table additionally requires `x-api-key`, `x-client-code`, `x-feed-token`; unresolved conflict prevents live validation. |
| Logout | `POST /rest/secure/angelbroking/user/v1/logout` | Same common headers plus bearer JWT in endpoint example and SDK shared request construction | JWT from successful login; client code in JSON body | General auth table's additional-header requirement conflicts with examples/SDK; exact server contract unconfirmed. |

HTTP methods and paths are documented in the [SmartAPI User guide](https://smartapi.angelone.in/docs/User); candle request details are in the [Historical API guide](https://smartapi.angelone.in/docs/MarketData). The table distinguishes endpoint examples from the generic authentication table; it does not claim independent authentication of the SDK or guarantee server behavior.

The daily request uses exchange `NSE`, an exact `symboltoken` resolved from Angel One's published instrument master (`symbol` such as `RELIANCE-EQ`, `name`, `token`, `exch_seg=nse_cm`), interval `ONE_DAY`, and Asia/Kolkata date bounds formatted `yyyy-MM-dd hh:mm`. The documented maximum is 2,000 days per request. The documented candle schema is timestamp plus OHLCV; this adapter never fabricates `Adj Close`, so only unadjusted `Close` is available. It rejects malformed, duplicate-date, non-finite, non-positive, and invalid-OHLC records. Each HTTP request uses a 10-second timeout, disables redirects, and does not retry. HTTP 401 is categorized as authentication rejection, HTTP 429 as rate limiting, and HTTP 403 as an ambiguous forbidden response because SmartAPI documents 403 for rate-limit denial as well as access rejection. The documented historical endpoint limit is 3 requests/second, 180/minute, and 5,000/hour per client code ([official rate-limit table](https://smartapi.angelone.in/docs/RateLimit)); actual account/provider throttling may be stricter. The service does not currently enforce a shared account-wide limit across symbols; do not use it for batch requests. Official documentation inspected here does not establish costs or market-data redistribution/licensing terms; confirm applicable SmartAPI/account terms before use. Instrument master is documented as refreshed daily; this adapter downloads it per call and does not persist/cache it.

Set these environment variables before an explicit SmartAPI call; never put their values in source files or logs: `ANGELONE_SMARTAPI_KEY`, `ANGELONE_CLIENT_CODE`, `ANGELONE_PIN`, `ANGELONE_TOTP` (the current six-digit code, not the long-lived authenticator secret), `ANGELONE_CLIENT_LOCAL_IP`, `ANGELONE_CLIENT_PUBLIC_IP`, and `ANGELONE_CLIENT_MAC_ADDRESS`. Missing values fail before HTTP calls. After valid login tokens are received, the adapter attempts the documented logout POST with the client code in its body and authenticated headers on both success and failure. It does not log or retain tokens beyond the call. A logout failure after a successful candle fetch is returned as a warning; during an existing fetch failure it does not replace the original error. The logout route is documented, but its exact required header combination remains subject to the ambiguity above. `persist=False` skips the service's candle-storage path, but the adapter still makes the required login, instrument-master, historical, and logout HTTP requests. Use `provider="smartapi", persist=False` for a provider-only non-persisting request. The shared service may return an explicitly stale prior snapshot from the same selected provider during cooldown or after failure; it cannot substitute another provider for this explicit selection. The documented limits are per client code; the project does not implement account-wide rate limiting, so do not use this provider for batch requests until that safeguard is designed. No exchange calendar is implemented, and mocked offline tests do not establish live account eligibility, historical availability, or general provider reliability.

## Local candle storage

Successful provider downloads are stored in `data/market_data.sqlite3`, created automatically the first time a valid dataset is saved. This uses Python's built-in SQLite library and adds no package dependency. The database is local and excluded from version control by the existing `data/*` ignore rule. Back it up by stopping the dashboard and copying `data/market_data.sqlite3` to a separate safe location; never place the only backup inside the project folder. SQLite journals may exist briefly while the application is writing, so copy after stopping the app for a consistent simple file backup.

Stored keys include ticker, interval, UTC timestamp, and exact provider/source. OpenChart and Yahoo Finance candles remain separate. Saving a repeated candle updates that source's prior row deterministically (last write wins). `MarketDataService.fetch()` persists successful results by default. Diagnostic callers can explicitly pass `persist=False` to skip `save_candles()` and this service's SQLite writes. Provider-only diagnostic requests can explicitly select `provider="yahoo"`, `provider="openchart"`, `provider="tejhq"`, or `provider="smartapi"` together with `persist=False`; OpenChart, TejHQ, and SmartAPI require NSE `.NS` symbols, and none of these modes silently falls back to another provider. The SmartAPI adapter does not invoke yfinance or use a SQLite-backed cache. The dashboard defaults to explicit Yahoo-only mode; automatic OpenChart-first/Yahoo-fallback mode remains an opt-in choice. The service does not merge stored rows into the dashboard result or substitute stored rows after failure. The existing in-memory cooldown and last-successful snapshot remain distinct: a stored candle has no new retrieval time, and loading stored history does not label it as freshly fetched. The storage module offers source-specific range and full-history queries for later offline analysis.

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

Provider timestamps that cannot be parsed are excluded with a quality warning; duplicate timestamps and out-of-order input are reported before preprocessing sorts the data and retains the last duplicate. Missing numeric values remain gaps. Quality checks report missing OHLCV columns/values, nonnumeric or non-finite values, nonpositive OHLC/adjusted prices, negative volume, and impossible High/Low relationships. The dashboard stops selected-price summaries when supplied values are nonnumeric, non-finite, zero, or negative, and stops summaries if valid prices produce non-finite returns. Invalid latest raw closes are shown as unavailable; incomplete or invalid volume aggregates are unavailable and invalid volume bars are omitted. Other field inconsistencies remain visible as quality warnings, and the SQLite storage layer rejects quality-flagged rows if persistence is attempted.

Freshness uses the latest valid selected-price observation, converts its timestamp to Asia/Kolkata for the reported observation date and age, and separately normalizes retrieval metadata to that timezone. Provider fallback/cooldown snapshots are marked stale; old retrieval metadata, old observations, future observations, and future retrieval timestamps are warned about. The observation-age threshold is calendar days, so weekends and holidays can contribute to it.

No exchange calendar is bundled: the app cannot distinguish a missing trading session from a market holiday or provider omission, and it does not infer missing dates from calendar-day gaps. Daily data gaps are not forward-filled. A 20% one-session move in recent daily data triggers a review warning only; it is not rejected because splits, dividends, and legitimate market moves can cause large changes. Yahoo responses may still be displayed with quality warnings for non-selected OHLCV fields; invalid selected prices fail the dashboard summary. Intraday data freshness and bar completion remain unverified.

The model panel identifies the selected price basis, training period, chronological split, and recursive horizon. Historical NIFTY 50 research has not established a reliable predictive edge; forecasts remain experimental and are not approved investment signals.

## Experimental backtesting engine

`backtesting.py` is an offline, independently testable engine exposed through an opt-in dashboard panel. It requires `Adj Close` by default; a caller can explicitly select another price column. Buy-and-hold and no-trade baselines and a long/flat simple-moving-average crossover are available. SMA crossover requires enough observations for its configured slow window (50 sessions by default). Daily signals use data through that session's close, execute at the next observed session's close, and begin earning close-to-close returns only after that assumed fill. This deliberately delays signal returns by one more close-to-close interval. Adjusted close is only a proxy execution price because adjusted OHLC fill prices are not generally supplied.

`build_backtest_report(result, periods=...)` provides presentation-ready fields for cost mode/preset, gross and net cumulative return, benchmark dates and provenance status, chronological segment descriptions, and limitations. Benchmark and strategy inputs must have exactly matching normalized daily dates; absent observations or NaN prices are rejected, with no filling or interpolation. A caller-supplied `benchmark_source` or series name is only a label: reports mark it `caller_asserted`, set `source_independently_verified=False`, and retain a fingerprint that identifies the supplied aligned values but does not authenticate them. Unlabeled data is marked `unknown`. No current API independently verifies benchmark origin. For a Series, `price_column` is the assumed price basis; a recognized conflicting Series name or an explicit conflicting `benchmark_price_basis` is rejected. DataFrame inputs select the requested price column. `run_chronological_evaluation()` returns development, validation, and final-evaluation summaries with gross and net returns. Prior-examined periods are retrospective; the final segment is an untouched holdout only if nobody previously examined its dates or outcomes.

The daily results frame uses the selected price-basis name (`Adj Close` or `Close`). Metric summarization rejects missing/non-finite returns, invalid costs/turnover, duplicate dates, and non-chronological rows. CAGR is unavailable for zero-duration periods or when annualizing would overflow numeric range; it is not emitted as infinity. Annualized volatility is unavailable for one observation or if the result cannot be represented finitely.

### Dashboard backtest panel

The **Experimental daily backtest** panel appears for NSE `.NS` tickers on the daily (`1d`) workflow. It reuses the data result already loaded by the dashboard and runs calculations only after **Run selected backtest** is clicked. The form-submit rerun reuses the matching session-local data result; it does not call the market-data service again. Normal dashboard refresh/periodic data behavior is unchanged. This panel supports buy-and-hold, no-trade, and the fixed 20/50-session SMA crossover; it does not expose parameter tuning or strategy ranking. The user chooses an inclusive evaluation date range, `Adj Close`/`Close`, an illustrative low/medium/high NSE delivery-cost preset, or a separate `ZERO COST / FRICTIONLESS` mode.

Backtest reports include a conservative session-coverage screen. It excludes weekend dates from the weekday-gap count and lists absent Monday-Friday dates as **unclassified weekday gaps**; it does not label them missing NSE sessions because no verified local NSE holiday calendar is available. Weekend dates are listed separately, including any observed weekend dates, because special sessions are not modeled. Coverage checks do not fill prices or add candles, and even a range with no weekday gaps is not certified complete. Freshness age remains a separate calendar-time policy and is not adjusted for trading sessions.

The existing optional benchmark CSV upload can be included in the backtest. Its filename is a caller-provided label, not verified provenance. The selected price basis and every normalized daily observation date in the evaluation slice must match exactly; missing/extra dates and invalid values stop the benchmark comparison. Benchmark freshness and authenticity are not independently verified. The backtest report shows evaluation dates, price basis, reported provider label, latest observation/retrieval/evaluation timestamps, freshness status, gross/net return, available CAGR/drawdown/volatility, turnover, orders, costs, benchmark provenance status, and limitations. The report is historical analysis, not a forecast, recommendation, or profitability claim.

Daily inputs use the validated backtest policy: unknown source/timestamps, future timestamps, invalid prices, and intraday data are not overrideable. Data older than the daily freshness limits or marked stale is rejected unless **Allow stale data for diagnostic/historical analysis** is explicitly checked; the report then shows `stale_override` and its warnings. This permits retrospective slices without calling them fresh. A stale override does not authenticate provider or benchmark provenance. The weekday screen cannot classify a gap as an NSE holiday or omitted session, including dates missing from both supplied series; execution assumptions remain based on observed rows.

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

The safe offline suite below uses synthetic data and mocked providers. `tests/conftest.py` blocks non-loopback sockets suite-wide; dashboard AppTest may use loopback internally. It excludes the SQLite-storage suite and the service persistence test. It does not access the project database or `data/`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q tests\test_backtesting.py tests\test_backtesting_freshness.py tests\test_data_collection.py tests\test_data_quality.py tests\test_dashboard_offline.py tests\test_exploratory_analysis.py tests\test_forecasting.py tests\test_investment_analysis.py tests\test_market_data_service.py tests\test_model_training.py tests\test_nifty50_downloader.py tests\test_nse_trading_calendar.py tests\test_offline_backtest_preview.py tests\test_preprocessing.py tests\test_smartapi_provider.py tests\test_technical_indicators.py tests\test_tejhq_provider.py -k 'not successful_fetch_persists_only_the_selected_provider_source'
```

This omits `tests/test_market_data_storage.py` because those tests create temporary SQLite databases, and deselects `successful_fetch_persists_only_the_selected_provider_source` for the same persistence-related reason. The standard `pytest -q` command includes those tests and is not the safe no-database command above.

## Troubleshooting

- Empty results can mean an invalid symbol, unsupported dates, no exchange sessions in the range, a Yahoo Finance outage, or throttling; verify the symbol and try refresh later.
- Yahoo Finance does not provide a reliability guarantee or stable API contract. The dashboard displays a concise error instead of a traceback.
- If package installation fails, confirm the environment's Python version and install into its explicit Python executable.
- Some short date ranges do not contain enough history for model evaluation; use at least several months (80 complete feature rows are required).

## Academic and portfolio disclaimer

This repository demonstrates data handling, visualization, and time-series evaluation for learning and portfolio review. Historical patterns do not guarantee future outcomes. The forecasts are experimental and are not recommendations to buy or sell securities.
