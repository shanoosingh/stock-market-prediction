# Investing Decision Tool Roadmap

Updated: 2026-10-25

## Phase 1 completed

- Added offline investment-analysis helpers in `investment_analysis.py` for recent observed-session returns, 20-session annualized historical volatility, selected-period maximum drawdown, volume outlier warnings, data freshness, and exact-date benchmark comparison.
- Daily dashboard analysis defaults to `Adj Close` when present, while allowing an explicit switch to traded `Close`. Price basis is shown and described.
- Dashboard surfaces latest valid price date, observation age, retrieval freshness, stale cached snapshots, missing/non-numeric prices, and other existing data-quality warnings. Missing data is not filled.
- Added optional benchmark CSV upload. It requires `Date` and the selected price column, compares the common dates only, and makes no market-data request.
- Explained existing indicators and risk formulas in the UI. Forecast labels show the model, horizon, price basis, and training end, and disclose that the NIFTY research has not shown a reliable predictive edge.
- Added focused offline tests for formulas, missing/invalid/insufficient data, freshness, date alignment, benchmark basis, and warnings.

## Current evidence and limitations

- No candidate model has met the evidence threshold for dashboard integration. The displayed models remain experimental comparisons, not recommendations or evidence of profitability.
- Existing holdout and walk-forward analyses are retrospective; previously inspected observations are not untouched test data. See `ML_RESEARCH_INDEX.md` and `FINAL_HANDOFF.md`.
- The app still retrieves and stores data through its existing provider service when run. This phase added no new fetch path, database write, or live benchmark request. OpenChart historical retrieval has returned empty candle lists in prior controlled diagnostics; Yahoo fallback and source/freshness warnings remain relevant.
- Benchmark comparison depends on the user supplying an appropriate benchmark CSV. Its provenance and freshness cannot currently be verified by the app. Matching dates are exact daily calendar dates; no holiday filling occurs.
- Phase 2 now includes an automated Streamlit `AppTest` harness with a fake service. It verifies the tested dashboard paths without provider access or SQLite writes; it does not exercise every sidebar interaction or upload widget.

## Outstanding reliability work

1. Add durable dependency locking and verify the documented Python support versions; this phase installed no packages.
2. Consider a separately reviewed, point-in-time benchmark/constituent data source and provenance workflow before automating benchmark acquisition.
3. Extend the AppTest harness to cover benchmark CSV upload rendering and additional sidebar selections, still using an isolated fixture.
4. Revisit forecast presentation only after new, independent data or a pre-specified evaluation produces repeatable evidence. Do not promote a candidate on the current retrospective dataset.

## Next milestone

Review the backtest execution timing and Indian-equity cost scenarios before evaluating real historical data. The previously used dataset is retrospective, so a final holdout claim requires a genuinely new, untouched period. Then address dependency reproducibility and benchmark provenance. Keep model experimentation paused unless a new hypothesis and independent evaluation period are available.

## Phase 2 progress (2026-10-25)

- Added `tests/test_dashboard_offline.py`, which exercises the actual Streamlit app through `AppTest` with synthetic data returned by a fake service. Provider fetch and SQLite persistence entry points fail the test if called. Coverage includes adjusted-close default selection, overview/performance/risk panels, quality warnings, stale snapshots, and insufficient model history.
- Existing `investment_analysis.py` helpers have offline tests for missing prices, stale retrieval, insufficient volatility history, and benchmark alignment on exact common dates; no benchmark data is fetched.
- Added `backtesting.py` as a standalone long/flat engine with buy-and-hold, no-trade, and SMA crossover strategies. It uses adjusted daily prices, next-observed-close execution after a signal, a further close-to-close holding delay, configurable transaction cost/slippage assumptions, benchmark alignment, and descriptive performance metrics.
- Added synthetic tests for signal timing, cost application, trade count/turnover, drawdown, volatility, CAGR, chronology, invalid/missing prices, and mismatched benchmark dates.
- Backtesting quality audit added hand-calculated entry/exit cost and slippage checks, exact CAGR and single-observation metric cases, delayed signal-return assertions, explicit long/flat checks, and finite-cost/return validation.
- The engine remains disconnected from the dashboard. No historical strategy evaluation was run: the existing dataset's relevant periods were already examined, so they cannot support an untouched final holdout claim. Backtest metrics on synthetic tests verify calculations only, not strategy performance.
- Reliability caveats confirmed during audit: zero cost/slippage remains available for controlled tests and is now labeled `ZERO COST / FRICTIONLESS`; a missing row represented by NaN is rejected, but an absent calendar date is treated as the next observed session because no exchange calendar is supplied; current vendor `Adj Close` may be revised backward for later corporate actions, affecting crossover signals near those events.

## Indian delivery-cost scenarios (2026-10-25)

- Added reusable `low`, `medium`, and `high` presets in `backtesting.py`, with side-specific brokerage, slippage, statutory charge components, and explicit buy/sell cost rates. Existing symmetric `transaction_cost_bps` and `slippage_bps` API remains supported. Combining legacy values with a named/custom scenario is rejected to prevent double charging.
- Official values verified from NSE/CBIC material: equity-delivery STT 10 bps on both sides; stamp duty 1.5 bps on buys; SEBI turnover charge 0.01 bps on both sides; NSE cash market transaction charge plus IPFT contribution ₹307/crore/side (0.307 bps, effective 2026-03-01); GST on stock-broking services 18%.
- The presets use these illustrative per-side brokerage/slippage assumptions (not official rates): low 0/1 bps, medium 2/5 bps, high 5/15 bps. Modeled resulting costs: low buy/sell 12.87406/11.37406 bps; medium 19.23406/17.73406 bps; high 32.77406/31.27406 bps.
- GST is modeled on brokerage + exchange + SEBI fee components, excluding STT/stamp duty under assumed pure-agent treatment. This base is a simplification and must be checked against the specific contract note. Presets are NSE-specific; brokerage, rounding, minimums/caps, exchange rates, DP charges, and slippage vary by account, broker, order size, exchange, and current rules. They are not personal cost quotes.
- Official source links and effective dates are documented in the README's backtesting section. NSE's statutory-levy page was updated 2026-04-17; the cash-market circular is effective 2026-03-01. Brokerage/slippage assumptions are not sourced as official rates.
- No prices were fetched and no strategy was run on previous NIFTY results. These presets only change the backtest cost calculation and do not alter dashboard behavior.
- Arithmetic reconciliation follow-up: all six stored low/medium/high buy/sell totals reconcile to their individual component rates. The exchange conversion is `₹307 / ₹10,000,000 × 10,000 = 0.307 bps`. Focused tests now independently calculate every preset and both sides from components. README clarifies that excluding STT/stamp duty from the GST base is conditional on qualifying pure-agent treatment and must be checked against the broker contract note. No preset values needed correction.

## Backtest reporting transparency

- Added `build_backtest_report()` as a pure reporting helper. It surfaces cost mode and preset rates, gross/net cumulative returns, aligned benchmark dates and source, chronological period interpretation, and simulation limitations. `summarize_returns()` now provides gross and net cumulative returns for full and subperiod summaries.
- Benchmark reporting records the first/last common return interval and source label. Missing benchmark dates remain unfilled. Callers should supply `benchmark_source`; an absent source is explicitly marked unspecified unless the input series has a name.
- Development, validation, and final evaluation segments are labeled chronologically. The helper warns that the final segment is only untouched if its dates/outcomes were not previously examined.
- Dashboard remains unchanged: the standalone engine is not currently part of its workflow, and adding strategy controls would alter production behavior. README documents the reporting helper and this boundary.
- Offline tests cover scenario/frictionless labels, gross/net comparison, benchmark dates/source, chronology interpretation, and caveats. No market data or historical strategy run was used.

## Offline backtest report preview

- Added `offline_backtest_preview.py`, a standalone deterministic demonstration using generated price and benchmark fixtures. It prints frictionless and medium-cost examples with gross/net returns, selected cost mode, exact-date benchmark comparison and provenance, chronological periods, and limitations.
- Every preview is labeled synthetic demonstration data, not real market data or investment advice. The module does not import the dashboard, provider service, or storage layer and does not save report files.
- Added tests that guard against provider/network/SQLite access and check repeatable report content and both cost modes. Production dashboard remains unchanged.
- Run with `\.venv\Scripts\python.exe offline_backtest_preview.py` from the project root. This is separate from `streamlit run app.py` and demonstrates formatting only, not strategy quality.
- Verification: `pytest -p no:cacheprovider -q tests\\test_offline_backtest_preview.py tests\\test_backtesting.py tests\\test_dashboard_offline.py` via the project `.venv`: **24 passed in 3.63s**. Running the preview produced both expected modes, identified the synthetic benchmark source, and showed 19 matched return intervals.

## Fair strategy-report comparison

- Added `compare_backtest_reports()` for two or more named reports over caller-specified exact start/end dates. It preserves caller ordering and does not select or rank a strategy.
- Comparisons require matching benchmark presence, source, matched interval dates/count, and benchmark return. Cost assumptions are checked; mismatches produce `comparable=False` and a warning, especially for zero-cost versus costed results. Metric omissions are listed per report.
- The offline preview now compares synthetic moving-average crossover and buy-and-hold reports on the same generated dates, benchmark, and medium-cost preset, alongside its separate frictionless and costed examples.
- README documents the interface and limitations. No dashboard, network, storage, historical data, or previous experiment output was changed.
- Verification: `pytest -p no:cacheprovider -q tests\\test_backtesting.py tests\\test_offline_backtest_preview.py tests\\test_dashboard_offline.py` via the project `.venv`: **28 passed in 3.55s**. The preview command reported a common period of 2025-01-06 through 2025-02-28, two strategies, compatible assumptions, and no automatic ranking.

## Market-data quality/freshness audit

- Implemented checks include invalid timestamp exclusion warnings, duplicate/out-of-order reports, retained missing-value gaps, OHLC relationships, invalid/non-finite numeric values, negative volume, and provider-specific stale snapshot reporting. Storage validation rejects data with recorded quality issues; it does not create missing candles.
- Added nonpositive checks for OHLC and adjusted-close fields. The dashboard summary now safely fails for invalid selected prices rather than presenting zero, negative, infinite, or nonnumeric values as a valid latest price. Other field-quality issues are surfaced as warnings.
- Freshness age uses the latest valid observation and Asia/Kolkata conversion; retrieval age is separate. Future retrieval timestamps now get a distinct status/warning. Old-observation age is in calendar days.
- There is no exchange calendar, so no missing calendar dates are classified as holidays or missing sessions. The 20% one-session recent-move detector remains warning-only; it cannot distinguish corporate actions from genuine jumps automatically. Intraday freshness/bar completion and Yahoo non-selected OHLC anomalies remain limitations.
- Safe offline verification excluding the SQLite persistence test: **48 passed, 1 deselected in 3.72s** across data quality, summary analysis, investment analysis, preprocessing, mocked market-data service, and dashboard tests. No market data was fetched and no project SQLite database was used.
- Process note: an earlier test invocation in this audit inadvertently included the service persistence test, which writes to its pytest temporary SQLite database. It did not target the project's database. The final verification explicitly deselected that test; no further SQLite-writing test was run.

## Backtesting integration-readiness audit

- **Pass:** Comparison requires exact requested evaluation dates, price basis, full observation-date fingerprint/count, aligned benchmark intervals, source labels, aggregate benchmark return, and a fingerprint of the actual aligned benchmark price series. Cost signatures include mode and buy/sell bps; mismatches are visibly non-comparable and are never ranked.
- **Pass:** Metric edge coverage includes two-row minimum for engine input, one-row summary with unavailable CAGR/volatility, constant prices, missing/invalid price rejection, and safe gross/net, drawdown, turnover, and order counts.
- **Pass:** Daily inputs normalize timezone-aware timestamps to Asia/Kolkata dates, reject duplicate dates after normalization, and sort chronologically. Period summaries omit pre-boundary returns but retain boundary execution costs; benchmark-relative returns likewise include boundary costs. Synthetic tests cover these boundaries.
- **Pass:** Preview tests block provider imports, socket connects, and SQLite connection calls while generating deterministic fixture reports. No dashboard or live data service is imported by the preview.
- **Pass:** Zero-cost labels and scenario/legacy per-side cost rates are included in reports and comparison rows. README's compilation command now includes the backtest and preview modules; the full suite's temporary SQLite use is explicitly documented, with a safe focused no-database command added.
- **Gap:** The dashboard clearly marks stale snapshots but continues showing analysis/forecasts from them; the standalone backtest API receives bare price data and cannot independently validate freshness or provenance. Decide whether production integration should block stale data or require explicit acknowledgement, and pass freshness/source metadata into any future backtest interface.
- **Unverified:** Exchange-session completeness cannot be assessed without a trusted exchange calendar. Benchmark identity fingerprints series contents, but source labels remain caller-supplied and are not independently authenticated.
- Offline verification (the documented focused command): **81 passed, 1 deselected in 3.84s**. The deselected service persistence test was not run. No price downloads, historical NIFTY tests, SQLite writes, dashboard changes, or package installs occurred during this audit.

## Backtest input freshness policy

- Added `BacktestInputMetadata` and `assess_backtest_input_freshness()` as a standalone layer; no dashboard wiring or provider retrieval was added.
- Daily defaults: latest observation no older than **7 calendar days**; retrieval no older than **24 elapsed hours**. Naive timestamps mean Asia/Kolkata; aware timestamps convert to Asia/Kolkata. Future observations/retrievals and explicit stale/old/failed/cooldown provider states fail policy.
- `run_validated_backtest()` requires known source/timestamps, daily policy support, and metadata latest observation date matching the actual final price date. Stale data is rejected unless the caller explicitly passes `allow_stale_override=True`; resulting reports carry `stale_override` and warnings. Rejections expose an inspectable `.assessment` with `stale_rejected` or `unknown` status.
- Low-level `run_backtest()` preserves calculation compatibility: absent metadata yields an `unknown` report, never a fresh claim; supplied stale metadata still rejects by default. Synthetic tests and preview now supply explicit synthetic metadata.
- Intraday inputs are unknown under this policy. Daily age limits are not applied, and production-style validation rejects them until a separate bar-completion/freshness policy exists.
- Limitations: metadata is caller-supplied and cannot independently authenticate a provider or guarantee that the observation timestamp is truthful. The freshness layer is not connected to the dashboard. Age limits use calendar days, so weekends/holidays count toward the daily observation age.
- Verification: focused offline suite including freshness, backtesting, preview, data-quality, service mocks, and dashboard fixtures: **91 passed, 1 deselected in 4.03s**. The deselected test writes SQLite; it was not run. No provider calls or database writes were made.

### Phase 2 verification and remaining work

- Verification command used the project `.venv` and did not launch the provider service or write SQLite:

  ```powershell
  $env:PYTHONDONTWRITEBYTECODE='1'
  & '.\.venv\Scripts\python.exe' -B -m pytest -p no:cacheprovider -q tests\test_data_collection.py tests\test_data_quality.py tests\test_exploratory_analysis.py tests\test_forecasting.py tests\test_market_data_service.py tests\test_model_training.py tests\test_nifty50_downloader.py tests\test_preprocessing.py tests\test_technical_indicators.py tests\test_investment_analysis.py tests\test_backtesting.py tests\test_dashboard_offline.py -k 'not successful_fetch_persists_only_the_selected_provider_source'
  ```

  Result: **90 passed, 1 deselected in 4.10s**. The deselected service-persistence test and `tests/test_market_data_storage.py` are excluded because they write SQLite databases.
- No UI production changes or live-data workflows were added in Phase 2.
- Historical next-milestone recommendation at the time of the cost-scenario review: add fixed-per-order brokerage minimums/caps and alternative-exchange profiles before using scenarios to assess real strategy performance. The current bps brokerage and NSE profile remain illustrative approximations. Any later evaluation must use a genuinely new, untouched chronological period. The later opt-in diagnostic dashboard integration is documented below.

## Benchmark provenance and session alignment (2026-10-25)

- Updated backtest reports to separate a benchmark label from provenance: explicit labels and descriptive Series names are `caller_asserted`; missing labels are `unknown`; all reports set `source_independently_verified=False`. The content fingerprint identifies the compared values and dates, not who supplied them.
- Benchmark and strategy series must now have exactly matching normalized dates. NaN/missing prices, date omissions, and extra dates are rejected; no forward filling or interpolation occurs. Reports include the selected benchmark price basis, how that basis was established, matched/input counts, and unmatched-date counts. A conflicting recognized Series basis or explicit `benchmark_price_basis` is rejected.
- Local-only calendar investigation found no repository calendar/holiday files and no installed `exchange_calendars`, `pandas_market_calendars`, or `holidays` package in `.venv`. The project still cannot identify whether a date absent from both supplied series was an NSE holiday or an omitted session; it makes no holiday classifications.
- Offline tests cover unknown and caller-asserted labels (including official-sounding but unverified text), exact date alignment, missing/NaN benchmark observations, basis mismatch, and report provenance fields.
- At the time of this readiness audit, the backtesting engine remained separate from the dashboard. Benchmark source authentication and trusted exchange-session completeness remain unimplemented; the later limited integration below does not resolve those gaps.
- Verification command: `$env:PYTHONDONTWRITEBYTECODE='1'; .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q tests\test_backtesting.py tests\test_backtesting_freshness.py tests\test_offline_backtest_preview.py` — **45 passed in 1.05s**. These are offline tests; no SQLite or provider tests were run.

## Opt-in experimental daily backtest panel (2026-10-25)

- Added an opt-in daily NSE `.NS` panel to `app.py`. It offers buy-and-hold, no-trade, and fixed 20/50-session SMA crossover; a selected inclusive date range; the selected adjusted/raw close basis; low/medium/high Indian delivery cost presets; a separate frictionless option; optional uploaded benchmark; and an unchecked stale diagnostic override. It does not expose strategy tuning or rank results.
- The panel reuses the dashboard's already-loaded `MarketDataResult`. A session-local keyed cache supplies that result to the form-submit rerun, avoiding another `MarketDataService.fetch()` call on submit. The normal dashboard data flow and forecast path continue unchanged. Calculations, date/price validation, freshness policy, and report construction remain in `backtesting.py`.
- Metadata is passed from the actual reported provider label, retrieval timestamp, selected evaluation slice's latest valid observation, one captured Asia/Kolkata evaluation timestamp, daily interval, and existing freshness/service/data-quality warnings. Missing metadata is not manufactured. Stale history is rejected by default; the explicit override is retained in the report. Unknown provenance, invalid prices, and intraday data remain blocked. Future timestamps cannot be overridden, and validated intraday input remains unknown/rejected even when the override is selected.
- Hardened the SMA engine to reject fewer than `slow_window` valid daily observations (50 by default), rather than silently returning a no-signal backtest on too little history.
- Offline AppTest coverage blocks external sockets, OpenChart/Yahoo provider functions, `save_candles`, and SQLite connections. It verifies opt-in behavior, no extra service call on form submit, source/retrieval metadata, cost/report output, stale rejection and override visibility, unknown metadata rejection, intraday exclusion, and invalid-price rejection. Existing dashboard forecast/analysis regression assertions remain in the same suite.
- Limitations remain: benchmark labels and contents are caller-provided, benchmark freshness is not authenticated, no exchange calendar can classify dates absent from both inputs, adjusted close is an execution proxy, and costs/execution are illustrative. Dashboard integration is limited to descriptive research; it is not predictive or trading advice.
- Verification command: `$env:PYTHONDONTWRITEBYTECODE='1'; .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q tests\test_dashboard_offline.py tests\test_backtesting.py tests\test_backtesting_freshness.py tests\test_offline_backtest_preview.py` — **59 passed in 7.40s**. No provider or SQLite-writing tests were run.
