# Final stock-market ML project handoff

**Audit date:** 2026-10-24. **Disposition:** Research paused on the current NIFTY 50 dataset; no new forecast integrated.

## Scope and inspected files

Inspected project structure and README, `app.py`, `model_training.py`, `forecasting.py`, `technical_indicators.py`, `data_collection.py`, `market_data_service.py`, `market_data_storage.py`, `nifty50_downloader.py`, `requirements.txt`, existing tests, NIFTY manifest/source roster metadata, experiment progress reports, result summaries, prediction files, and walk-forward output. No `AGENTS.md` instruction file was present at the project root or parent checked.

Research index: [ML_RESEARCH_INDEX.md](ML_RESEARCH_INDEX.md).

## Findings

- The dataset is an external Temp run with a saved 50-row official NSE roster and 50 successful CSV histories (first-history dates range from 2021-10-25 to 2023-08-21; all end 2026-10-23). TRENT.NS and TMPV.NS were excluded from ML comparisons because of audited structural/unexplained adjusted-price discontinuities. The dataset reflects current constituents, so five-year current-roster results have survivorship/constituent-selection limitations.
- Original price-level test targets cover 2026-02-03 through 2026-10-23 (47 stocks start Feb 3; JIOFIN.NS starts May 15). Results: mean MAE/RMSE were Naive `30.402413 / 40.908094`, Linear Regression `30.933280 / 41.469638` (6/48 stocks improved MAE), Random Forest `192.701460 / 220.710308` (0/48).
- Return test results: Zero Return `30.402413 / 40.908094`, return Linear Regression `31.182598 / 41.776692` (4/48 improved MAE), return Random Forest `41.731686 / 52.262664` (0/48).
- Scale-normalized analysis, fixed rolling-mean test, and 15-window walk-forward all failed to establish a repeatable improvement. Full formulas and results are recorded in the research index and linked experiment outputs.
- No candidate has met the evidence threshold for dashboard integration. No new model was trained in this final audit.

## Dashboard accuracy/performance wording review

Current UI text is generally appropriately cautious: it labels results experimental, describes the chronological split, says R-squared is not investment return, and disclaims advice/profitability. No user-facing text or behavior was changed.

Concerns to address only through a separately approved UI/metric change:

1. The displayed “Directional accuracy” table includes Naive Persistence. It predicts no direction; under the current equality-of-signs formula, neutral predictions are counted wrong on every nonzero move. That naturally produces a near-zero baseline score while nonzero ML calls may score near 50%, which can overstate the comparative impression. A neutral baseline should be labeled as no directional call, with direction-call coverage and a training-only directional baseline shown separately.
2. The adjusted-price checkbox defaults off. The dashboard therefore evaluates raw `Close` unless the user opts in, while all NIFTY research used `Adj Close`; corporate actions can make those results incomparable.
3. Future recursive forecasts use models fitted only on the earliest 70% of the selected history. The UI displays the fit date, but users may overlook that the model is not refit on the validation/test or latest observations before forecasting. Current behavior is disclosed in the fit-date caption; do not silently change it.

## Reproducibility and reliability gaps (prioritized)

1. **Environment mismatch:** README specifies Python 3.11/3.12, but `.venv` used for this audit is Python 3.14.2. Relevant tests pass in this environment, but the documented support matrix was not verified here.
2. **No dependency lock:** `requirements.txt` uses broad version ranges (except `openchart==0.2.0`). Observed versions were NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, Streamlit 1.65.0, Plotly 7.1.0, yfinance 1.7.0, OpenChart 0.2.0, pytest 8.4.2. External experiment scripts and CSVs live in user-specific Temp paths and are not portable project artifacts.
3. **Provider limitations:** The downloader preserved the official roster CSV, retrieval time, and roster hash, but the stock manifest does not record the installed yfinance version. Historical data may be revised. OpenChart symbol search succeeded in prior diagnostics, while controlled 5m and 1d historical calls returned successful HTTP/JSON responses with zero candles; Yahoo fallback remains the working path observed in prior experiments.
4. **Data limitations:** Current constituents are not point-in-time NIFTY membership; two price-discontinuity cases were excluded. There is no market-cost, turnover, or profitability analysis. The repeated holdout and walk-forward results are retrospective, and no untouched evaluation period remains in these files.
5. **Test coverage boundary:** The repository has no dedicated Streamlit UI test module. The README's full suite includes storage tests that create temporary SQLite databases. Those tests were deliberately not run under the explicit no-SQLite-write boundary; project persistence behavior is therefore not reverified in this audit.

## Tests and commands

Ran the non-SQLite-writing existing project tests with the project `.venv`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '.\.venv\Scripts\python.exe' -B -m pytest -p no:cacheprovider -q tests\test_data_collection.py tests\test_data_quality.py tests\test_exploratory_analysis.py tests\test_forecasting.py tests\test_market_data_service.py tests\test_model_training.py tests\test_nifty50_downloader.py tests\test_preprocessing.py tests\test_technical_indicators.py -k 'not successful_fetch_persists_only_the_selected_provider_source'
```

Result: **63 passed, 1 deselected in 2.69s**. The deselected service-persistence test and the entire `tests/test_market_data_storage.py` module were excluded because they write temporary SQLite databases. No network-dependent tests were intentionally invoked; downloader/provider tests use mocks.

Also ran a read-only environment inventory with `.venv` Python and inspected results with `Get-Content`/`rg`. No packages were installed. No SQLite was written, no data downloaded, and no backup folders touched.

## Files changed in this handoff

- Added `ML_RESEARCH_INDEX.md`.
- Added `FINAL_HANDOFF.md`.
- No production code, dashboard behavior, test code, datasets, experiment outputs, SQLite files, or backups changed.

## Recommended next steps

1. Keep current dashboard behavior and treat displayed models as educational baselines; do not integrate the experimental candidates.
2. If a dashboard metric revision is approved later, separate neutral persistence from directional classifiers and report directional coverage with a train-only directional baseline.
3. Before the next modeling cycle, pin a supported Python/dependency environment and copy research artifacts from Temp into a durable, access-controlled location if long-term retention is needed.
4. Resume predictive evaluation only with newer independent data or a stronger pre-specified hypothesis. Preserve a future chronological period from selection and report stock-scale-normalized errors, clustered uncertainty, and trading costs only if evaluating a trading strategy.


