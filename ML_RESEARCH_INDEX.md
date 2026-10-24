# Stock-market ML research index

**Decision:** Keep the dashboard's existing models as educational baselines. No candidate has met the evidence threshold for dashboard integration. All results below are retrospective: the histories were audited, and the 2026 holdout was repeatedly examined. No period in this dataset is now an untouched test set.

## Dataset and provenance

- Dataset: `the local NIFTY 50 dataset run directory`.
- The saved official NSE roster contains 50 constituents, retrieved 2026-10-24 17:24:16 UTC from `https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv` (SHA-256 recorded in `nifty50_constituents_source.json`). The dataset manifest reports 50 successful ticker histories, generally 2021-10-25 through 2026-10-23, including `Adj Close`.
- TRENT.NS and TMPV.NS were excluded from the main model comparisons for the audited price discontinuities; 48 stocks were eligible. JIOFIN.NS has shorter coverage.
- Historical prices were obtained with yfinance. The installed version observed in this audit is 1.7.0. OpenChart symbol lookup was confirmed, but previous controlled historical requests for RELIANCE returned HTTP/JSON success with zero candle items at both `5m` and `1d`; the project retains Yahoo Finance fallback.

## Experiments

### 1. Price-level Linear Regression and Random Forest

- **Hypothesis:** Existing models can improve next-session adjusted-close forecasts over Naive Persistence.
- **Dataset/design:** 48 eligible NIFTY 50 histories. The evaluated targets were 2026-02-03 through 2026-10-23 (47 stocks start Feb 3; JIOFIN.NS starts May 15). Existing `train_and_evaluate()` pipeline; feature rows are chronological, earliest 70% fit, next 15% reserved, final 15% evaluated, with the one-row training-label purge. The NIFTY evaluation wrapper passes `Adj Close`; features are trailing close, lags, returns, and technical indicators.
- **Recorded mean per-stock results:** Naive Persistence MAE/RMSE `30.402413 / 40.908094`; Linear Regression `30.933280 / 41.469638` (lower MAE on 6/48 stocks); Random Forest `192.701460 / 220.710308` (lower MAE on 0/48).
- **Limitations/conclusion:** One final chronological holdout, reused in later comparisons; absolute-price errors mix price scales. RF tree predictions were bounded by training target levels and stayed materially low on stocks whose test prices rose above the training range. Neither model earned integration.
- **Files:** `ml_evaluation\results\aggregate_summary.csv`, `per_stock_metrics.csv`, `test_predictions.csv`, `run_metadata.json`.

### 2. Return-target Linear Regression and Random Forest

- **Hypothesis:** Predicting next-session simple return and converting it to price may avoid RF's price-level extrapolation limit.
- **Dataset/design:** Same 48 stocks and the exact same saved final-test target dates, 2026-02-03 through 2026-10-23 (JIOFIN.NS starts May 15); target `AdjClose[t+1]/AdjClose[t]-1`; same chronological split and purge; zero-return baseline; return forecasts converted with feature-date Adj Close. TRENT/TMPV excluded.
- **Recorded mean per-stock MAE/RMSE:** Zero Return `30.402413 / 40.908094`; Linear Regression Return `31.182598 / 41.776692` (beats baseline MAE on 4/48); Random Forest Return `41.731686 / 52.262664` (0/48).
- **Limitations/conclusion:** RF errors improved versus price-level RF but remained worse than zero return. The holdout had already been repeatedly inspected; no independent confirmation.
- **Files:** `ml_evaluation\return_experiment\results\aggregate_summary.csv`, `per_stock_metrics.csv`, `test_predictions.csv`, `run_metadata.json`.

### 3. Scale-normalized and directional reanalysis

- **Hypothesis:** Absolute price scale may hide comparisons, and ML may have directional skill even without lower price error.
- **Dataset/design:** Re-analysis of saved predictions only; no training. Normalized MAE = `100 × stock MAE / mean test actual Adj Close`. MAPE = `100 × mean(|prediction − actual| / actual)`. Direction hit compared predicted and actual next-session return signs; actual zero returns were excluded and predicted zero on a nonzero return counted as a miss. The directional baseline used each stock's training-period majority sign.
- **Recorded macro normalized-MAE/MAPE:** Zero `1.247691% / 1.256744%`; price LR `1.274690% / 1.284911%`; price RF `7.223030% / 7.106905%`; return LR `1.287689% / 1.297426%`; return RF `1.686671% / 1.686786%`.
- **Limitations/conclusion:** The same dates were used for many ticker and subperiod comparisons. No ML model consistently improved on zero return; diagnostics are retrospective and not trading tests.

### 4. Fixed 20-session rolling-mean return

- **Hypothesis:** A simple trailing mean of recent returns may capture short-term drift without added model complexity.
- **Dataset/design:** Same prior 2026 final-holdout dates (2026-02-03 through 2026-10-23) and 48 stocks. Fixed `20` observations, no parameter tuning; prediction uses returns known through each feature date. Compared with zero-return price forecast.
- **Recorded results:** Zero MAE/RMSE `30.402413 / 40.908094`; rolling mean `31.345547 / 41.967164`. Normalized MAE `1.247691% / 1.287720%`; MAPE `1.256744% / 1.296028%`. Rolling mean improved MAPE on 0/48 stocks. Direction hit was 50.4619% versus a training-majority 49.4671% on 8,444 nonzero observations in this single reused holdout.
- **Limitations/conclusion:** The small directional difference was not a significance test and was observed after repeated inspection of the same holdout. Price errors worsened; do not integrate.
- **Files:** `ml_evaluation\rolling_mean_return_experiment\PROGRESS.md` and `results\run_20261024T175311Z`.

### 5. Quarterly expanding-origin directional walk-forward

- **Hypothesis:** The fixed rolling-mean directional hit rate may recur across market periods.
- **Dataset/design:** 15 contiguous, non-overlapping target-date windows, actual targets 2023-01-02 through 2026-10-23 (configured window ends 2026-10-24 exclusive), 48 stocks; each quarter's majority direction uses only returns with target dates before that quarter. Rolling mean updates from past returns known on each forecast date. Fixed window 20; no tuning. Results from 2026-02-03 onward overlap the previously inspected holdout; historical quarters were in old training/validation data and are also retrospective.
- **Recorded aggregate:** 44,875 stock-date forecasts; 44,638 nonzero actual returns. Rolling mean hit `50.3293%` (22,466/44,638); training-majority hit `50.5242%` (22,553/44,638). Rolling minus majority `−0.1949` percentage points; two-way ticker/20-session-block bootstrap 95% CI `[-1.6852, +1.3362]` pp, centered bootstrap `p=0.806`. Rolling mean MAPE `1.192049%` versus zero `1.155419%`; no ticker had better overall MAPE.
- **Limitations/conclusion:** The candidate was proposed after earlier holdout results, so the bootstrap is conditional and not independent confirmation. Quarterly/stock slices are descriptive, not separate tests. No repeatable directional advantage; stop experiments on this dataset pending new independent data or a stronger pre-specified hypothesis.
- **Files:** `ml_evaluation\directional_walkforward\PROGRESS.md`, `PRIOR_EVALUATION_USAGE.md`, and `results\run_20261024T175935Z`.

## Integration decision

No price-level, return-based, or rolling-mean candidate has demonstrated a credible, repeatable improvement over simple baselines. None is approved for dashboard integration. The dashboard's current models remain educational experiments only.


