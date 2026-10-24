"""Leakage-aware next-trading-day model training and holdout evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from technical_indicators import add_technical_indicators

FEATURES = ["Close", "Lag1", "Lag2", "Lag3", "Return1", "SMA20", "SMA50", "EMA20", "RSI14", "MACD", "MACD Signal", "Rolling Volatility"]


@dataclass
class TrainingResult:
    """Trained estimators, test predictions, metrics, and feature definitions."""

    models: dict[str, object]
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_columns: list[str]
    train_end: pd.Timestamp


def prepare_supervised_data(data: pd.DataFrame, price_column: str = "Close") -> pd.DataFrame:
    """Align features at date t to close at t+1; never include future values."""
    if price_column not in data:
        raise ValueError(f"Price column {price_column!r} is missing.")
    base = data[[price_column]].rename(columns={price_column: "Close"}).copy()
    base["Close"] = pd.to_numeric(base["Close"], errors="coerce")
    # Keep missing-close timestamps in the timeline. Dropping them before
    # shift/indicator construction would make observations on either side of
    # a missing candle look adjacent and create a multi-session return/target.
    base = base.sort_index()
    if base.index.has_duplicates:
        base = base.loc[~base.index.duplicated(keep="last")]
    feat = add_technical_indicators(base)
    feat["Lag1"] = feat["Close"].shift(1)
    feat["Lag2"] = feat["Close"].shift(2)
    feat["Lag3"] = feat["Close"].shift(3)
    feat["Return1"] = feat["Daily Return"]
    feat["Target"] = feat["Close"].shift(-1)
    return feat[FEATURES + ["Target"]].replace([np.inf, -np.inf], np.nan).dropna()


def train_and_evaluate(data: pd.DataFrame, price_column: str = "Close", min_samples: int = 80) -> TrainingResult:
    """Fit on earliest 70%; report metrics only on final 15% test period.

    The middle chronological 15% is reserved as validation and is not used to
    fit or tune this fixed educational model configuration.
    """
    supervised = prepare_supervised_data(data, price_column)
    n = len(supervised)
    if n < min_samples:
        valid_prices = pd.to_numeric(data[price_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        # SMA50 needs 50 closes, and the last feature row also needs its next close as target.
        first_row_minimum = 51
        evaluation_history_estimate = min_samples + 50
        raise ValueError(
            f"Insufficient history for model evaluation: {len(valid_prices)} valid closing-price "
            f"observations produced {n} complete supervised rows. SMA50 needs 50 prior/current "
            f"closes and the next-session target needs one more, so at least {first_row_minimum} "
            f"valid consecutive closes are needed for even one row. Evaluation requires "
            f"{min_samples} complete rows (about {evaluation_history_estimate} valid consecutive "
            "closes with no gaps). No model was trained."
        )
    train_end_i, val_end_i = int(n * .70), int(n * .85)
    if train_end_i < 2 or val_end_i <= train_end_i or val_end_i >= n:
        raise ValueError("Not enough observations for chronological train, validation, and test periods.")
    # Purge the final training feature row: its next-session target would be
    # the first validation date's close, leaking a validation observation into fit.
    train = supervised.iloc[:train_end_i - 1]
    # Validation is deliberately kept out of fitting; its span is reserved for later model selection.
    test = supervised.iloc[val_end_i:]
    X_train, y_train = train[FEATURES], train["Target"]
    X_test, y_test = test[FEATURES], test["Target"]
    estimators = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    models: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {"Naive Persistence": test["Close"].to_numpy()}
    for name, model in estimators.items():
        model.fit(X_train, y_train)
        models[name] = model
        predictions[name] = model.predict(X_test)
    rows = []
    for name, pred in predictions.items():
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": mean_squared_error(y_test, pred) ** .5,
                     "R-squared": r2_score(y_test, pred),
                     "Directional accuracy": float(np.mean(np.sign(pred - test["Close"].to_numpy()) == np.sign(y_test.to_numpy() - test["Close"].to_numpy())))})
    prediction_frame = pd.DataFrame({"Actual": y_test, **predictions}, index=test.index)
    prediction_frame.index.name = "Feature date (prediction made after close)"
    # The final training label is the close on the next row's date.
    return TrainingResult(models, pd.DataFrame(rows).set_index("Model"), prediction_frame, FEATURES.copy(), supervised.index[train_end_i - 1])
