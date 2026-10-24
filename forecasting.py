"""Recursive experimental forecasts using only the latest history and predictions."""

import pandas as pd

from model_training import FEATURES
from technical_indicators import add_technical_indicators


def recursive_forecast(data: pd.DataFrame, model: object, horizon: int = 5, price_column: str = "Close") -> pd.DataFrame:
    """Forecast 1–10 trading sessions recursively (no future actual prices used)."""
    if not 1 <= horizon <= 10:
        raise ValueError("Forecast horizon must be between 1 and 10 trading days.")
    if price_column not in data:
        raise ValueError(f"Price column {price_column!r} is missing.")
    history = pd.to_numeric(data[price_column], errors="coerce").dropna().sort_index()
    if len(history) < 50:
        raise ValueError("At least 50 valid closing prices are required to forecast.")
    outputs = []
    for _ in range(horizon):
        frame = pd.DataFrame({"Close": history})
        features = add_technical_indicators(frame)
        row = features.iloc[-1].copy()
        row["Lag1"], row["Lag2"], row["Lag3"] = history.iloc[-1], history.iloc[-2], history.iloc[-3]
        row["Return1"] = row["Daily Return"]
        vector = row[FEATURES].to_frame().T
        prediction = float(model.predict(vector)[0])
        next_date = history.index[-1] + pd.offsets.BDay(1)
        outputs.append({"Date": next_date, "Predicted Close": prediction})
        history.loc[next_date] = prediction
    return pd.DataFrame(outputs).set_index("Date")
