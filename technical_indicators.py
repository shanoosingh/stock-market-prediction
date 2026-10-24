"""Causal, independently testable technical indicator calculations."""

from __future__ import annotations

import pandas as pd


def add_technical_indicators(data: pd.DataFrame, price_column: str = "Close") -> pd.DataFrame:
    """Add trailing indicators; unavailable warm-up values remain NaN.

    RSI uses Wilder's exponentially smoothed average gains/losses (alpha=1/n).
    Volatility is the 20-observation sample standard deviation of daily returns,
    annualized by sqrt(252) trading sessions.
    """
    if price_column not in data:
        raise ValueError(f"Price column {price_column!r} is missing.")
    out = data.copy()
    close = pd.to_numeric(out[price_column], errors="coerce")
    out["Daily Return"] = close.pct_change(fill_method=None)
    out["SMA20"] = close.rolling(20, min_periods=20).mean()
    out["SMA50"] = close.rolling(50, min_periods=50).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    out["RSI14"] = rsi
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["MACD"] = ema12 - ema26
    out["MACD Signal"] = out["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["MACD Histogram"] = out["MACD"] - out["MACD Signal"]
    out["Rolling Volatility"] = out["Daily Return"].rolling(20, min_periods=20).std() * (252 ** 0.5)
    return out
