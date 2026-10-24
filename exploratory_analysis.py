"""Summary statistics and Plotly figures for market history."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def summarize_market_data(data: pd.DataFrame, price_column: str = "Close") -> dict[str, object]:
    """Summarize valid prices while preserving gaps in per-candle returns."""
    if data.empty or price_column not in data:
        raise ValueError("Valid market data and a price column are required.")
    raw_prices = data[price_column]
    price_series = pd.to_numeric(raw_prices, errors="coerce").sort_index()
    supplied = raw_prices.notna()
    invalid_prices = supplied & (
        price_series.isna() | ~np.isfinite(price_series) | (price_series <= 0)
    )
    if invalid_prices.any():
        raise ValueError(
            f"Selected {price_column} contains {int(invalid_prices.sum())} nonnumeric, "
            "non-finite, or nonpositive price value(s); summary is unavailable."
        )
    prices = price_series.dropna()
    if prices.empty:
        raise ValueError("No valid closing prices are available.")
    volume = pd.to_numeric(data["Volume"], errors="coerce").dropna() if "Volume" in data else pd.Series(dtype=float)
    returns = price_series.pct_change(fill_method=None)
    return {
        "observations": len(prices), "candles": len(data),
        "start": prices.index.min(), "end": prices.index.max(),
        "highest_close": prices.max(), "lowest_close": prices.min(),
        "latest_close": prices.iloc[-1], "latest_close_date": prices.index[-1],
        "latest_return": returns.iloc[-1], "average_close": prices.mean(),
        "average_volume": volume.mean() if not volume.empty else float("nan"),
        "total_reported_volume": volume.sum() if not volume.empty else float("nan"),
        # Keep missing sessions in the series so a gap is not reported as a one-day return.
        "daily_returns": returns.dropna(),
    }


def price_volume_figure(data: pd.DataFrame, price_column: str = "Close") -> go.Figure:
    """Create interactive price and volume subplot."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], name=price_column), secondary_y=False)
    if "Volume" in data:
        fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", opacity=0.25), secondary_y=True)
    fig.update_layout(title="Price and reported trading volume", hovermode="x unified")
    fig.update_yaxes(title_text="Price (currency depends on exchange)", secondary_y=False)
    fig.update_yaxes(title_text="Shares reported", secondary_y=True)
    return fig
