import numpy as np
import pandas as pd
import pytest

from forecasting import recursive_forecast
from model_training import train_and_evaluate


def test_recursive_forecast_length_and_future_dates():
    close = 100 + np.arange(180) * 0.2 + np.sin(np.arange(180))
    frame = pd.DataFrame({"Close": close}, index=pd.bdate_range("2023-01-02", periods=180))
    model = train_and_evaluate(frame).models["Linear Regression"]
    forecast = recursive_forecast(frame, model, horizon=5)
    assert len(forecast) == 5
    assert forecast.index.min() > frame.index.max()
    assert forecast["Predicted Close"].notna().all()


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError, match="horizon"):
        recursive_forecast(pd.DataFrame({"Close": np.arange(60.)}), object(), 11)
