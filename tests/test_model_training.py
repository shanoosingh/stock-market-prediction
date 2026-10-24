import numpy as np
import pandas as pd
import pytest

from model_training import prepare_supervised_data, train_and_evaluate


def synthetic_prices(n=260):
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.01, n)
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + returns)}, index=pd.bdate_range("2022-01-03", periods=n))


def test_target_is_next_observation_and_feature_date_is_current():
    data = pd.DataFrame({"Close": np.arange(1.0, 100.0)}, index=pd.bdate_range("2024-01-01", periods=99))
    supervised = prepare_supervised_data(data)
    first = supervised.iloc[0]
    assert first["Target"] == first["Close"] + 1
    assert first.name < supervised.index[-1]


def test_future_changes_do_not_change_earlier_features():
    frame = synthetic_prices(180)
    cutoff = frame.index[120]
    altered = frame.copy()
    altered.loc[altered.index > cutoff, "Close"] *= 50
    left, right = prepare_supervised_data(frame), prepare_supervised_data(altered)
    cols = ["Close", "Lag1", "SMA20", "RSI14", "MACD"]
    pd.testing.assert_frame_equal(left.loc[:cutoff, cols], right.loc[:cutoff, cols])


def test_missing_close_is_not_removed_before_target_shifting():
    frame = synthetic_prices(200)
    missing_date = frame.index[80]
    frame.loc[missing_date, "Close"] = np.nan

    supervised = prepare_supervised_data(frame)

    # The previous date cannot jump over the missing target, and the rows just
    # after the gap cannot pretend their lag features were consecutive closes.
    assert frame.index[79] not in supervised.index
    assert missing_date not in supervised.index
    assert frame.index[81] not in supervised.index
    assert frame.index[82] not in supervised.index
    assert frame.index[83] not in supervised.index
    assert frame.index[130] in supervised.index
    assert supervised.loc[frame.index[130], "Target"] == frame.loc[frame.index[131], "Close"]


def test_chronological_training_and_all_models_evaluated():
    result = train_and_evaluate(synthetic_prices())
    assert result.predictions.index.is_monotonic_increasing
    assert set(result.metrics.index) == {"Naive Persistence", "Linear Regression", "Random Forest"}
    assert result.predictions.index.min() > result.train_end
    assert {"MAE", "RMSE", "R-squared"} <= set(result.metrics.columns)


def test_training_target_does_not_overlap_validation_start(monkeypatch):
    from sklearn.linear_model import LinearRegression

    frame = synthetic_prices()
    supervised = prepare_supervised_data(frame)
    train_stop = int(len(supervised) * .70)
    expected_last_training_label = supervised["Target"].iloc[train_stop - 2]
    seen_labels = []
    original_fit = LinearRegression.fit

    def capture_fit(model, features, target):
        seen_labels.extend(target.tolist())
        return original_fit(model, features, target)

    monkeypatch.setattr(LinearRegression, "fit", capture_fit)
    train_and_evaluate(frame)
    assert len(seen_labels) == train_stop - 1
    assert seen_labels[-1] == expected_last_training_label


def test_insufficient_data_rejected():
    with pytest.raises(ValueError, match="21 valid closing-price observations produced 0 complete supervised rows"):
        train_and_evaluate(synthetic_prices(21))


def test_insufficient_data_message_explains_sma50_warmup():
    with pytest.raises(ValueError, match="at least 51 valid consecutive closes"):
        train_and_evaluate(synthetic_prices(70))
