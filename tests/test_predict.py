
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.predict import FEATURE_NAMES, predict_price


def test_predict_price_returns_finite_value(tmp_path):
    X = pd.DataFrame(
        np.ones((5, len(FEATURE_NAMES))),
        columns=FEATURE_NAMES,
    )
    y = np.array([2.0, 2.1, 1.9, 2.2, 2.0])

    model = DummyRegressor(strategy="mean")
    model.fit(X, y)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    result = predict_price([1.0] * 8, model_path)

    assert np.isfinite(result)
    assert result >= 0


def test_predict_price_rejects_wrong_number_of_features():
    try:
        predict_price([1.0, 2.0])
    except ValueError as error:
        assert "eight" in str(error).lower()
    else:
        raise AssertionError("Expected ValueError for invalid input")
