
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.predict import predict_price


def test_predict_price_returns_finite_value(tmp_path):
    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    # Create a small model for testing the prediction interface
    X = pd.DataFrame(
        np.ones((5, 8)),
        columns=feature_names,
    )
    y = np.array([2.0, 2.1, 1.9, 2.2, 2.0])

    model = DummyRegressor(strategy="mean")
    model.fit(X, y)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    values = [1.0] * 8
    result = predict_price(str(model_path), values)

    assert isinstance(result, (int, float, np.floating))
    assert np.isfinite(result)
    assert result >= 0
