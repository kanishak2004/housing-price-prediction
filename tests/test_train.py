
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def test_models_produce_valid_predictions():
    # Use a small subset to keep the test lightweight
    housing = fetch_california_housing(as_frame=True)
    X = housing.data.iloc[:200]
    y = housing.target.iloc[:200]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        make_pipeline(SimpleImputer(strategy="median"),
                      StandardScaler(), LinearRegression()),
        make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestRegressor(n_estimators=10, random_state=42)),
        make_pipeline(SimpleImputer(strategy="median"),
                      XGBRegressor(n_estimators=10, max_depth=3,
                                   random_state=42, n_jobs=1)),
    ]

    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)
        assert np.isfinite(predictions).all()
        assert mean_absolute_error(y_test, predictions) >= 0
