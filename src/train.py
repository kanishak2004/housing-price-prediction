
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"


def main():
    # Load the California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Use the same split for every model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression(),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }

    results = []
    best_model = None
    best_rmse = float("inf")
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": rmse,
            "R2": r2_score(y_test, predictions),
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df.to_string(index=False))
    print(f"\nBest model by RMSE: {best_name}")

    # Save the entire pipeline, including preprocessing
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"Saved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
