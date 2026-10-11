
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def main():
    # Load the housing dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Split data consistently for fair model comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression()
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        })

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
