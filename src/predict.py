
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.joblib"

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def predict_price(values, model_path=DEFAULT_MODEL_PATH):
    if len(values) != len(FEATURE_NAMES):
        raise ValueError("Exactly eight feature values are required.")

    model = joblib.load(model_path)
    input_data = pd.DataFrame([values], columns=FEATURE_NAMES)
    prediction = float(model.predict(input_data)[0])

    # Dataset target is expressed in units of $100,000.
    return prediction * 100_000
