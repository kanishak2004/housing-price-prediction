
import argparse
import joblib
import pandas as pd


def predict_price(model_path, values):
    model = joblib.load(model_path)

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

    input_data = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(input_data)[0]

    # California Housing targets are measured in units of $100,000
    return prediction * 100_000


def main():
    parser = argparse.ArgumentParser(
        description="Predict a California housing value."
    )
    parser.add_argument(
        "--model",
        default="models/best_model.joblib",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--values",
        nargs=8,
        type=float,
        required=True,
        metavar=("MEDINC", "HOUSEAGE", "AVEROOMS", "AVEBEDRMS",
                 "POPULATION", "AVEOCCUP", "LATITUDE", "LONGITUDE"),
        help="Eight input features in California Housing dataset order",
    )
    args = parser.parse_args()

    predicted_value = predict_price(args.model, args.values)
    print(f"Predicted housing value: ${predicted_value:,.2f}")


if __name__ == "__main__":
    main()
