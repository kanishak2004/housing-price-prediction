# 🏠 Housing Price Prediction using Machine Learning

## Overview

This project explores housing price prediction using Python and regression algorithms. It focuses on organizing a machine learning workflow into separate modules for model training, evaluation, and inference.

The project is being developed as a practical exercise in machine learning, with an emphasis on reusable code, model comparison, and maintainable project structure.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Evaluation:** MAE, RMSE, R²
- **Model Persistence:** Joblib
- **Testing:** Pytest (test files included; execution should be verified)

## Project Structure

```text
housing-price-prediction/
├── src/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── tests/
│   ├── test_train.py
│   └── test_predict.py
├── models/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

## Workflow

1. Load the California Housing dataset.
2. Split the data into training and testing sets.
3. Train Linear Regression, Random Forest, and XGBoost models.
4. Compare model performance using MAE, RMSE, and R².
5. Select the model with the lowest RMSE and save it for future predictions.
6. Use the prediction module to estimate housing values from input features.

## Evaluation Metrics

- **MAE:** Average absolute difference between actual and predicted values.
- **RMSE:** Prediction error metric that penalizes larger errors more heavily.
- **R²:** Indicates how well the model explains variation in the target values.

## Setup

Clone the repository:

```bash
git clone https://github.com/kanishak2004/housing-price-prediction.git
cd housing-price-prediction
```

Create and activate a virtual environment on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install pytest
```

## Current Status

The source files for training, prediction, evaluation, and testing are present in the repository. End-to-end execution and model-saving behavior still need to be verified in a Python environment.

## Future Improvements

- Integrate the evaluation utility directly into the training pipeline.
- Add API-based predictions with FastAPI.
- Expand automated tests for model training and inference.
- Add experiment tracking and model versioning.
- Explore containerization and deployment after local execution is verified.

## Learning Goals

This project helps me strengthen my understanding of regression algorithms, model evaluation, Python project organization, and the foundations of production-oriented machine learning workflows.
