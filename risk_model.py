from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "wdbc.data"
MODEL_PATH = ROOT_DIR / "models" / "tabular_model.pkl"


# The original UCI dataset contains 30 measurements.
# To keep the MVP UI simple, we use the 10 "mean" measurements.
FEATURE_NAMES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
]


ALL_COLUMNS = [
    "id",
    "diagnosis",

    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]


def confidence_level(probability: float) -> str:
    """
    Convert a model probability into a simple confidence category.

    This is model confidence, not clinical certainty.
    """
    confidence = max(probability, 1.0 - probability)

    if confidence < 0.60:
        return "Low"

    if confidence < 0.80:
        return "Moderate"

    return "Higher"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the UCI Breast Cancer Wisconsin Diagnostic dataset.

    The downloaded file is expected at:
        data/wdbc.data
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download the UCI Breast Cancer Wisconsin Diagnostic dataset "
            "and place the wdbc.data file there."
        )

    data = pd.read_csv(
        DATA_PATH,
        header=None,
        names=ALL_COLUMNS,
    )

    # Convert diagnosis:
    # M = malignant
    # B = benign
    data["diagnosis"] = data["diagnosis"].map(
        {
            "M": 1,
            "B": 0,
        }
    )

    if data["diagnosis"].isna().any():
        raise ValueError("Unexpected diagnosis values found in the dataset.")

    X = data[FEATURE_NAMES].copy()
    y = data["diagnosis"].astype(int)

    return X, y


def train_model() -> dict:
    """
    Train the structured classification model.

    Returns:
        Dictionary containing the trained model and real evaluation metrics.
    """

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # StandardScaler + Logistic Regression is simple and suitable
    # for this small numeric dataset.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }

    # Save the model and the information needed to interpret it.
    model_package = {
        "model": pipeline,
        "features": FEATURE_NAMES,
        "metrics": metrics,
        "dataset": "UCI Breast Cancer Wisconsin Diagnostic",
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    print("\nStructured model training completed.")
    print(f"Model saved to: {MODEL_PATH}")

    print("\nEvaluation results from the held-out test set:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return model_package


def load_model() -> dict:
    """
    Load the saved structured ML model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}.\n"
            "Run:\n"
            "python modules/risk_model.py"
        )

    return joblib.load(MODEL_PATH)


def predict_risk(patient_data: dict) -> dict:
    """
    Generate a prediction for one set of structured measurements.

    Example input:

    {
        "mean_radius": 14.2,
        "mean_texture": 20.1,
        ...
    }
    """

    package = load_model()

    model = package["model"]

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in patient_data
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    input_data = pd.DataFrame(
        [
            {
                feature: float(patient_data[feature])
                for feature in FEATURE_NAMES
            }
        ]
    )

    prediction = int(
        model.predict(input_data)[0]
    )

    probabilities = model.predict_proba(input_data)[0]

    benign_probability = float(probabilities[0])
    malignant_probability = float(probabilities[1])

    if prediction == 1:
        label = "Malignant-pattern prediction"
        probability = malignant_probability
    else:
        label = "Benign-pattern prediction"
        probability = benign_probability

    confidence = confidence_level(probability)

    return {
        "prediction": label,
        "class": "malignant" if prediction == 1 else "benign",
        "probability": probability,
        "confidence": confidence,
        "benign_probability": benign_probability,
        "malignant_probability": malignant_probability,
    }


def get_feature_importance() -> list[dict]:
    """
    Return the absolute logistic-regression coefficient magnitude
    for the ten features used by the MVP.
    """

    package = load_model()

    model = package["model"]

    classifier = model.named_steps["classifier"]

    coefficients = classifier.coef_[0]

    importance = []

    for feature, coefficient in zip(
        FEATURE_NAMES,
        coefficients,
    ):
        importance.append(
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "importance": abs(float(coefficient)),
            }
        )

    importance.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return importance


if __name__ == "__main__":
    train_model()