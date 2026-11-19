"""
train_model.py
--------------
Loads the synthetic student dataset from 'student_data.csv', trains a
Multiple Linear Regression model using Scikit-learn, evaluates it with
MAE and R² score, and provides helper functions to validate inputs and
predict a student's expected exam marks.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Default path to the CSV dataset stored in the same folder as this file
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "student_data.csv"

# Input features (X) and target column (y)
FEATURE_COLUMNS: List[str] = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
]
TARGET_COLUMN: str = "final_exam_marks"

# Human-readable labels for displaying features nicely in the app
FEATURE_LABELS: Dict[str, str] = {
    "study_hours": "Daily Study Hours",
    "attendance": "Attendance (%)",
    "previous_marks": "Previous Exam Marks",
    "assignment_score": "Assignment Score",
}

# Allowed input ranges for validation
INPUT_RANGES: Dict[str, Tuple[float, float]] = {
    "study_hours": (0.0, 16.0),
    "attendance": (0.0, 100.0),
    "previous_marks": (0.0, 100.0),
    "assignment_score": (0.0, 100.0),
}


def load_dataset(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Load and validate the student CSV dataset using Pandas.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Please run 'python generate_dataset.py' first to create it."
        )

    df = pd.read_csv(csv_path)

    # Verify all required columns exist
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Remove any incomplete rows just in case the CSV was manually edited
    df = df.dropna(subset=required_columns).copy()
    if len(df) < 10:
        raise ValueError(
            "Dataset needs at least 10 valid student records to train and test the model."
        )

    return df


def train_and_evaluate_model(
    csv_path: Path = DEFAULT_CSV_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Train a Scikit-learn Multiple Linear Regression model and evaluate its performance.

    Steps:
    1. Load student records from CSV.
    2. Separate input features (X) and target exam marks (y).
    3. Split into training (80%) and testing (20%) sets.
    4. Fit LinearRegression on the training set.
    5. Predict on the test set and compute MAE and R² score.
    """
    df = load_dataset(csv_path)

    # Step 1: Select input features (X) and target variable (y)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Step 3: Create and train the Multiple Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model on unseen test records
    y_pred_test = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    # Map each feature name to its learned weight (coefficient)
    coefficients = {
        feature: float(weight)
        for feature, weight in zip(FEATURE_COLUMNS, model.coef_)
    }

    return {
        "model": model,
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "mae": mae,
        "r2": r2,
        "coefficients": coefficients,
        "intercept": float(model.intercept_),
        "total_records": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def validate_student_inputs(
    study_hours: float,
    attendance: float,
    previous_marks: float,
    assignment_score: float,
) -> Dict[str, float]:
    """
    Validate all user inputs before sending them to the ML model.

    Raises ValueError if any input is missing, non-numeric, NaN/infinite,
    or outside its allowed range.
    """
    raw_inputs = {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_score": assignment_score,
    }

    validated: Dict[str, float] = {}
    errors: List[str] = []

    for field, value in raw_inputs.items():
        label = FEATURE_LABELS[field]
        min_val, max_val = INPUT_RANGES[field]

        # Reject booleans or non-numeric types cleanly
        if isinstance(value, bool) or value is None:
            errors.append(f"{label} must be a valid number between {min_val:g} and {max_val:g}.")
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a valid number between {min_val:g} and {max_val:g}.")
            continue

        if not math.isfinite(numeric_value):
            errors.append(f"{label} must be a finite number.")
            continue

        if numeric_value < min_val or numeric_value > max_val:
            errors.append(f"{label} must be between {min_val:g} and {max_val:g}.")
            continue

        validated[field] = numeric_value

    if errors:
        raise ValueError(" ".join(errors))

    return validated


def predict_student_marks(
    model: LinearRegression,
    study_hours: float,
    attendance: float,
    previous_marks: float,
    assignment_score: float,
) -> Tuple[float, float]:
    """
    Predict a student's expected exam marks using the trained Linear Regression model.

    Returns
    -------
    tuple[float, float]
        (clamped_prediction, raw_prediction)
        - clamped_prediction is bounded to the valid exam range [0, 100].
        - raw_prediction is the direct mathematical output of the linear equation.
    """
    cleaned = validate_student_inputs(
        study_hours=study_hours,
        attendance=attendance,
        previous_marks=previous_marks,
        assignment_score=assignment_score,
    )

    # Create a 1-row DataFrame with the exact feature names used during training
    input_df = pd.DataFrame([cleaned], columns=FEATURE_COLUMNS)

    # Generate prediction from the trained Scikit-learn model
    raw_prediction = float(model.predict(input_df)[0])

    # Exam scores cannot be below 0 or above 100
    clamped_prediction = float(np.clip(raw_prediction, 0.0, 100.0))
    return clamped_prediction, raw_prediction


def categorize_performance(predicted_marks: float) -> Tuple[str, str]:
    """
    Map predicted exam marks (0-100) to a simple performance category
    and a matching status color name.
    """
    if predicted_marks >= 85.0:
        return "Excellent", "green"
    if predicted_marks >= 70.0:
        return "Good", "blue"
    if predicted_marks >= 50.0:
        return "Average", "orange"
    return "Needs Improvement", "red"


def build_prediction_explanation(
    study_hours: float,
    attendance: float,
    previous_marks: float,
    assignment_score: float,
    predicted_marks: float,
    raw_prediction: float,
    coefficients: Dict[str, float],
) -> str:
    """
    Create a short, plain-English explanation based on the student's inputs
    and what the Multiple Linear Regression model learned.
    """
    strengths: List[str] = []
    areas_to_improve: List[str] = []

    # Check daily study hours
    if study_hours >= 6.0:
        strengths.append(f"strong daily study time ({study_hours:.1f} hrs/day)")
    elif study_hours < 3.0:
        areas_to_improve.append(
            f"daily study time is low ({study_hours:.1f} hrs/day; each extra hour adds about +{coefficients['study_hours']:.1f} marks)"
        )

    # Check attendance
    if attendance >= 85.0:
        strengths.append(f"high class attendance ({attendance:.1f}%)")
    elif attendance < 70.0:
        areas_to_improve.append(f"attendance is below 70% ({attendance:.1f}%)")

    # Check previous exam marks
    if previous_marks >= 75.0:
        strengths.append(f"solid previous exam marks ({previous_marks:.1f}/100)")
    elif previous_marks < 55.0:
        areas_to_improve.append(f"lower previous exam marks ({previous_marks:.1f}/100)")

    # Check assignment scores
    if assignment_score >= 80.0:
        strengths.append(f"strong assignment scores ({assignment_score:.1f}/100)")
    elif assignment_score < 60.0:
        areas_to_improve.append(f"assignment scores have room to grow ({assignment_score:.1f}/100)")

    sentences: List[str] = []

    if strengths:
        sentences.append(
            "Your estimate is supported by " + ", ".join(strengths) + "."
        )

    if areas_to_improve:
        sentences.append(
            "Key areas pulling the prediction down: " + "; ".join(areas_to_improve) + "."
        )

    if not strengths and not areas_to_improve:
        sentences.append(
            "Your study hours, attendance, previous marks, and assignment scores are all in a balanced, moderate range."
        )

    if raw_prediction > 100.0:
        sentences.append(
            f"Note: The raw linear formula estimated {raw_prediction:.1f}, which was capped at the maximum possible exam score of 100."
        )
    elif raw_prediction < 0.0:
        sentences.append(
            f"Note: The raw linear formula estimated {raw_prediction:.1f}, which was capped at the minimum exam score of 0."
        )

    return " ".join(sentences)


if __name__ == "__main__":
    results = train_and_evaluate_model()
    print("=== Student Study Time Predictor: Model Training Report ===")
    print(f"Dataset loaded: {DEFAULT_CSV_PATH}")
    print(
        f"Records: {results['total_records']} total "
        f"({results['train_size']} training, {results['test_size']} testing)"
    )
    print(f"\nModel Evaluation (on test data):")
    print(f"  Mean Absolute Error (MAE): {results['mae']:.2f} marks")
    print(f"  R² Score:                  {results['r2']:.4f}")

    print("\nLearned Linear Regression Weights:")
    print(f"  Intercept: {results['intercept']:.4f}")
    for feature_name, weight in results["coefficients"].items():
        print(f"  {FEATURE_LABELS[feature_name]:<22}: {weight:+.4f}")

    # Run one sample prediction to confirm everything works end-to-end
    sample_pred, sample_raw = predict_student_marks(
        model=results["model"],
        study_hours=5.0,
        attendance=85.0,
        previous_marks=75.0,
        assignment_score=80.0,
    )
    sample_category, _ = categorize_performance(sample_pred)
    print("\nSample Prediction (5 hrs study, 85% attendance, 75 prev marks, 80 assignments):")
    print(f"  Predicted Final Exam Marks: {sample_pred:.1f} / 100 ({sample_category})")
