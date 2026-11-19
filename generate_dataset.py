"""
generate_dataset.py
-------------------
Creates a synthetic (artificially generated) dataset of 200 student records
and saves it to 'student_data.csv' for demonstration and model training.

NOTE: These values are randomly generated for educational purposes and are NOT
real student records.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Default output path: always save student_data.csv next to this script
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "student_data.csv"


def generate_student_dataset(
    num_records: int = 200,
    random_seed: int = 42,
    output_path: Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset of student study habits and exam marks.

    Parameters
    ----------
    num_records : int
        Number of synthetic student rows to generate (default: 200).
    random_seed : int
        Seed for NumPy's random generator so results are reproducible.
    output_path : Path
        File path where the CSV file will be saved.

    Returns
    -------
    pd.DataFrame
        The generated student dataset.
    """
    # Create a random number generator with a fixed seed for repeatability
    rng = np.random.default_rng(random_seed)

    # 1. Daily study hours (realistic range: ~0.5 to 12.0 hours per day)
    study_hours = rng.normal(loc=4.5, scale=2.3, size=num_records)
    study_hours = np.clip(study_hours, 0.5, 12.0)

    # 2. Attendance percentage (40% to 100%)
    # Students who study regularly tend to have slightly higher attendance,
    # plus natural individual variation.
    attendance = 69.0 + 2.0 * study_hours + rng.normal(loc=0.0, scale=11.0, size=num_records)
    attendance = np.clip(attendance, 40.0, 100.0)

    # 3. Previous exam marks (25 to 98 out of 100)
    previous_marks = 56.0 + 2.2 * study_hours + rng.normal(loc=0.0, scale=13.0, size=num_records)
    previous_marks = np.clip(previous_marks, 25.0, 98.0)

    # 4. Assignment scores (30 to 100 out of 100)
    assignment_score = (
        42.0
        + 1.8 * study_hours
        + 0.25 * attendance
        + rng.normal(loc=0.0, scale=10.5, size=num_records)
    )
    assignment_score = np.clip(assignment_score, 30.0, 100.0)

    # 5. Final exam marks (target variable, 0 to 100)
    # Realistic weighted combination of all four features plus natural exam noise.
    noise = rng.normal(loc=0.0, scale=3.0, size=num_records)
    final_exam_marks = (
        -6.0
        + 2.8 * study_hours
        + 0.22 * attendance
        + 0.35 * previous_marks
        + 0.25 * assignment_score
        + noise
    )
    final_exam_marks = np.clip(final_exam_marks, 0.0, 100.0)

    # Put everything into a clean Pandas DataFrame rounded to 1 decimal place
    df = pd.DataFrame(
        {
            "study_hours": np.round(study_hours, 1),
            "attendance": np.round(attendance, 1),
            "previous_marks": np.round(previous_marks, 1),
            "assignment_score": np.round(assignment_score, 1),
            "final_exam_marks": np.round(final_exam_marks, 1),
        }
    )

    # Save the dataset to CSV without row index numbers
    output_path = Path(output_path)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    dataset = generate_student_dataset()
    print(f"Synthetic dataset generated successfully: {DEFAULT_CSV_PATH}")
    print(f"Total records: {len(dataset)}")
    print("\nFirst 5 rows:")
    print(dataset.head().to_string(index=False))
    print("\nSummary statistics:")
    print(dataset.describe().round(2).to_string())
