# Student Study Time Predictor

A beginner-friendly Python and Streamlit web application that predicts a student's expected final exam marks (out of 100) using **Multiple Linear Regression** (Scikit-learn).

> **Note on Data:** The included `student_data.csv` file contains **200 synthetic (randomly generated) demonstration records**, not real student data.

## Inputs Used for Prediction

- **Daily study hours** (0–16 hours)
- **Attendance percentage** (0–100%)
- **Previous exam marks** (0–100)
- **Assignment score** (0–100)

## Project Structure

```text
student-study-predictor/
├── app.py               # Streamlit web interface and Matplotlib charts
├── generate_dataset.py  # Generates 200 synthetic student records into student_data.csv
├── train_model.py       # Loads CSV, trains & evaluates Linear Regression, predicts marks
├── student_data.csv     # Synthetic dataset (200 rows)
├── requirements.txt     # Required Python libraries
└── README.md            # Project guide and instructions
```

## Technologies Used

- **Python**
- **Streamlit** — web user interface
- **Pandas** — reading and managing CSV data
- **NumPy** — numerical computations
- **Scikit-learn** — Multiple Linear Regression (`LinearRegression`), train/test split, MAE, and R² score
- **Matplotlib** — scatter plots and model evaluation graphs

## How to Run (Windows PowerShell)

1. Open PowerShell and navigate into the project folder:
   ```powershell
   cd "z:\ai proj new\student-study-predictor"
   ```

2. Install the required libraries (if not already installed):
   ```powershell
   pip install -r requirements.txt
   ```

3. *(Optional)* Regenerate the synthetic dataset:
   ```powershell
   python generate_dataset.py
   ```

4. *(Optional)* Run the model training script directly to view console metrics:
   ```powershell
   python train_model.py
   ```

5. Launch the Streamlit web app:
   ```powershell
   streamlit run app.py
   ```

Your default web browser will open automatically (typically at `http://localhost:8501`).
