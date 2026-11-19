"""
app.py
------
Streamlit web interface for the Student Study Time Predictor.

Designed with a warm vintage-minimalist editorial aesthetic inspired by
classic lightweight web pages: cream paper canvas, Tempus Sans ITC display
typography, deep crimson title, ASCII logo artwork, hairline dividers,
and matching Matplotlib charts.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from generate_dataset import generate_student_dataset
from train_model import (
    DEFAULT_CSV_PATH,
    FEATURE_LABELS,
    build_prediction_explanation,
    categorize_performance,
    predict_student_marks,
    train_and_evaluate_model,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Study Time Predictor",
    page_icon="📖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Color palette (warm vintage-minimal cream, crimson, slate navy, olive ink)
# ---------------------------------------------------------------------------
CREAM_BG = "#FFFEE6"
PARCHMENT_BG = "#FAF6DC"
INK_DARK = "#232318"
INK_OLIVE = "#4C4B2F"
NAVY_HEADING = "#2D354C"
CRIMSON_ACCENT = "#9E0B1C"
RULE_DARK = "#3A3828"
BORDER_SUBTLE = "#9A967D"

# Exact font from the inspiration image (Tempus Sans ITC), with graceful fallback
DISPLAY_FONT = '"Tempus Sans ITC", "TempusSansITC", "Klee One", Georgia, serif'


# ---------------------------------------------------------------------------
# Minimal vintage editorial styling (overrides both light and dark modes)
# ---------------------------------------------------------------------------
def inject_minimal_theme() -> None:
    """Apply the minimal cream-and-crimson retro editorial theme."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Klee+One:wght@400;600&display=swap');

        :root {{
            color-scheme: light !important;
        }}

        /* Main page canvas and header bar */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        .stApp {{
            background-color: {CREAM_BG} !important;
            color: {INK_DARK} !important;
            font-family: Arial, Verdana, "MS PGothic", sans-serif !important;
        }}

        [data-testid="stHeader"] {{
            background-color: {CREAM_BG} !important;
            border-bottom: none !important;
        }}

        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] svg {{
            color: {INK_OLIVE} !important;
            fill: {INK_OLIVE} !important;
        }}

        /* Centered classic editorial container */
        [data-testid="stMainBlockContainer"],
        .block-container {{
            max-width: 980px !important;
            padding-top: 3.2rem !important;
            padding-bottom: 4.2rem !important;
            padding-left: 1.8rem !important;
            padding-right: 1.8rem !important;
        }}

        /* General text color & comfortable larger size inside Streamlit markdown */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {{
            color: {INK_OLIVE};
            font-size: 1.04rem;
            line-height: 1.72;
        }}

        /* Top Hero Title — ONLY "Study Time Predictor" in Tempus Sans ITC */
        .hero-header {{
            text-align: center;
            margin-bottom: 1.7rem;
            padding-top: 0.2rem;
        }}

        .hero-title {{
            font-family: {DISPLAY_FONT} !important;
            font-size: 4.35rem !important;
            font-weight: 400 !important;
            color: {CRIMSON_ACCENT} !important;
            letter-spacing: 0.02em !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.08 !important;
        }}

        /* Middle Band: ASCII logo on the left (like the chick) + Dashboard Overview on the right */
        .overview-band {{
            display: flex;
            align-items: center;
            gap: 2.2rem;
            margin-top: 0.4rem;
            margin-bottom: 1.1rem;
        }}

        .ascii-logo-col {{
            flex: 0 0 285px;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .ascii-logo-img {{
            width: 100%;
            max-width: 285px;
            height: auto;
            display: block;
        }}

        .contents-wrap {{
            flex: 1;
            min-width: 0;
        }}

        .contents-label {{
            font-size: 0.96rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: {INK_DARK};
            margin-bottom: 0.6rem;
            text-transform: uppercase;
        }}

        .contents-columns {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.2rem;
        }}

        .contents-col-title {{
            font-size: 0.98rem;
            font-weight: 700;
            color: {NAVY_HEADING};
            margin-bottom: 0.35rem;
        }}

        .contents-list {{
            list-style: none;
            padding: 0;
            margin: 0;
            font-size: 0.97rem;
            line-height: 1.8;
            color: {INK_OLIVE};
        }}

        .contents-list li strong {{
            color: {INK_DARK};
            font-weight: 700;
        }}

        .contents-list a {{
            color: {INK_OLIVE};
            text-decoration: none;
            border-bottom: 1px dotted {BORDER_SUBTLE};
        }}

        .contents-list a:hover {{
            color: {CRIMSON_ACCENT};
            border-bottom-color: {CRIMSON_ACCENT};
        }}

        /* Classic minimal counter readout */
        .retro-counter {{
            text-align: center;
            margin-top: 1.35rem;
            margin-bottom: 0.35rem;
        }}

        .retro-counter-digits {{
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.02rem;
            letter-spacing: 0.08em;
            color: {INK_DARK};
        }}

        .retro-counter-caption {{
            font-size: 0.86rem;
            color: {INK_OLIVE};
            margin-top: 0.25rem;
        }}

        /* Signature crisp 1px horizontal divider */
        .editorial-rule {{
            border: none;
            border-top: 1px solid {RULE_DARK};
            margin: 1.85rem 0 1.65rem 0;
        }}

        /* Section headings matching "What's neat?" in Tempus Sans ITC */
        .section-heading {{
            font-family: {DISPLAY_FONT} !important;
            font-size: 2.35rem !important;
            font-weight: 400 !important;
            color: {NAVY_HEADING} !important;
            margin: 0 0 0.55rem 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
            letter-spacing: 0.01em !important;
        }}

        .section-subheading {{
            font-family: {DISPLAY_FONT} !important;
            font-size: 1.45rem !important;
            font-weight: 400 !important;
            color: {NAVY_HEADING} !important;
            margin: 0 0 0.85rem 0 !important;
            line-height: 1.3 !important;
        }}

        .editorial-lead {{
            font-size: 1.04rem !important;
            color: {INK_OLIVE};
            margin-bottom: 1.2rem;
            line-height: 1.7;
        }}

        /* Editorial list items in the StudyMark section */
        .editorial-item {{
            margin-bottom: 1.2rem;
        }}

        .editorial-item-title {{
            font-size: 1.06rem;
            font-weight: 700;
            color: {INK_DARK};
            margin-bottom: 0.28rem;
        }}

        .editorial-item-body {{
            font-size: 1.03rem;
            color: {INK_OLIVE};
            padding-left: 1.15rem;
            line-height: 1.72;
        }}

        .editorial-bullets {{
            margin: 0.35rem 0 0.2rem 1.35rem;
            padding-left: 0.8rem;
            color: {INK_OLIVE};
            font-size: 1.03rem;
            line-height: 1.8;
        }}

        /* Style Streamlit form cleanly to match flat minimal borders */
        [data-testid="stForm"] {{
            background-color: {PARCHMENT_BG} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 0px !important;
            padding: 1.3rem 1.35rem 1.2rem 1.35rem !important;
        }}

        /* Input labels and fields (forced light cream even in dark mode) */
        [data-testid="stNumberInput"] label,
        [data-testid="stNumberInput"] label p {{
            color: {INK_DARK} !important;
            font-size: 0.96rem !important;
            font-weight: 600 !important;
        }}

        [data-testid="stNumberInput"] [data-baseweb="input"] {{
            background-color: {CREAM_BG} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 0px !important;
            box-shadow: none !important;
        }}

        [data-testid="stNumberInput"] input {{
            background-color: {CREAM_BG} !important;
            color: {INK_DARK} !important;
            -webkit-text-fill-color: {INK_DARK} !important;
            font-family: "IBM Plex Mono", monospace !important;
            font-size: 1.02rem !important;
        }}

        [data-testid="stNumberInput"] button {{
            background-color: {PARCHMENT_BG} !important;
            color: {INK_DARK} !important;
            border-left: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 0px !important;
        }}

        [data-testid="stNumberInput"] button:hover {{
            background-color: #EFEAC7 !important;
            color: {CRIMSON_ACCENT} !important;
        }}

        [data-testid="stNumberInput"] button svg {{
            fill: {INK_DARK} !important;
        }}

        /* Predict button: flat crimson minimal button */
        [data-testid="stFormSubmitButton"] button {{
            background-color: {CRIMSON_ACCENT} !important;
            color: {CREAM_BG} !important;
            border: 1px solid {CRIMSON_ACCENT} !important;
            border-radius: 0px !important;
            font-family: Arial, Verdana, sans-serif !important;
            font-size: 0.98rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em !important;
            padding: 0.55rem 1.45rem !important;
            margin-top: 0.45rem !important;
            transition: background-color 0.15s ease;
        }}

        [data-testid="stFormSubmitButton"] button p {{
            color: {CREAM_BG} !important;
            font-size: 0.98rem !important;
        }}

        [data-testid="stFormSubmitButton"] button:hover {{
            background-color: #7D0815 !important;
            border-color: #7D0815 !important;
        }}

        /* Minimal Result Card */
        .result-box {{
            background-color: {PARCHMENT_BG};
            border: 1px solid {BORDER_SUBTLE};
            padding: 1.35rem 1.45rem;
            min-height: 270px;
            box-sizing: border-box;
        }}

        .result-kicker {{
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: {NAVY_HEADING};
            margin-bottom: 0.45rem;
        }}

        .result-score {{
            font-family: {DISPLAY_FONT} !important;
            font-size: 3.3rem !important;
            line-height: 1.05;
            color: {CRIMSON_ACCENT};
            margin: 0.15rem 0 0.55rem 0;
        }}

        .result-score-denom {{
            font-size: 1.45rem;
            color: {INK_OLIVE};
        }}

        .category-stamp {{
            display: inline-block;
            font-size: 0.92rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            padding: 0.22rem 0.72rem;
            border: 1px solid currentColor;
            margin-bottom: 0.85rem;
        }}

        .minimal-bar-track {{
            width: 100%;
            height: 7px;
            background-color: #E5E0C2;
            margin-bottom: 0.95rem;
        }}

        .minimal-bar-fill {{
            height: 100%;
            background-color: {CRIMSON_ACCENT};
        }}

        .result-explanation {{
            font-size: 0.98rem;
            color: {INK_OLIVE};
            line-height: 1.65;
            margin: 0 0 0.75rem 0;
        }}

        .result-footnote {{
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.84rem;
            color: #6B694E;
        }}

        /* Formula box */
        .formula-box {{
            background-color: {PARCHMENT_BG};
            border: 1px solid {BORDER_SUBTLE};
            padding: 1rem 1.15rem;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.93rem;
            line-height: 1.78;
            color: {INK_DARK};
            margin: 0.7rem 0 0.85rem 0;
            white-space: pre-wrap;
        }}

        /* Expander & minimal data table */
        [data-testid="stExpander"] {{
            border: none !important;
            background: transparent !important;
        }}

        [data-testid="stExpander"] details {{
            background-color: {PARCHMENT_BG} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 0px !important;
        }}

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p {{
            color: {INK_DARK} !important;
            font-size: 0.98rem !important;
            font-weight: 600 !important;
        }}

        [data-testid="stExpander"] summary svg {{
            fill: {INK_DARK} !important;
            color: {INK_DARK} !important;
        }}

        .minimal-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            background-color: {CREAM_BG};
            color: {INK_DARK};
            margin-top: 0.35rem;
        }}

        .minimal-table th {{
            text-align: left;
            padding: 0.5rem 0.65rem;
            border-bottom: 1px solid {RULE_DARK};
            color: {NAVY_HEADING};
            font-weight: 700;
            background-color: {PARCHMENT_BG};
        }}

        .minimal-table td {{
            padding: 0.45rem 0.65rem;
            border-bottom: 1px solid #DCD7B8;
            font-family: "IBM Plex Mono", monospace;
            color: {INK_OLIVE};
        }}

        /* Alert box override */
        [data-testid="stAlert"] {{
            border-radius: 0px !important;
            background-color: #FCEAE8 !important;
            border: 1px solid {CRIMSON_ACCENT} !important;
            color: {CRIMSON_ACCENT} !important;
        }}

        /* Responsive stacking on narrower screens */
        @media (max-width: 780px) {{
            .hero-title {{
                font-size: 2.85rem !important;
            }}
            .overview-band {{
                flex-direction: column;
                align-items: center;
                gap: 1.4rem;
            }}
            .ascii-logo-col {{
                flex: 0 0 auto;
                width: 240px;
            }}
            .contents-columns {{
                grid-template-columns: 1fr;
                gap: 1rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_trained_pipeline(csv_modified_time: float) -> Dict[str, object]:
    """
    Train and cache the Multiple Linear Regression model.
    Automatically refreshes if 'student_data.csv' is updated.
    """
    _ = csv_modified_time
    return train_and_evaluate_model(csv_path=DEFAULT_CSV_PATH)


def ensure_dataset_and_model() -> Dict[str, object]:
    """
    Ensure 'student_data.csv' exists and return the trained model and metrics.
    """
    if not DEFAULT_CSV_PATH.exists():
        generate_student_dataset(output_path=DEFAULT_CSV_PATH)

    csv_mtime = DEFAULT_CSV_PATH.stat().st_mtime
    return load_trained_pipeline(csv_mtime)


# ---------------------------------------------------------------------------
# Matplotlib chart helpers (matched to the cream vintage editorial palette)
# ---------------------------------------------------------------------------
def _style_axis(ax: plt.Axes) -> None:
    """Apply flat, lightweight vintage-paper styling to a Matplotlib axis."""
    ax.set_facecolor(CREAM_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_SUBTLE)
    ax.spines["bottom"].set_color(BORDER_SUBTLE)
    ax.tick_params(colors=INK_OLIVE, labelsize=9.2)
    ax.grid(True, linestyle=":", alpha=0.55, color="#B8B397")
    ax.set_axisbelow(True)


def plot_feature_vs_marks(
    df: pd.DataFrame,
    feature_col: str,
    x_label: str,
    title: str,
    point_color: str,
    user_point: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """
    Create a scatter plot comparing one input feature against final exam marks,
    including a trend line and (optionally) the user's predicted point.
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.1), dpi=150)
    fig.patch.set_facecolor(CREAM_BG)

    x_values = df[feature_col].to_numpy()
    y_values = df["final_exam_marks"].to_numpy()

    ax.scatter(
        x_values,
        y_values,
        alpha=0.72,
        s=36,
        color=point_color,
        edgecolors=CREAM_BG,
        linewidths=0.45,
        label="Students",
    )

    slope, intercept = np.polyfit(x_values, y_values, 1)
    line_x = np.linspace(x_values.min(), x_values.max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(
        line_x,
        line_y,
        color=RULE_DARK,
        linewidth=1.6,
        linestyle="--",
        label="Overall Trend",
    )

    if user_point is not None:
        user_x, user_y = user_point
        ax.scatter(
            [user_x],
            [user_y],
            color=CRIMSON_ACCENT,
            edgecolors=INK_DARK,
            marker="*",
            s=230,
            zorder=5,
            label="Your Prediction",
        )

    ax.set_title(title, fontsize=11.8, fontweight="normal", pad=10, color=NAVY_HEADING)
    ax.set_xlabel(x_label, fontsize=9.8, color=INK_OLIVE)
    ax.set_ylabel("Final Exam Marks (out of 100)", fontsize=9.8, color=INK_OLIVE)
    ax.set_ylim(0, 105)
    ax.legend(
        frameon=True,
        facecolor=PARCHMENT_BG,
        edgecolor=BORDER_SUBTLE,
        fontsize=8.8,
        labelcolor=INK_DARK,
    )

    _style_axis(ax)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    mae: float,
    r2: float,
) -> plt.Figure:
    """
    Create a scatter plot comparing actual test-set exam marks against marks
    predicted by the trained Multiple Linear Regression model.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.3), dpi=150)
    fig.patch.set_facecolor(CREAM_BG)

    actual = y_test.to_numpy()
    predicted = np.clip(y_pred, 0.0, 100.0)

    ax.scatter(
        actual,
        predicted,
        alpha=0.82,
        s=44,
        color="#2E5B5E",
        edgecolors=CREAM_BG,
        linewidths=0.5,
        label="Test Set Students",
    )

    min_mark = max(0.0, float(min(actual.min(), predicted.min())) - 5.0)
    max_mark = min(100.0, float(max(actual.max(), predicted.max())) + 5.0)
    ax.plot(
        [min_mark, max_mark],
        [min_mark, max_mark],
        color=CRIMSON_ACCENT,
        linestyle="--",
        linewidth=1.6,
        label="Perfect Prediction (Actual = Predicted)",
    )

    metrics_text = f"Test R² = {r2:.3f}\nTest MAE = {mae:.2f} marks"
    ax.text(
        0.04,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9.0,
        color=INK_DARK,
        bbox={
            "boxstyle": "square,pad=0.45",
            "facecolor": PARCHMENT_BG,
            "edgecolor": BORDER_SUBTLE,
            "linewidth": 0.8,
        },
    )

    ax.set_title(
        "Actual vs. Predicted Exam Marks (Test Set)",
        fontsize=11.8,
        fontweight="normal",
        pad=10,
        color=NAVY_HEADING,
    )
    ax.set_xlabel("Actual Exam Marks", fontsize=9.8, color=INK_OLIVE)
    ax.set_ylabel("Predicted Exam Marks", fontsize=9.8, color=INK_OLIVE)
    ax.set_xlim(min_mark, max_mark)
    ax.set_ylim(min_mark, max_mark)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=PARCHMENT_BG,
        edgecolor=BORDER_SUBTLE,
        fontsize=8.6,
        labelcolor=INK_DARK,
    )

    _style_axis(ax)
    fig.tight_layout()
    return fig


def _category_hex(category: str) -> str:
    """Return a muted editorial ink color for each performance category."""
    palette = {
        "Excellent": "#2B6435",
        "Good": NAVY_HEADING,
        "Average": "#8A5A12",
        "Needs Improvement": CRIMSON_ACCENT,
    }
    return palette.get(category, INK_DARK)


def _render_sample_table_html(df: pd.DataFrame) -> str:
    """Build a crisp light-themed HTML table for the sample dataset preview."""
    preview_df = df.head(10).rename(
        columns={
            **FEATURE_LABELS,
            "final_exam_marks": "Final Exam Marks",
        }
    )
    headers = "".join(f"<th>{col}</th>" for col in preview_df.columns)
    rows_html = []
    for _, row in preview_df.iterrows():
        cells = "".join(f"<td>{val:.1f}</td>" for val in row.values)
        rows_html.append(f"<tr>{cells}</tr>")
    return (
        "<table class='minimal-table'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Main application UI
# ---------------------------------------------------------------------------
def main() -> None:
    inject_minimal_theme()

    # Load dataset and train Multiple Linear Regression model
    try:
        pipeline = ensure_dataset_and_model()
    except Exception as exc:
        st.error(f"Could not load the dataset or train the model: {exc}")
        st.stop()

    model = pipeline["model"]
    df: pd.DataFrame = pipeline["df"]
    y_test: pd.Series = pipeline["y_test"]
    y_pred_test: np.ndarray = pipeline["y_pred_test"]
    mae: float = pipeline["mae"]
    r2: float = pipeline["r2"]
    coefficients: Dict[str, float] = pipeline["coefficients"]
    intercept: float = pipeline["intercept"]
    total_records: int = pipeline["total_records"]
    train_size: int = pipeline["train_size"]
    test_size: int = pipeline["test_size"]

    logo_src = f"data:image/png;base64,{_ASCII_LOGO_B64}"

    # =======================================================================
    # SECTION A: HERO TITLE, ASCII LOGO & DASHBOARD OVERVIEW
    # =======================================================================
    st.markdown(
        f"""<div class="hero-header">
<div class="hero-title">Study Time Predictor</div>
</div>
<div class="overview-band">
<div class="ascii-logo-col">
<img class="ascii-logo-img" src="{logo_src}" alt="ASCII Art Logo" />
</div>
<div class="contents-wrap">
<div class="contents-label">Dashboard Overview:</div>
<div class="contents-columns">
<div>
<div class="contents-col-title">Student Dataset</div>
<ul class="contents-list">
<li>Total Records: <strong>{total_records}</strong></li>
<li>Train / Test: <strong>{train_size} / {test_size}</strong></li>
<li>Input Features: <strong>4</strong></li>
<li>File: student_data.csv</li>
</ul>
</div>
<div>
<div class="contents-col-title">Model Evaluation</div>
<ul class="contents-list">
<li>· R² Score: <strong>{r2:.3f}</strong> ({r2 * 100:.1f}%)</li>
<li>· MAE Error: <strong>±{mae:.2f} marks</strong></li>
<li>· Linear Regression</li>
<li>· Scikit-learn</li>
</ul>
</div>
<div>
<div class="contents-col-title">Contents</div>
<ul class="contents-list">
<li><a href="#predict">Predict My Marks</a></li>
<li><a href="#charts">Study Hours Graph</a></li>
<li><a href="#charts">Attendance Graph</a></li>
<li><a href="#actual-vs-predicted">Actual vs. Predicted</a></li>
<li><a href="#studymark">StudyMark</a></li>
</ul>
</div>
</div>
<div class="retro-counter">
<div class="retro-counter-digits">RECORDS {total_records:04d} &nbsp;|&nbsp; R² {r2:.3f} &nbsp;|&nbsp; MAE ±{mae:.2f}</div>
<div class="retro-counter-caption">Evaluated on {test_size} unseen test records using Multiple Linear Regression.</div>
</div>
</div>
</div>
<hr class="editorial-rule" />""",
        unsafe_allow_html=True,
    )

    # =======================================================================
    # SECTION B: STUDENT PREDICTION
    # =======================================================================
    st.markdown(
        """<div id="predict"></div>
<div class="section-heading">Predict your marks?</div>
<p class="editorial-lead">
Enter your daily study hours, attendance percentage, previous exam marks, and
assignment score below to estimate your final exam marks out of 100.
</p>""",
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.22, 1.0], gap="medium")

    with form_col:
        with st.form("student_prediction_form"):
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                study_hours = st.number_input(
                    "Study Hours (0–16)",
                    min_value=0.0,
                    max_value=16.0,
                    value=5.0,
                    step=0.5,
                    format="%.1f",
                    help="Average hours spent studying each day (0 to 16).",
                )
            with row1_col2:
                attendance = st.number_input(
                    "Attendance (0–100%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=85.0,
                    step=1.0,
                    format="%.1f",
                    help="Percentage of classes attended (0% to 100%).",
                )

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                previous_marks = st.number_input(
                    "Prev. Marks (0–100)",
                    min_value=0.0,
                    max_value=100.0,
                    value=72.0,
                    step=1.0,
                    format="%.1f",
                    help="Score achieved in the previous exam (0 to 100).",
                )
            with row2_col2:
                assignment_score = st.number_input(
                    "Assignments (0–100)",
                    min_value=0.0,
                    max_value=100.0,
                    value=78.0,
                    step=1.0,
                    format="%.1f",
                    help="Average coursework assignment score (0 to 100).",
                )

            submitted = st.form_submit_button("Predict My Marks")

        if submitted:
            try:
                predicted_marks, raw_prediction = predict_student_marks(
                    model=model,
                    study_hours=study_hours,
                    attendance=attendance,
                    previous_marks=previous_marks,
                    assignment_score=assignment_score,
                )
                category, _ = categorize_performance(predicted_marks)
                explanation = build_prediction_explanation(
                    study_hours=study_hours,
                    attendance=attendance,
                    previous_marks=previous_marks,
                    assignment_score=assignment_score,
                    predicted_marks=predicted_marks,
                    raw_prediction=raw_prediction,
                    coefficients=coefficients,
                )

                st.session_state["last_prediction"] = {
                    "study_hours": study_hours,
                    "attendance": attendance,
                    "previous_marks": previous_marks,
                    "assignment_score": assignment_score,
                    "predicted_marks": predicted_marks,
                    "raw_prediction": raw_prediction,
                    "category": category,
                    "explanation": explanation,
                }
            except ValueError as validation_error:
                st.error(f"Invalid Input: {validation_error}")

    with result_col:
        prediction_data = st.session_state.get("last_prediction")

        if prediction_data is None:
            st.markdown(
                """<div class="result-box">
<div class="result-kicker">Prediction Result</div>
<p class="result-explanation" style="margin-top: 0.9rem;">
Fill in your four study details on the left and click
<strong>Predict My Marks</strong> to calculate your expected exam score.
</p>
<div class="result-footnote" style="margin-top: 1.5rem;">
Computed directly by the trained Scikit-learn Linear Regression model.
</div>
</div>""",
                unsafe_allow_html=True,
            )
        else:
            score = prediction_data["predicted_marks"]
            category = prediction_data["category"]
            explanation = prediction_data["explanation"]
            cat_color = _category_hex(category)
            bar_pct = min(max(score, 0.0), 100.0)

            st.markdown(
                f"""<div class="result-box">
<div class="result-kicker">Predicted Exam Marks</div>
<div class="result-score">{score:.1f} <span class="result-score-denom">/ 100</span></div>
<div class="category-stamp" style="color: {cat_color};">{category}</div>
<div class="minimal-bar-track"><div class="minimal-bar-fill" style="width: {bar_pct:.1f}%;"></div></div>
<p class="result-explanation">{explanation}</p>
<div class="result-footnote">Typical error (MAE): ±{mae:.2f} marks · Marked with ★ on the graphs below.</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="editorial-rule" />', unsafe_allow_html=True)

    # =======================================================================
    # SECTION C: DATA VISUALIZATION
    # =======================================================================
    st.markdown(
        f"""<div id="charts"></div>
<div class="section-heading">Data &amp; charts</div>
<p class="editorial-lead">
Visualizing all {total_records} student records across study hours and attendance,
followed by actual vs. predicted exam marks on the {test_size} test records.
</p>""",
        unsafe_allow_html=True,
    )

    last_prediction = st.session_state.get("last_prediction")
    user_study_point = (
        (last_prediction["study_hours"], last_prediction["predicted_marks"])
        if last_prediction is not None
        else None
    )
    user_attendance_point = (
        (last_prediction["attendance"], last_prediction["predicted_marks"])
        if last_prediction is not None
        else None
    )

    chart_row1_col1, chart_row1_col2 = st.columns(2, gap="medium")

    with chart_row1_col1:
        fig_study = plot_feature_vs_marks(
            df=df,
            feature_col="study_hours",
            x_label="Daily Study Hours",
            title="Study Hours vs. Final Exam Marks",
            point_color=NAVY_HEADING,
            user_point=user_study_point,
        )
        st.pyplot(fig_study)
        plt.close(fig_study)

    with chart_row1_col2:
        fig_attendance = plot_feature_vs_marks(
            df=df,
            feature_col="attendance",
            x_label="Attendance (%)",
            title="Attendance vs. Final Exam Marks",
            point_color="#5D633F",
            user_point=user_attendance_point,
        )
        st.pyplot(fig_attendance)
        plt.close(fig_attendance)

    st.markdown('<div id="actual-vs-predicted"></div>', unsafe_allow_html=True)
    chart_row2_col1, chart_row2_col2 = st.columns([1.2, 1.0], gap="medium")

    with chart_row2_col1:
        fig_actual_pred = plot_actual_vs_predicted(
            y_test=y_test,
            y_pred=y_pred_test,
            mae=mae,
            r2=r2,
        )
        st.pyplot(fig_actual_pred)
        plt.close(fig_actual_pred)

    with chart_row2_col2:
        st.markdown(
            f"""<div style="padding-top: 0.4rem;">
<div class="editorial-item-title">Learned Linear Equation.</div>
<div class="editorial-item-body" style="padding-left: 0; margin-top: 0.3rem;">
Multiple Linear Regression assigns one weight to each of the four inputs:
</div>
<div class="formula-box">Marks = {intercept:.2f}
  {coefficients['study_hours']:+.2f} × Study Hours
  {coefficients['attendance']:+.2f} × Attendance %
  {coefficients['previous_marks']:+.2f} × Prev. Marks
  {coefficients['assignment_score']:+.2f} × Assignments</div>
<div class="editorial-item-body" style="padding-left: 0;">
Points lying close to the red dashed diagonal show where predicted marks
closely match actual test scores.
</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="editorial-rule" />', unsafe_allow_html=True)

    # =======================================================================
    # SECTION D: STUDYMARK ABOUT SECTION & SAMPLE RECORDS
    # =======================================================================
    st.markdown(
        f"""<div id="studymark"></div>
<div class="section-heading">StudyMark</div>
<div class="section-subheading">A simple exam marks estimator.</div>
<p class="editorial-lead">
A small project that looks at four things that can affect a student's exam performance
and estimates their expected marks.
</p>
<div class="editorial-item">
<div class="editorial-item-title">Four things go into the prediction:</div>
<ul class="editorial-bullets">
<li>Daily study hours (0–16)</li>
<li>Class attendance (0–100%)</li>
<li>Previous exam marks (0–100)</li>
<li>Assignment scores (0–100)</li>
</ul>
</div>
<div class="editorial-item">
<div class="editorial-item-title">How it works</div>
<div class="editorial-item-body">
The project uses Multiple Linear Regression from Scikit-learn. It learns from student
records and uses those patterns to estimate marks based on the details entered.
</div>
</div>
<div class="editorial-item">
<div class="editorial-item-title">Model testing</div>
<div class="editorial-item-body">
The model is trained on {train_size} student records and tested on another {test_size}
records. In the current dataset, it gets an R² score of {r2:.3f} and an average
prediction error of around {mae:.2f} marks.
</div>
</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("View Sample Student Records (student_data.csv)"):
        st.markdown(_render_sample_table_html(df), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Embedded transparent ASCII art logo (inline PNG data URI — no extra files)
# ---------------------------------------------------------------------------
_ASCII_LOGO_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAgQAAAEhCAMAAAA3VUi8AAAAflBMVEXs7OihrtRgbpa4yOvu+P+03v23zveOmLJ3h6mm3f3RXaft"
    "pNSXqPNlnvOpt+zBxuf9rvn4pPhSX3lXaOH6sO+zWpOdpKR5jMYrZdAWWqZy0PnTjrW5veYMIV8QKKAXJhwAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAA/e4gcAAAAIHRSTlMC///xTlyd//8q//knG1rtXCj/FJn/JP8TEyH/kRAPD9n0RVIAAFeVSURBVHja"
    "7V2HYqO6FpRQQwnFJYlL2v//5dPMkQDXOGX37e5Fd68TGwwEiVPnzFFqHr9mvM+3YB7jWmjifBv+i+NJPVm1tuk3y3/z+I+OJnhd"
    "96Hvq17r+Xb8F0dUTdVXlU7zn17CLA3+Y6ogT7ipdFWlF41/UbmsJ+bx7w+rmmIHmOjc3qcX50zkZ7M8+G8M3avQd0EZvGmpGKKK"
    "1qVPtAvG+G6+R//4WCtdGygBF5IZ0A6fb/3K6ZC26MrNd+kfH61KFkBV68qE0E43uGQYpI/TCvFz1OCftQTKL73xOv3zxhzs0OFD"
    "lz7W3trX2Tb4F0efzEBljDPbsh7ak3WyLcvEK+M6M9+0f2x0lTaMClQvSr1e3G1HOzForZJuiLv5vv0rA7kBq0Iy+mj2fRQafLLr"
    "wMhBPauEf8gShPxvQnL/vNY+3HOmr5oPIfi0b3CzJPhXRrQ+h4e21l57uJ+GxWHt1qYXFfcxje18D/9qj0CvVKV7iHanPu33ta4y"
    "eqVDaOY7+TdbgxaZIfr+n18DyZCodVjNqaW/XRP0ladBqOvuC4/zS1oEdTqAmuPIf+fYWsxcVNtoo90r+8V5tDEmyxL/wbicb+vf"
    "M5JxX3ldmdqobVYLXznM0ysyS2khmOBCOpqeLcS/aOigal1VyaTD8yuRofdvKPZOAgzGXokyzePPGk0AUCS9hBohQEuV/vU1YIE9"
    "qZNtGXbtfHP/ClMQehsYEWfT/7sfSQqvEWRIR8v+xXq+y3/geMrPvwEmJCQFri76g9/QCa2KOh1f93r2GP/MFYAHFBkis0qawF+e"
    "pdeng+98TiBYLbaB55HmpfDnjbcIsGiapL4ycfrMry8smq8YnIJLNXYGnfxZA+jQPU1444EWSA7dYUhgjX1iWRNvah3fvvoQdyYa"
    "57xDCNqqeSX8KQPz7p02BjJ6bxHTseuTPZwrL8Zp5/VXjTusuNY6LdCk+e7/ISOsAA1UQVf2KO+bw3vbULvKmMrBZPDyZvUlUVDK"
    "V50F3CBphjj7CX/GSGZaLbUjF2K6u1B7bHY1FkH6iczgNzVQSCsgmGrOKPwBIz2I0Wij0z+XSwmPzD67ay02cx8gS/Freonf0+ca"
    "eNRed+oMTnEev3W8dMop56mod8cCvlOtV8mEM4jywBywzlmDIJJPX1LGKb/9rrdgvcKh434WCf+n0SgGh3V9pkjgVfWrJPlVpX3S"
    "FsGs6qQ1Kh3SIyz2g6OCCKHh4vkqV0XLIsa6dqt+Dhr89ukX+wwZojTLZyx9q2gGplmGBZBWwAbmgIBHtaqBL9CqcrpWdvv29Qsp"
    "oGRVzeD0/5tNqE2ahdAr+3r0MDfKB6MDLIEaLxuxCeRNbeRnlb7tkFtqkWH6ilbo0unTcYMPminrOYj4u0fSx86iiNg2MM/eDiy0"
    "LRw5a5lG4qvCL/iA72WLtRG1iNFu4zp+SSfYNtqW2NVkZyCCNPuMvzdIlJ7muhUfwRqUkk4eQyDKfLIWgC3MOmGVVEd6UwefPkkf"
    "15pvVnQaETz6ylMs4iO6ytM+MdUcRPydw/qq11WZOJf0e+XWEz+eUQHWHkt8KNkPtfZ1MgMYOdKyhcujUvW38oLriOPD5vTVrA9+"
    "43iKntNXxALM9Oyyt+lpDHVfM5sEQxCF6HiTvIQqfyiPrQnDlg5Jha/aqU2Ho/Q4/mwg/tbBROE5/9zGJgcK8MaKLYBBfImD2Mdn"
    "pKbBNnziaB58YxSDhAcdNctMfPNrh0ui/bgo5K0PLuhOWAfSTKxDcgRcXZs6GQgO7mSfZHZ6kzQBtwS+MXwT9Ms3i0zWwKPWvn4J"
    "YZ6fXz1gx1v4+sd5oDX0e4irAVNWFcMACSRJMWyKgVC2bFQ2GbVafS/yZ19Q9UyT1MYZkvrrR8eY38tx5KDqgTY3JaGILE+hJzM5"
    "zwTQaHrZlC2b9BLMygAwEoBL/TKuvCHmRBPiGmcT8dc7B2mm4klNkW065Xrnnahlq+ILbQHiQDAQWvCCQbXGAZCqkEqQjz0L2JPv"
    "/+WckO06Fb0SQOr12td5/IAgsC4YZdf6pddH5iJuvjYqdCbYRtk3u1YxOKNtMC44QEUdko4AlyCPgDfBwXk0PkSt2/TmWyrdtuBA"
    "0x6WRj/P1K9zEBEPWtXpAaZreGypO0edYOr0ewsLgtEAV4thQFABGEwRJ6Cj6aAg6OaDkgCftN8L+gRJUOnZQPzF/mFQmkrfr/zp"
    "ZuYMgQreiZTmm1qmPG0xK1/XHnECTpavAvJAYjr4WvffNBDTIvCZJHWeql9qFESjtrbpG9N3drcfAAWSAej63nhvipeAt6bV6RPA"
    "SrCFv3gjn2BPoA/xg9s1YgclcvB5zd70L13/kowO89KOIQM7xw1+VBtY5Y01sW3tS5NsOeOSfjcvp8Ek1XfKxK4XE42pJqMcFoPV"
    "Mu/pBW+clzeWy8H36SUJmrRlnLhPzJ9FHqmhcRp1MhCU9vOk/fzYwalLPgC8QYV6A2b1j1N4uwgX0UAzZz9irQQBUBlV5UIS+c3I"
    "T/HwNophBI9Kxq8WNOfhM1XWTJn8CwZSPxrQkbBTsd6I/j3pXmKJLAphHZXUmKt1CDlYsMnBIky5LpEjGIgbvSHsqOQm7DcMhNYx"
    "GDEbiL9ipLk0yQ1sOw0nDIxkRoeYnLODyMG7Qig4bdklLyEvkECIaOgEfWr4z6cXAFBIWOYFthocvrjP4n335WoVYFlskj7p2uKc"
    "Z/7R0Sprk3bfCieZYwJoT0TpwTIgdT0a3HTKdTYn/99gIdiOMBNAU1RGmUgUyeZ65nT89Iov8Oe3kaRO+XSGeRn8ZKSIHpi1PXRC"
    "8hNtVHAACfs8wzJlRL5PhDJrFQRzQISBeI4Ze8B9TRIwG7iRcBpN9fIFzND7aKOKAVKbeRX83CpQZCRLjxcnlirXgKsO0I4zi0Aq"
    "kCbE9ntba5dBJYwgOWR98KZOb2hH5GSTBl5Fq+qb9r3P2ap57n5ugJGsTm5cmtjkuwVZCaYGiPScEeaBR+WWvAjaWGN3cBkAcxII"
    "QcUbw0+wrwd8Lf0sm793wTr0vGR5NycYfyRUxOc7K20rGRugQ4ge3e+UjdPKoLhVOWizLjTm6UtbK5kGJQkkwk7IR2KJNfFWcKkY"
    "PiNSvnXJEjiwqpMg1NPsM35TEKgggX670ko1tbbbmEsMpOy0rnrI96Q3xlvd4HlkVkEYjGjrQQJU8Acq+onyYiqL4xOWHmAt1ICk"
    "kBDtm75eDGRXrdTciPEHBEGZ8to0cPsocKnNoSCyEWaqATgoJpp9qQg07e3QDjMyPqQlgYR2KBudG+PQQExvsjbH/Fc85NdnLzma"
    "sGXS0dbzEvj+aEIBjYZ9o0IdWxsKgjQIUCSZ/6v+KIjYVz1KlAP9SItQsKSW0svKhFoDWZIBKKxkrs0q4JNauNCwLL4DQNs7lDMn"
    "y2XVvdE5nce3HW8nGFJR44gJWhezWqenH1XJGQwSpG2wdW1zFBeRg4iaEaE6kyCBFKxIjQp/5jCCYFPjABV5/6z0sgJ7BbYl2jmO"
    "/P0RBBsCEz6ApqROflxM8x7R0U4qzhwAArpPmwdF/iRu+prxQBMmrDUIOaZPPDAnEAcMJBOUagBE4dF43ORo+OkhPyG8SrsFk+xW"
    "wGFnjtzvL4Li2aN2xFUK8YH0gIk2r5T4i6QaCyeRA11Z0e+TTxwOKZGjDQ0EtdHCblLqWJhnYqGz+3wyoIHlsoJJUqPcKV3yaq5n"
    "/9ZoC2yDFkBW4CwB2zvyktRSaiLaXFddO/HNrbCbpM0D+OtVpdkNGW6EnKQBFUkBp5pNncGpcDwkMVl91tlPl8xiaBwgvQE/qp7D"
    "h9+0CLx3nmAAYEg9/u/knnpuMoo0VY7vkqkQ1ZR1Tlis0Ad1bRumljqvTNLTnkeTb6UjA3WChIT3/Im3PK0nlvWqnzdsQdqI8FV8"
    "sYu98n5tI4kz5kXwjfHUKJDTWKPT1KabSdAoexx6HU1HC2zNaBF7HMrmpMjbiYXY2tZuTc8uiDDyss/4on3fGxzNa4BTMx6VACTu"
    "S+ARX/yLjtp/gExuNFGtXk5eHFMV03KakSbfHGa1chCwTtmiuENtdxYiPYyosAI3Sb5+vTpNLdlKLIvJ4mCUAaklfid5jnUQc4Bq"
    "xtSAm8BaqJlvGk52eRGgbBYXuDp47COOMoMMvhcs6laVR7db1aX14PXGiJ0X9QGAI8ax3mSD/GL+fDmYl7KA2rGWUEvO0HPK0TwP"
    "s52rV5wcRQ5ZEKrXDYGty/uaozqZutareRF8a2yVh/BPgla9dboXcGjfQR2YqZTdaZHqZC0ThdFMHXR9/wIRH8E1IR+zImEi9PFF"
    "L+hTvmo//Cag1EtmAeIWClhIKJd0bfpl4hG+RqJa54n81uhIOw/isvbtRRlvTbdPt7x1/tQ8A26YJGfEo4LP7I2y4P0JpgPVNEnO"
    "QIlqWbsiAScjxqc1FsdXxlhXDETgTwhSTYfszzAmpZP1aYvyXSa/Jcdqc2jXmtkm+KY+WGm7yylkifpSOcMmOOtLgI+Evp9Oyn4l"
    "ktk2RS94iQaEwZtrlIBKarcyWZpLnKBgT5KDqepcwnBGCmRoip1aIQMi/glRylXhSp/Hl0cDm9AxG8dsT1GwF0hGc2qxzgjSY14S"
    "Fq3nANCTTFgIgSuDoSHmJisYilwrmRCthumAZOZxVIihp1orBLAuGrYzwuQHIoZBuXSPvVUmY0EwsW365OwiQJDYlIByCIjcT6DJ"
    "jgcYZozlAsSukh1NvpPpz7ScTDjQ8FOHw7mkckjflu+E7fmQkuOh52n87thFJIi6FkEB8JDxbl8pCLd2H3Npsmo75PCaA8uBX7Ut"
    "LQ2V6UZISyYppQwxYS4p85GM/yvhRVlbF3GOPUxXeyWahGCRt3ZmxP3eSB7+CoHeJLir2KEZbq5SvZrdk+oCZI39qj6nzVXW5tll"
    "XL+wsR7RgdwSVslZ9CsIoteTEGFm0WZBzAeDp5+9g28ahoV3rIVNANPO5Vl+Offw5Y8yk0hd2MvOhH57pKOYWnrKi61ARGkYDMXL"
    "1cnBGwBdVuTN0zcsAuBU5rDx91zEUCOeB0uAOO5845s0Bde+xsbZJqeJDJ/YdphvHGANEJkebLb4gk7LRvhwpdkuUad6ZEvqskiw"
    "NrkM2Fx/DEpNnixBMTPz5TdFgWXRiXrbWqEjkzCxdecewuX0qZ0ARlDDMujlUnBKspJG4j0gK12Xb7rJgz/YHshMqX3MIUer1C3d"
    "kQRdMoNKvukhRoj0JJSTOabrnbKmlrur6w++yce2TYK7FsIyiIWgTmuOATlHoYIrIsSFs2mC1lQ4GiGIb426DW9EOKSdk4jfG1EQ"
    "IFWl6ZMr62nkNa25jRUiFx85wZaefmddgKyD/AZj6tlHt1tJcaz5BB8FcDABrNzz+N4iENWehDqKy5WpoaQbq6tbWILekGcSOCmT"
    "QXDvdsc2h6SJMgkVgKb1JA4VB3vwnhDUOh/lprHWgkqZ9cH3LIIobOWgKH8loAgvS2ja23zvTrXbpFQi6kq8j4AXxMNGNq3qmoYG"
    "AkuSm0ONwU+k1V6DHdynwMNOULAzbcl3BgAiAXBPALT6AGMvQLYGZ87EaR/uJ5bhgRW4BZKUrXU9e2Kc+JW2CZ1HuxR1BArNHGhg"
    "SfvK9Ws3MxZ8d2T6jx7EoTEnEJK9/iYe/vEDtnh+vl8uz5mXoC2pFMtO67NUY7ZjAikcO/653AnFy9KX81NjzfjTjDX+3tAF57GK"
    "EaASuwa6w6qVPpdAWt4/3t0/nNEpyfyryF3jc70JQw2Th/5JeHH9yfJoaZIwm/gFxd4icLlS8yr4XqjIFCDoWq27rklKmdG31njf"
    "ndhb9v7u7u754Rz1VPfS+YxVxdGUlKtOasS6vnvpm07bpgQX+XOdPjFmb5TvD4MMt0kCk87XzHbh90bhHYMecC2IH5yBaQgSsjO3"
    "9v5O3d3dT5fFoXO2J9wE3XRBLOO0e7g/UQu67U06ftev+17pbDPEr03kOp3nmGttHp9XB174xhR1NsC7qy6qbtoGY7oI0hq4u19c"
    "WVQCHQgOHrxmA9XtofGghcMkKSAoiFV33JH3szaBq2bD8NuGoQdE1KDpgTCa+kpK1M+CSu65Ch4uH2+PShQfpMRESPBPl0lGGHmA"
    "Svrv6XOaGbab9cG3biLhnnDPOlQGJjPAoC8eoKH67dIiuL9iqQEEgjJEFhui1ODQsthGHrqAS0B4+t3r17Mk+N7Ys9stc0VbkJOQ"
    "sJ6pISBG7JlFcD9dBGfcxVypgPgTSoucSWbfENnvyYsv9GaCIvlu9o8olbdZEHxr1CgUhGuoRiCol14o1RnPSxbBsDoWD+cTeGvB"
    "nPRCNbZq7GiAQEGQHc2NS+YbIzDAMM/jd4atCQROirsRchJUJfexU3jTnJMED3cwCkQCLO7v7i8sg2RUVCttVV+hwimfbKhXRr3q"
    "z5BUC5PaHCf41iIgR5lgN8A/JthP0tboc0X/ySagPpCJf7h7vLu/XxyvAqyQHtDQdZr3MEGB6tDjvTTd/SHvhkxqsyj4+mgJ9iRw"
    "VBQ5tawtSv2ci/hsx0WwxK93Z0OImU1EjhSRYprYDD9KLWJdOy+BT46nURFL0B5loQD8r0vjElPtmFI4GyfouQiSDEAGePHMVXB/"
    "1lmIitCR1iFoQK7LVT+K7a4yP5P509WMMf3m/avJQg41TcYPwD7TG8QHLsQJ+vsFjQKKgmQR8PfF8oKkFkpEQQslM8CMbBTNT9FK"
    "EBs7i4KvjV1rQ2YwJu0H+tqUME/fgInMgJ7u1CZYDvpg+Xx/eRFYlwx3UQKlgW4V3ocDbusfWwR+Th99TY9CUVuXecZgBkTBCYCQ"
    "VLAk2Vo4swgoCpRdqsVjWQRnzxFtnh1EC+xEG2G83YQive1vGfkq5nH7wxO8oEJZHoYgsa/dm+0D0ofJRcQ9fanN2UBcWgRqQSfR"
    "Lu+fHy4vAnCUSWlhE9BqLRyGBteoevsZ7yDO7ZC+MPwqo0IRH6i5CFg2UAdrQTsqQSRzFu6ZvAPJJ98vklmoLi8CoTelGYE4gdP1"
    "waSv07b3H1nR1Qw0/vRYcsrBOJr71q1Mzhl1yMY1lQCBG2zxZ+ME4hneP9w/L+wVSbDStZQlpUNGG0N1ADvc/pRhqFfdWs2y4DNx"
    "gQgYsSNFGdgh+B+0M3HADbwBJ2zn7HjbnVcHS/EPnp/tlUVAmlSeT5qpuUPs6av6IeZBQErm8Qk1ADipTY4gpPQa9450Mo48NDE2"
    "IU3UTg8YwP5MTG+RZj5ZDM+Y+ruHa5KA1Day8oJpGnsMI9U/hAQxeqYm+JQcYP1fctdH9q+qUEX4tk1a3O5ilaf+1R4HiwAIvw9h"
    "kfWBel4srywClytNn5QEjY7rTW6oNL1phJWePYNPDLKQgxVkIIFtK2lcBJug7TI9aL6n8WwsbgmAoRJ9sEQCaUwkHBts6WS7iEoU"
    "8t4ck99UevsjfxSzVLNFcPsi0L7XHhjAop5b0xuhljQoTEc3s/QGvMZonWsuenELwouWyyuLAO1zDefZ4rjRH0rtt/6nbAI/h40/"
    "Hk+K2E8PAHCJr9hxqjIrrVON813Xg3ZW82MvpGB63aijQA9SyI+PmHp7eRG4npRia48VZht1tAjW/c9UErduNgluGuAiXaEeBL7U"
    "Onc+VtkRBBBwwBaOjKZ+QJi8nE0mPj5CLVxZBJ4tc3L/Oh+TOjjAD7TVz6SSI7m4Z6vgpkWgdBgqzcf03VaoakMGmgotmdQDJSdi"
    "5dAYCdUA8dRLoDV4TRLgKHGrSGTiMFkHezU/ZRgiWDSbBDeoTS1NbE+jM23uPmEy0LSXxrbyCUoU8Yk615PWLkUtXDYMcYBGfpq0"
    "GPRB2LiNPwMq2c0u4ocKs7MOna9LnuhsQEf64oq1pgpVicv5HfKAOBcv1CdflgStEoMtsiceCuCPUkg/Y85tlZv5KS6OdzroScmH"
    "TAzuQ/b9ng4dNfasSJI/qmQGWNDR9SQUG6Yd5CErfV7iLp4fL0kChqXXAloIo4vYULmEMyyYXxrJclmZ2SK4GkgJelX3GUEcwimd"
    "h+UW4Eh0bG0VqPxfqo58dlY63YO25hJdxRV1EFaBaUiamelYU5uAJPg/sQjS9dczwflVZQAQJpnCcstb1pK8ToQFW95qaW6jLAuG"
    "G+VIKNatVYdFs40uhL6mAl+qk0TN4v7+7vG8d4CjUBKkM8c073WTD6GadQg/w0GK66+dmjkqzmgCoYBhnzsqZKtaIEUs6cTaA53g"
    "0PqOLvsWVaG2dZl+1A6pH3XQCO8ogLh4uH9+vvigvkpUQqpZphtyU95vSwIYL7M2uDBeAt19aTkhCTtL1x1qoZ04aiCU2imhpN6B"
    "gxTtJvEGzdQjGalEunNLdZr7szZJg1P6EqiBtmOcwJKYtK5Hi6Qho+n3J28rfujMaHvhCanZpj63uSt3CXwkIHwb9ttXOpOY9sj2"
    "WCPFJ54IkNzgVsyBvvYbXXhLbxlNBqew/Qm67NXHNsFPOIi0aWZRcG68pikIpPUShKcQVqvS+G4f99k46AJ228mWznaaeUZwiCVv"
    "gYQjRiYLjXOHN7epa+6L/nimsKNNrEZcxk/YBGLOzOPs2LO1bVRtjE129/l5stTandSXiNJXUoxq3ySeDM0dM4skthAzbvc5fNAV"
    "qtFPxOgQhIjonesmUjv+kG+PBp7zZF9cBGhKUO70mm8Gczz2wbEzLqnjYDVIHchbQBvbkL3BQCayJCXQQBWEpFxMTQj7oG6pBX8J"
    "WCpPFpDwVtUHzXC3OvyMHu+C8WGOGl+wCSIa1LWDLaAV9XwuAzdICrgcZKmrkCYWj3+TK5CEcJi8ouCSajKXiW2a3Iym1xnceQ0p"
    "2uWqhVoFS8aSCUluDPXP6PF0lX01T/clISwFJfkZgYFAy49pxKYvHWZh60vju24bXxwVbKg5waQT0cmkb2gbBLPaqrQKYtbvH7JN"
    "NsSpbJWr02XE7tAmWIfqJ9AE73KV5qP1+J8dRk0S7V2nvH15ya3q1s2L6l4axPZRgWIQ5WeVqPS19SVhoIQXrAEd2db3Ej5ApzOT"
    "D32tELSxvpN4g3Nr1cV0jIl30PqfUQfspLafZ/uslETicOQZycVfyfh3yBx3hVbOGBu808ruVKYac9QFIer0Hq0PI/GiBhRjVoU+"
    "HcD0JBrbY193LYGHPrgm2RNeo0cSjzLqbvNDFr1H093ZQ7ywCpjDP7Lim45xgiFo3wZpkM7ITrIaNIMGMTI+wAa3abqRWqg9WSZg"
    "ZpiMSnAZoXp5AljcIk1x28YzZjVxEX8o3o/6lqo9E8mcB8TkqnLbvAZe7WCrCWrEDt66p8nIOa1E84NSCPOf/Hi/8rZlTC6547GN"
    "ge1KxCa0IJwpb87Pj0KlW7UxuJAG2b5hU4teSz+k9DbSc0XNVPenFpM1Z5oC9UYIqIf3mkRiyehns9rkEHrtrZUOtmhs62wSuJ5b"
    "VOt66ZDLvELHTui5MuGcbZBsjdIENx3/hU1wR5PAmB+BmVp0aWUBwywIzmrL80EUySZ51XfNS2szwaxXW7StjTQH02SlN52SfrgN"
    "rEO3TQZh53okpJJ+71XbK6wxYIn7ZJupXr316kAo77yUOZl0QDAc+2m82SSd8yOz5tUUPjuPo5mOl0syGpWbzfkinCnaQ713EZ6j"
    "tLZP6yFUlv3L6OGHSrrUwbJAUqgSniC9Mrm17nDyB4lMmNL4TIG9atqyTq9c+BHnviYv6kxxfj5Os1NXegAsGy8ZAjOaadKNGAkk"
    "ndsYp0XgJYHELVw4NYuXPQI0PP4+zSwTVOnNWLzKxcc0YTrURteIR7wcJB1WWFc/4CMmi7VO19LNM35eFCTv7RJj8MJ6cA8MzeaA"
    "GiGhWKuikIwGr5PraNG2ltDT0KhXaWcbOlgSafRcZLbryEQGxjN/oA/kkMJbukXueOIT8mTff3w7NuvQJvTzhJ+7PVviSa4ukzTa"
    "PQlFfW6ApxoaeC6zlpEpdMSa79n9rtCR8XXL7FKcUJG5qDL1ofTYs9adI0FzP6LGrcqUq0TEsIfPPPWT2xNCX9WuPWsQDD+70sbW"
    "AVOUVsOQO4h7sBltSD/K45mqI7+R5SPO3QBA8cJ+mlZQm6QIFIMjKJUqw+dmuPTj+2nSv0BPv+kdrkxNbna1ywbIcbvt//R4BSld"
    "Vbn1R08SCed97bLiiLm1vXXRoLedgFPFdNRmlZaWXa9Dxqkg6oCcVIGgvikGmCTp1CqBswgaCQ22Jkn/Vh12xv3iaGvp4qe2bPML"
    "W6ae65Mnz3pOD320JxQ3utFa6WHn+Sbd0bYNmYM4uQxIzRg2rRXkeGADvE7KjusJNiRzoRlaC7UIDMOipv3OTQ1DjxZJ3xbd6dAh"
    "dMUA4bXMDbEGg4D5H2uRIHj6KNNO1tEY1bqNtrXbjE8dSMfY5H5LMGeMtBiWTC7hW5niDNCUNX6PbHcr1sIaUBLaFGy2BxsiHniw"
    "30WHAuzSwRhp8uEyH1tUyc6xc9tk3JQjJOFlkSovQI2gehTfaeAiRPQh0Ln1rdNVX5GsfkvaMw/R0dZwCGr0L0vPXzB9nZsdNrt2"
    "naz2DfthooFuFgfTOAEcji+TEyABkmROPa6rd8QsEfYMvUmKIUmIbJ3+twMI+npu53AZWCWMo3WR2c0qZs3vqhDJWC9u2B4rgsWr"
    "O6QHKxukukRNoR2tbV/Sqqlq9MxNi2BgTJ1M4zGV2adGsk/Bnu8POiU0a9ZSMmVlKjLhLP/jwoAwkJv2pFiVjkSwIyyojHrBo0p7"
    "mp5GeAW//jWalaa1kJ47RBl7gktCmvK0CGSWdxIS1PAtJv3x/NQmAJB19eW/bYFDwvkpgmCgx2yq5BSZqp6hp4TYOOcLReltIrZl"
    "XWq3pXJ1ardz0ZHaTNrQOhdpYyRFHx0DCDFaD5IQK7o/vbxlG0PcCTQ2gRvPFEL6d1h/6mL7lYChNDhJZ+hyqOF9ug6atzdsQKgz"
    "Wzp/61gejy8c48mWXOFnHWdroN+9XlNGeGSDPYAobGOLAvds1BOC6jS6pcgWQ44yHzy2vLAYnlRpcBFrG4BXGU+SDvmFJbDWBpFJ"
    "vfvAM0IrxhaXrO3fCjlaLhYPx2PBgZ+3utCUw4jsfFb1hpqVIgwBx9ILR0vMh1mFUlCisrHXw+e3oD2jZdHnaFPOPGn5DtjR3FQd"
    "fKFcpE1LSa/qYw60C2shObuqdn9pBHEZHu8uj8f7G5eB2AToXKs+13PQUIHnDKTJ2rw0LkoGokQFLIvZqOfZ3hBeQnQZp4R6M2Qg"
    "hUAdoBLJM07iBF8hnYtKSFivGRNPg3sUVubvXQTq6iIAadwtQn0NNAcS/gZKvbvFbx5Qh2h0azr6WDgK3/p0NGdc7wr5VN/73FXV"
    "c4ChymIf4Eg8kEoeV/DiLa8EB3CjzWL959R1w7gAbAsc6OPAh7INr8V9q+vm/9PHXxyNQR/g9eYHWlvjmgZ9hNHtNn4Ixmw0tLlm"
    "nAdXYeSGI/yDKBFWgpGJJOq0z7HifHwYY1qa7drkR0buzrxR5PyTCG1ihn7OUiHBnjftXqqrP9w9XXrLn17/pZJgeWwbfsXhbark"
    "o4lNkFvaXPcX31tJ/o+t5XLqJ9cgx6G1PXkvEACg0dUmZb/yoUo2ANuckQNNakSryrdwNIQDK7hpoUK38p/RA6ECCesNtgDKqJ5Q"
    "NLOK23UnhRb/4VGtDFM/DAAw5mOv+2Q+A00HnSo2YZfJH3zGFrOvEWMAEjSWROIqOZAG99xy5VUEmFa+kchOL51v+lEnmdUnWO5b"
    "q8Ua+RA6ki2CeiO8/ZW+r5b/WkJpcZ8Uwo1CgThRlZnD+AKO8bW6yP9KomtfyghaxVa3RpcbD9QpeM1cEuVpS++zZMG3+vRPAZwK"
    "CIkX7xCfZzVj+vR5euknLqK5BQbymjMRjmecHuCiVYN/O1xG/5bOnE7c/JWgo/cra+ARLQhvWNpPsMuEe8w56y047YHJ7BpI90ur"
    "IO2BaiGDAJFHG/uI6FCbvgu4RpPvs1NyNFprPYNFCN4gIsTYEP6TtrteojsGH27Xatq90t04NQYsqBmwtD0MBZzTgZ16aaIvy2Hr"
    "f4og7Y8SBI93WAY3GYcha1BJ6Irr5sKVUv4efpza2bBiz9x8pyMVbFIDRZRrvKG0XdMBzCmAyOPX6FO6VH60H4AtRIHBKkzs9Ljz"
    "q+4G96CAXn0R9e8fmA4SphgMD1zG7m9phmHt/cNNUn75cI9+pDfIgpgb11idbYKNLhki214wJXt8J9Yr9swdDLcQ/FBP/KSKnZc0"
    "fwyyCCQoV2daHAaM2WxXylrRTAecVYfJrK66CRWY7RTNM79fNgLkT4065zyHv4hwlr/nEX++ede7mxYBkrXyMNWG9GRICwsg9IKE"
    "NGyKu06rBinkoCVusO7YQHdkFgCvme5rvVXphvc1zgMYIuCe2K2zaY1hKkBctlWyBbwHYXd0sg8jF0mdZWhrf1PFcYz880aEWYNr"
    "8H/LIgBf+K3+48PdnfpQH9j3jNiImb+MSR55SXq7sW9ncJ6ljb1FQzwrzCKl0y1ABo3aStrZLqlzBYbKTXFAoCb7QbAdePOWlAOX"
    "3BoGaTwXmroYHISdsRXf4MM1v46qIEwiOfplbAdY7N8gCJ4f7241+hcQBQ+3SIKalQKK5kCo/CqUCH4AMaVeARQ8TcUmmwCic52E"
    "fYeIjAC2JgqWHFi6UXZB17ASkjMCDUmMlHfzjExER2luBfyBtPNRUPu6ZQh+9f7mdDCtHbU6PuT2eqXkn2UR3Crhh71vWATbPqlG"
    "26dFgMZ3AI1WhIOQcIT0ZJGgoqdBrTbJ2HNP5KdPb6LTG61qkJPIcyuVrDqUFtigpQuu9ul5I1d+EvrpeI3LgGO73+8dzDRNm6A9"
    "wv69nGfPHwQBrlfrcPMiSDrHneSkmup8S78/0yK47dkuuiMtmY/32um6FyBo0ANAhKgbncFe5sjX6iu9lztqJC6Q953YAiNOxZKW"
    "rnYr01ozYEttZz1Pwt2iLWdMazLUB2e7QlzWqNw3pw/hFs3BjKmh3DoxdXH6f1AQsAHVTQaEPLFNIS2HF58LQchgPuz1nkOCtuhS"
    "bGd00casm2XHbfovvo1X2mHSlRSxWNeWkL6QiZVqFJ7xlD3AXggNKlKdZWqz5pa/0W6h1WJsj50FxJr+miRiWQS3erMPty0CXVCB"
    "CgjS4IMhaNQT7lfLGxcaPRB/+VpsODxWxo90EoF9snQXXshdZkITQqzduBlmhqbFn950mf9Coot18i0cHsbmUB3Evj7rn+DC8FTf"
    "CDoydczHPx9wrJz5OxDofLIXn7Bhb5QEKMTgHZDUj5WqEo2of20lDyB1JqPl56EOLCs43MrFnYiHpAbUBiznL9l1F4SJzU5bQPEJ"
    "ildBhYrjC/KIb3DmmgDT5CLa+DRVB2/nQkOuzvGnm8y5Nxqrl3lRG17YX2MRfCZJeJsk2AaUd0Cw2ozzCASIZAMh5Eb2OqDLQZtt"
    "AquemnX64gaWJJG8rQN+O39Hval94TCRG2+7TmfGVOsNmM7Qe7tplSvmAFhQg31b1wdRm/1pYcxWWiVxEdxaZJz/iPB0aRFMbJo/"
    "XhB8ZrXaGyVB59CpgHylznWd65JIcAj8W6A+EQIQNx/PrQAISljFW++JHBXM6J49Uxx3hNngXOxeui6XsMYOGRqEEPK3XuB5JpsQ"
    "KQYH9GoLAGomQh3+iKPGtnYPQwJYVJcuo/mwcapdy+kNyDb8JZqaFn/s9m8QBM+Pd/eflRy3oYuAAkWLK0F4tlukh4EAIV4U4NHe"
    "JQ9Cl+a1nZbaYY8MIHYL4CbTUXsXgOoJDXJ56R9F9TpDUMGY27DYxZDkLM211k1wqG03aig7arSfRAyBEdnaIXDcsHUv6FOtpYF3"
    "9aFoe61C95IMkA8jz+m44S9YBPazguATDqXQkpFwAhXkrXqzyAP4QigG/YviEJGYtq9knogT7SVzI/3x0necmnzRDruZwUCM0kYv"
    "2C05TBA5YG2CEkHTHBTCoMta9yTAdEp1v0mGB67lhoKhNuRKl2RrNhfjjWLlmLoKf34vBAiCh89cJVJIdx+GjRvxCgITMFD2W6nK"
    "yuQuBlGgZLNBTqTZljxgJ43sLZ3unpkb8dYRAFTRrwQoYqoXVPpYmhnEEcnlc62EOlkRUNNy6ElcN9QT8LcccvCH5FoQ0rzJFgjs"
    "2IQL+0gUVqGu9D8oCGBDnBOXT82pX01xTXyfyNh2i+6nAAoCPAqhnzGiMlFajIIOLGNCZWZchoiiJZ7pjHRMfSHMD0DT3vddJjzf"
    "at/prtckPHUEnnozsGuDG+1AHfRCZNYokmoCQZi+86KbW+4Fixw6vf6QDTMK5PFPzyRbCIKrO8gYMIYWmIKTZfOkuq7rfXppdsfR"
    "FJpPnU83pO3FWrLsmQvchwNh2cA1mw3DzticbN6DMNZCgSvvW3oaxkXQhcHcyE9t2iOtBJMLj6wi6hS1R6SbRfVK38X+EEXCLZjO"
    "rvM2LVGhwV4f/1HH4q1XTfdqorM7m/5dMR169WKazigsYms65bvmTy5Mfbi/YhXeTwa9SHvPnlOn2sDWgTUeqCd9Or43buWpAQQ0"
    "+joocHbBC4NNIIQjSeWu9GBU0Busa8QWZFnkAAONCUUoTyWMloUOg3BCMJU4yy31Sk8RJjmGQVCiQBN6XHdn3z56qnOJrVuZj2MI"
    "0g02X1gxVtRfOZbPj+OgQ8AOhM/HNsQTtSxrPHrkXF7tsWos2ryXYK50jOlzY5xcvLzzzCI+qTo5DHkh1VJvshn0+46EIDAmNpk2"
    "ztGyHDjkGiXLpE6LgGTo6bQbacY38RERuIQkGi/sYlh4sgxgmmbCnCsjLcZ9vdFDz2dmTv1fk028JgmeF2UR3E/XALxrMgJIE1y4"
    "RKdtqrARDlvy2tgCRUU39MHlFj75u33I5ER6CBjIbsSfBlElbJSRT4bEwdblo3Q58I3uW2AtixYx6sDvanKVTY0VbLGW3+VJuvMI"
    "k9eyDsiA4XGULhjVfWAUWx6XgBScSU6/u6Rl/njfoZgExShANaKdBAl2W6ZwOmEQizkx1HXWn+RZmDeKnU061TFV5MYOdSCf3KcP"
    "RCYPoFIhH7NWMk4tOqc6dMzDN3z68jqJD1uAJ006+A79eHNahzGmTFvG7ne2K0Gd1luWKqu8VW0/cnVQIa2IF7Ef5BQ7FftkYwJL"
    "E9NFotnfumHTPwmDqX9g2MMwkWWNYA/XvcWUxSE0oE+YIh1YRDVq+eoiRlhpalSVea8x+VbqDaJqqdLTXWQNoiAIeVwBgdtcophU"
    "7zr75KUXb3bm5U26/SuWqCahX57gXY3LoGa55W/uCVdMZkb3cVa5k+rHCuWy2SagnRLY//1m3o6/aewAJ02Clg2OJWOgbfbwVdwd"
    "6Vb2xwPCpBqtBVp+SOYky0n4aFBIpAOZTZEcdEwTek5wzQwkIWZIFyWTIeCej61yPMhpdI46AIhU45A4AGtH+nawCTSyFyhUefuw"
    "uMpaFkLdNH2dJYiFZgaNFfD2K5LwqgKH/tcWwd6y9a0J6GVNZUdUKD85JJB+YoxP2uUWKtGdJSxTko2BOaNGvSse4BWd8jy2rAsv"
    "GHwCHyZValqSzCJXOgJN68Fwk80h7NtMiKb92BRXDpn0+02SgJfc30RLZrEvLrXn3xoEaHof5GTmHyQ3i9adkoWKs04ToKQHT6yD"
    "/IsinHSC/nAAdb6PJieWWeYiI9ZjcjbbinbG0ViyKtZIHAR2CdZKc1whPynhrF3Grt42ttKv56bddzm0Mlgj5fItGE54V36hqj6s"
    "DrN2AAjgni4/hAvYqzLxDFkJupprUx1SgqHcB/m8ZHlJvUnInt0RjV0LYmNmmpu2hHVF6afnBUUi4lpV7CWxr4lLI+FExRpEC2nL"
    "0vKG9NJQA7AaKHn84Hra/NCXglQ4jfLm2t/6NHS+Xnewdix8y9v56HCpyVUGE6sWPN2k5oWM3uZXIQzs/fPUe18iuDN9+/x8/+H3"
    "r0WKn59P4kMVtV990HN0iIpF8stCD9dn3Slbaof2g4Cg4qbmhFkgWjydIe5tLjFJs12rmsz1VZVbjai3pPk3SFBAZifTO25JZUW8"
    "SsfCx3SSOkdrnCXt2U31IKJSkoFjP0dyJgTtJTUC+2fybQ3WHP+rWuo+3B0UDixYJbQcUj6I8CyuJoXuHp8vyoLl4vnxcXEk8wnX"
    "6C82pvOIIML4WpnLm4uX0OHJq4WLlGaAYjNcvGDmajbbRcqnJi8Y2MtM1vwvlbwZW96OCJaeJl/IqNZQ7Z10xu1Xt1KSQ5IlefUJ"
    "1741K8a36hzspKR8Gjd6oml+xSJYIp5/N5YIIitU8kLvkvu9niZaIBa8vLw1rTB7IC87KfqAi94NAZBJyjRO2EO3UN/74+ibxAPo"
    "eWf78Y1VKs61a2xJxkB0uV4l/e+lamWwQWyuO2EdqiP/aaTipvs4cKMv2W/Po+WJpQmD0MNNUXzmqNbqE6ZDsZNIpUbcjOXP/VRS"
    "AlejvhE7XF4VBPfq8b6YBUkQTJODi/R7etIXl4+1gKS4dHz7cH+QbV73IXOIWQL9cw4PXGVmRFBEMJYzhpq0OZhHT2JlTZuMudrB"
    "cDe82Z7Q0ODR2zbS/0SxOmvSCCZB9M9sNunO4pBeYnEaBNdwRoEOC0na9gJNiVuUKjraGPggBp2PfwPcK5kOgW5q2zbqc/y00Gna"
    "bzZ+g6tMkkAfpyAMrvKLsmCRTL/l2UVBQXCPmeI8Ly3f3qW3yyIX7k8yxsvJQljeX6s0Ps42C2FDEnNTzWoBtEjaNtrDmHuDskDC"
    "NY4fQBLbO4j4QgxeOMoQRuhK0kELfQnxogIqScKA9p0R4lC88Hg7J1zz1cBGxACDSSp9ty0GIuAMH7c+CiTJrL8yTaxfNYqXHBgw"
    "OTpbZC+XL+UU7QLG3fLAYl9MNPr9w5DdswCCjm8hze8fFETB8oIseHh+vFKUvHg+QhJhKmD4HEi1TEnui8p4miwPBmZO/+z30gWv"
    "EjHRrQoE9SGvoi4vAiMtdRmLqwv/udgcOEA5dDJAqhGCqiIti7CiqoiVWDDIJn5EJbUOAmf5ArtENLAjyiXD7jmuTYt+9dVgQXq6"
    "H4fM/3KZPMI0c8+DzkZWh3Odn9wHC3GwWOZv3vNtediX4Cl8Ds8XdP6Ja3B3ICbSsw1oh88MEIOp13tJ41upFp0YgYQO9J2Fq73f"
    "HUiKTojIJXDQ8LDOd4tFUcVdL6DUvIU6Nq09/4KLMOka+FI4TPlBboVrhbDKu64DEMEmZdyD5qzznbqGJ8e+W88DfSnMD4wEaFh5"
    "wXzTHJnU2PilZjlLPI6DVE5aOs3MIxdBVAua9raUjC8fnvHTQjzQdErTb8VssDkHNH47S47HK75DshcyIr0TiKgThnnB+cXeMFpH"
    "Phe8oFVhaUA7rea2wO4kq8G0R6kZvghlqS7IXCtAU2oJdsxrVQ8g0oQAjbW/2gulmfYPenIeHM07vc4U9JGsaMZnAWjRG60/ZBXL"
    "YrCXLyZboFHqS1z15fRyN8i3Og0nhsjNX03niYW/yFEA8oXk5zNDwGG/wc2zucw8+XXPFgZCXgyDdXf/zG8/FO0g319+bBF0ScbR"
    "HS/z17ZJM68IEaUyTlpUutyG2r6fDw2UqO/TsSY90OZvbJYnSjnHozNh2dSy2BFW6JLQV/UQALJhIL4sf9O+BIvGnBecxkyb8zSR"
    "TDkD9Y2gXtSrShdjRR1XJ/L4evXV9AHmMmt1Qj3uJwxSz+LjY/bFynuWyX58fJDJx46LvBhUUI93z+MSkKDC4opFAEHCzU1mF5M/"
    "YQ2DPN8yn3NGdTa9dNXEC/GhamIRTONsE21OB91LZC8Oc1ujRHm15qU0+et5N63QWjenn9m0lnP5oqT9wT4vzLguUwEvQYIFzVQQ"
    "tLpiVsF/w4djfGiwWKtRQCkhv0EC5Os5pAfMe47fpd/sGOhdZINwIUixB6kpW8rb/KP8fFLPgd+eRgEuuwZZySx5r0ASBgCoKHAV"
    "W8g+Q0pRI8MLZxgEt2QVpyIVgjDJ5a1S6sQ+z+rAE1yIdut9jzM1qhChbjOV2YudyixSTLJXbhK9ThIVPjmJVB0iitk8s+91l85s"
    "BWuQ2+8+SQezHNtHnavN+JWXr6+B9xb41nQQL8ycQ+9PWjk8r9FfX2NLO8zckeW1PAh1DluzE1h8wcG/XB7g4GERXK5Cs8mtuHtI"
    "EgTcpJOzGrCHeidMs7DFsh2US4sAsgV32GEz4pwqAvbTmeO0k1T0OB7PSCgqqX1DqGgxQW1rmxeSm700A1oc50RimSyi3N3l1FKU"
    "Drm9KpBQVKpZA6CrRV6q8ar3BJy+ofQo+kbZ7zY8tg2sWLDr4jK2o3Gcjt7x1jilbidQPE1VX016KHWm3efTObfsjM7/wCKAFA/Q"
    "5sMZ+AkyROgpgoJRTaqA0mAiNzhNQnl7klhIInID5+7cpbYMDo9A0+6ACtWWM+sxn4xkQwAyTbQQKlmrgvFGvBmQg2ENoCmqH+PY"
    "IEwBsVSrXkiSueqa73PLjNiWmLUR1sAqWS4kP/GZA+vL45eAluEaPFwJFD2qB8VpA2P9IMeY0wFxLGxBLRhTV1Ad2UBgY7ozfwWO"
    "BhKJeOquW4GOgIA8zca+FS7SccptxvGmqX+jZ7QHpiVdmNUrt6HNEYb2y6AwEytkzVTzzkl+kdJ4Z19bZqvSASIb63GxvV56eG4W"
    "BEK/XxebYzB5mNRiM98/EFRyNUYgW0EFpnOAw77KZBEAGrIO1WaTuckE8yk/+XIqgzrGeoOO50LoQmMQEAvY0986gIimye4EmRoa"
    "sEi0GfvZMsZXQKNDTDajWQOyieC2wDWNqI7kYAozWZIMbMPrwvdrhRyo0nA03Y+Fimvl85nYUvfPWwPL50vpQ+SJ1veP+mHBPE5R"
    "lsI+BiiH5HakMazLcAm8i9uhCZ2TIL7Um0ziJVTaNB/OtB+JYj/EkbrMSt4lP2vF3CObvYW+lU+EIy1fUW6QY2kL8lDsolMgHkx8"
    "2YJSaaZn/s4SQCUNr9eWbNoTT0+QibIq/qHQsgvU9byvVXioHgjutFnWST3FyghXJKPxUPIkjixK2e5inR4vXapHseXhnC9F7Kc7"
    "V/TRCqhkVVysXhjJfJHT70nLSlQWWkgQ/ek76VbXAkXzcNTSmc0K7p4EnauCfxlbppiswL9++56G+7UKyepjIuQwDSENWsmy+Vcy"
    "3CbJ+QiQxCBgyU5KsAdYvE1tmf8R0E+tZEttAeQArKJmLaohHOT04C/ZkT5rkJem9+nMMC3fGDSYRtxj1Ss5c19lfEkdovCfJ+2U"
    "qdJ0uia132eAS+54A4QqtoAGX6JNX1fT4xpAgqqWNrnTZykSDgtClupvBJgupeMcms+EMdgmcE9ouNyAtjCRjbRkyd1i6Y/JDCJC"
    "H/bW2AP/hrXHiKtYhGjVNh5mcMrxQ5RnF0hPPrHryHsPsZP0LlCtm1r5rtSeh4EqrXxnr4hcS+5LH3Qn/XvTF8F4lwnRDmqTPrsI"
    "aOq062abAeaG7MrZpQVVu/Cama/lJP/vA3zjWZun131UuYBESMVzT3pPBa8EWCk61mWGstzg1tmhCOQgcWOTdocLsOenR7bBurPs"
    "i2vtNP6RbS9yjDGqBMuBNoZC5MftZIsTPIe1Yq2kD9OOe5KLDm5G0c7uGCf7+bEjupVcq/mKG3HlnBxfoK5/oRzYk8RrNMtB3Byc"
    "4Ls8+pALTlTqAbK4w4OAJ9PIFgNwHzjKqgoIkID0cz1dYh09jYz9DPrM07j3JENOer6OKHSyrCpInv0oUugq0CtsMzlQJ6hTA6Oi"
    "5nWLrGlaXkZeh23Jbesr5GO3xYmTW5HuRg6I7N7lj9u+1IS+uGn1219WgBYDCz0mMRlHmxA6z5CxHOmcmoRvpKXLW2p2oNPTnvUl"
    "t25Gy2w0PQkETbudUDnYMeqfYaWsTcKd7atubFxg6h73GqXldaG1ReyilmuSlSyhZ+vZOU+VNx3cgeabBcOvKgcAhl6pNq9fCZkc"
    "QJv/srbJToCg1WjKe+r5jYA+zGYzspSaumyhTbAZlbIQllXyUxrT7Y5dEAGC6nPgyyfgUY1gSw1ccCEn6af7SpvcSjdtCSy+TM5M"
    "O2XVqbdljlWOtcG5Px4P+Z2CYQG4mOOOStEMd+Hlpybl9/bRsbZtd7tWNTkNsRblRuIx+cnIOFSvLwo440CnQ8meZSspTXO7uIMn"
    "Y7eHvI4SeDhyFXZtuyaQMeZIhF0skWZoBvuiU+R/EFdcdLEdr0axmW4udcFn2+K8FzP1q5yj0bLQZQeClrcCrEUK5A1XhzPHdv2T"
    "k7K4f/hN/belqexgCyyYWkIQD51rK4I7nRGQJ7JkNQi/avbDjeklaQRAR4gfkbggw4cAk9QsTWe32xPwpXiFwJzguGdyJA1BK0Ga"
    "Glkl9elylTncf8/j8hPum9RBsBKlM+kv6nUX+g6nH2SdwPzecEH3X7lTRuC2+Q0IbPYhN/MNEX/MDze7WDw//q4KtowKbXMOkdwU"
    "QisipSZ1Vv/I08DYq/TggftcMMrICCqFNqzM1JtN7nXHmJIUrJzK33W2Oc6i8kl8qgfoSAxEnVaj9QCpDnCna9ZkEWWcgIxkchm6"
    "hLaq8ZC+cKV/Kb/fkuJkjD95VpUUM6DSbvWz/a+W94839yX5pj2ogtDGiaAG+DCJAhLKmVBPPfBJA1recNA+CXvoRpreKz1yiMny"
    "2GwYzBuPP31UmoYtb+tChXu0CFYs5yG72NubkjiEXo0huDUhqDht88ALYwlQtlxwlaTFSQ/sAB4twUjWvn3eHrSvphRf47ZpsYwk"
    "52mE3OxnkwSL++v1Qz9pEKoe4Mjc/ev+7o6Jhb5TWr30yN2DE4ytYtl1lAi/5KvvETJnNt9JVt8PyFCADggAJQ0Z90jfiQrhATu1"
    "sr1peQDgCO1xkL2X1qddvrE8b2eUHYpDOsETJEeyA6jD+Xwt3glVmvcCjx28weaFBluDnfrbdff7k23z1fWdK/xXSv5guT+4gcY0"
    "5kfpKIa5+OWjg2ZLd6Vwy0+oS5t15y37FKMrrQA4TO5KawCVIMinLAJkFGRfq13uMItJAJiEPW0t+UqNbw59BWgJTZhR+k483Aa6"
    "sNamLwKzkwHFfdfpWJoVMkqjXxTP2HPPdNIIu9byjEQhDQURWesAAfI53w0wJqVLVNimFY1rchmvBIysySvzB+MCJ7j/X6YMcvGG"
    "GdfA3WCQxgqhoaHlrTYT6AiiNRD2dUZ4+kxDKW82RX8gwJTjzekAQek6nEJ6u6pkns48SfvoSiUrGt81mRdqFCeqkpCW+LeV9MxN"
    "JruA+5DUioM66Io2rz8luE0+vnDzrSRztqk2VfrHgmT07/3pDmgC7vkN6mCripqLkxMP95/kLoSOpDu7qUuvuNzz0GQ97HWxEgpv"
    "V3+koaU/Yp6SwS4Y0AZ2IBQ711/bAikEvpuWYeTTSnN+wnoW5hndqPn1IcdpiRPQjL3dMHyKrsBt90Liz1sii0DpbJK6adzoRxbB"
    "3d2N3Si/O9oe4Ej2qd1lQbAYYvZNDwCnwDbQjbYgSnVpgitAUOlTC+ynbM6U1NIm1wtTWHkDh82qo/vVFKSqNul5aqdIBGFC6V/S"
    "F8MTIohd5kDbT13cjgcg7pVnyv7cC4GsJoNI322fW9XxAj9zl5Kn2oMMG+gkVS41azC5fSY0v0IQPPyeIIGgHjpSjC3seTJzkJR5"
    "ktUTOiLGn/L8zVuBVSBCZFEJNKnEcZbNbwm1TLtEiTGpNNP93u/fBn/Krq1EYZyleaH6tQX1/FunxmSSFZ44AENUk/ZvBq2+zHRn"
    "Bv+STSA8aUJpgrdjMUiTpc/N3CE+mb6tK+t1q9LVGcUGwPkPBHIFLXXt7ocpC5cQBA+/L164lwi7qVDXfFdAPIcWTlj5jBYB0FSJ"
    "7M42gRMuT7pwzAArNLCQ3MFmVAN+EnRG1OGoVDG+qcz0URNOGNAZ9+Aq9qt0ZnYuTcvGrDyDt3ESWDDFaCmhyc6KAsorwK8cc48k"
    "Obvl/kYAaGALYB3QZ5UGuqVSkowINZ3GX2ARPN7/zpjxrlsZMIw/qkutLWzPCptaGsvUDAhR5crtGBJIiBMNU57zQH3ezeYSoxIf"
    "8if1qp2QoGcFO2r+EmoWzEkO82Re1FG5kFI/86u/FjEXEOwS8BrpU0scKtS3uIhIcylARflk4Po1/0KGzjzMS7FGgMB5/1lBgPKg"
    "h9+4Bp7US+hBu3WvWMJ4PnkqOWPpfysxZGAoSxaR7GVsEAu+Lk8SL4NPCLXML9iNYd0gHF9HMrnp8iHZc5Z7FHUkP0EShjAPKxlI"
    "EhamXKqAe9pHnUyIwjKxVjwavAr4JCbTp+JarkUMkUdJ53xVzUg+9pQMg/Q1iYoH/CS+FX/Rj8BVT+NEv1cQHEigBXrcqhP2d/J6"
    "FryIdTkrVNAkuSXuIT+YdSWlI6wjGX4i8Euh+wL9J1DmU3dFKWFJJTOooDXaiSUZiUoQqGtO4KiBKz2/a1SLohdSonqaG2ThHTvy"
    "2as9r2wX2SoHlVDCmWIzjaLAaCcoGuvs54lNbhEED8+/zyJ4mpIWJAn0+CD0WydB1TelcjRAZzw9hSU5xDLvkIYFUJc4QWYbMjmC"
    "QK6JOkcDcvGiMqIgxpNBpL6uu9KgbJND87WeFCZv96AuJFUpMMO5De9QftYoogUYwuYF6gLw4dWSmI6nv5JLkTboGjjRVvDyApir"
    "CZ30gpGTzh1gZ/h5FuvlkoJg+X+QBPePd4+LJOVUuK/sob5u1kulxZnXm9ySNiq2ufOi8emTg7erZrMb7bL6LAARUnkMcJMMKsn9"
    "67IVNljxQmyPo5XijfH2CPYos6enOULnk+Am7cfoxwtfiBCmDESY4DApx7+SO9gDpxRGEtZkayhZBFVuuedzA2BJo7mvAUeWi4U9"
    "bZJDkbpYLIJ6Xvw/1sAC5PY6CHlX15ym/MgBUhegZpLcoeBI8pbxTVoRtTSmIwDFH6JRTAaVlCdqkll6koieADV5gNwM9yh+R5Iz"
    "TDmsEZAnjkWrJh/XyLdHRlPSpz6V41+LD5bGek61250qf1lVLtkMABr8ZV9penV/fSwW98/3/481kGyR+8XSNg2xnOuTiA4yTVFQ"
    "mvD9xWB2UvSLJR1VVvlusBVoCWScKld9waUKStUOsFQ7pSXFimCv3EwS6uzkMvBTKM+2BJ4Kb9hgNcQcgty2A6xl1CSkW32CrEhX"
    "claGNwSysmIlEgybr9xmCjWibOVPlNNFtf2KGDhoKnEw7thm4v8iBEp8otjU22QXoz/IkdVLx4toEZd7Fvjg2+jYDDfJyQ0eEW0y"
    "PB1hRPlEk8wU6bekU2v0yq1NLd8R3lJsNqEfIf1go4n5JGiC3gM8qgr2BLpqJ3xj+KJtSZfUhN6ELikAeTRBSIk6ufInvAXjg7RZ"
    "vlQZBkgKFJAUl7dsoFv+mPKywbUkM6ZlF+evVjBdlAHPz6AoVX/AsO1arVB7rE9bu1m2hnd64AUj3BOwP3jpqFdOX6wzGiWnfjJc"
    "1aqBymGFYFHthbc0M5lCQh9UKYCeLhlf2LfHGX1OLSTz8S0tgz1DD2GodS798dSBZTFsFiaUnEA6VyNqm6rkATyCk8RE4y/SBN2I"
    "TUuwSj/asl81CpcgkSqdJRaTYe0ZPuHfLg6yYKR5HmrbHzkT9NaRgOmF1z3slMvYCi+1x5K0GdSoEvrRGrabX2m6DLUkG4d9qbvr"
    "enC2XiOtBLahC8XMYGCObt6TVHqQ9kyXmvaIzCa40MZ6eKJShkde7IhGmt7X9akYfynsaFLJvkVEMuh8FMQhhTetP0NP9o8OL7Rh"
    "BwpZ1kGXiSmA2TBElpA/rHORv0jawJfv+5JEYOKBm6yR/INkGNRAdeFVKX/MAcSmizwo90SWwg/84VgpnWhqZTMbOfZ1kkYGCjFn"
    "OCZ/kXfZ3cVp7TSyACukJEX2rLlZ82qVYGLkuoc/LXZt+6fP3/IHvi3eS7qrkhHsjhJKSR22GTrSS+daaWJsJG1nMhMZC8VdzuXZ"
    "4HMCDls2gKsyC8dcJFNyTAEm1Z6f5VhiQQM5WDpADGnfl74fZDhykNhMtlXVphkOyiQ318dMiFbmC0hYOSSb7Y7T2OneyfEN7J7Y"
    "8ZIFKEv+mfLH4I+Ig3H0fx6Lq9bjR1vttcjRkT6yOSUj0ZryiLbJ/lZMCpjcEs+vSC8ghBMDjiSHhqqVyd3swW6xYSUrIKKwCaY4"
    "FUQopOn9YdWyr8fdagIcFFzC12LGAuGJLngiQtpCkmr3e4SRhjgh2+QigRQ1ohkTUmbiKcUWwJnvpZ5U/nC9CmEAxZjVdqoY/69P"
    "+lX7cXl/Neh8f5X1+j4cIRsJBDWnNbwNW8QLEDRZ/EII02U0zwSUWk/gJpU0rhH9zshRyDgVFrfUGxgIK9CPttNYVUPCstJAV0IP"
    "oZrIcoGrrsy6wdy0Lif41G5PPp0xTlBoa7RUjQ+R6ir0U96xlxwIIICmoGaTOVhf4m//6vhGiQISTFciCcurW8F0uLiSPTgmtAIp"
    "pBlQqNMtDwIE9R3ZucDN9ApqMTY/V33Prrdemt96YfDC28J5Nv7mp1vkJ5IKjSqpQGBbCFKMSfdwDyoi2xbT3PRuIFhLyt1nhInK"
    "DGuDt5AurCu/oYnvW7Z3gBGY/pnjNWagJLYQXNn9WKIoqtM4zOeSC3ePV0onFme7mU4Wwd31RXBuBe0BzsBtOP5T5E8ggdgLGGSd"
    "j1IjnBnJBEuc37j8BhxoxGaSEA1sooSisp8ym+F6mApN/1rELjI567a1BPjmbspyaFdgJy2JyyLxrqhEXqPtGg29MRA4BTAibqT6"
    "Fwt7grwn6dCqtwXiLNcyXr9X0sW3+ek8gV3cf+17D1cbGSyub73a/pwUyYuj9GHxt46AoHaM9VaZnQlt6ETC1qVZbUYo8tur3O12"
    "VWd/e4SbmCC8qPQGpbhFr04vT+UGty7zprkpmqNbGUPoyyD0XUadZptgWmqbvri1UANo1rMKadlWoRtqaOUayNXKPlzpuL+mt+Hy"
    "/qpk/mARLL66CIAdslcXwfJ8PN1c6izbZJZ48oKtZLIIx61KHlCwuRmhuiFPuWRjgsQJcuM7YlclDZW+6w7qPZ/kTGlqVqXgmahW"
    "PV5FzPznSfMznYiQlnTWyTZBKO22SbuDo+U8U1oe0QY0y8t8JwM7ac3OfOiVbH+BPbgEL/3j/e9fBOru+XKK8vIi6AVO2tuz2xD6"
    "zexluUQL/gHb5OqMSx3JzQQLujEFfXr4MrbS1Wavpj6JHJdNLyZQV+d2Q4YrY2Iz0DTTqvXjIiBDRppI7uOQV+QX4MaCNK2fHrc/"
    "uJzQ7348MmC5BJ7vF18xCxbfUAdLVDbZ6+rg0jYRIB05w96mCAB2pInCTebRGrbL2RuiTz1Zz5z1JcFU2trYoaR5UsAsGBQPOAjj"
    "S17oIQt2q0nWorCgsXuuEKJ0rR34T1Vum6NoDRzLvCb31UhXxea1A/cZ2E19Mk0Kx4qXOJT01MW/158GeICQGJzjC/vFFfTZRbA8"
    "sAmuSoKPgG3Irfd6dQzT5DM7souhLm/lWycQVLhWEgDQZJJY+dLanqI9mQGrunQbPyhhwZYksZv1iW+O2K7O5GkoO111UhL+nkwE"
    "VhoKXLWdFj+OikEqJqkG9lGaaSidSY5E3dTQAAGGRzyxS34qZ8s2BvbLmuRTi8A+JLEzBBZY47i83UU83oE4kjAhbh/IDNSesEyT"
    "ec08utVZgY4IQsNmze9ya/vs+4PqZgX8iN4UpKoZ6om55UyWhpWyfZWxTbimodsiIMKZRC0er98RMJAJTdwelSha7FNN9rVVlTla"
    "zSGc5YcFAfsTPSy+vohuXwSWlOVJ89y2CNQFF3EMrKB1LNvAxqyvn8YnFHEigC8Fj5oeMyvcNexYCzZkzRdAN2uzyYGfGmVcSCnm"
    "ZG3o6wxb3cgWoE7tMXBDA+eZ9pWjaYZ5muyu9Pw2k5eEBMp3Xxs9wNMF74o0MWwBodsxQtJW81qEkJiXsf5Fq+D+jq1r7K9fBGiV"
    "gf6II0phcdH0uxYnOJSH5OmMxzabmlhxNhOECWokU3lNgKoj05mbYjan+yipZik8oRDs8chEaVtbbAvFihSyiOTOdgS82P2kJsQ6"
    "Yevu7GA94PgxX6Ur1Gs2d+aJQpf2q8K+dAwevowjvH0RPCTT41Dq2G8vAhVb1aHZXIVU7zl9qSv2BH0RM6BKJlwnflYQ0KBoc6FE"
    "9LkzriFqsxaqLMh3YVLjbvImhBN9HLdUGaESHAHd1HEnXU4mIr0TktG0u+EKNisrDqB4m6haUJl9ixBK9tGJ6tetggUNw68Kg9sX"
    "weL5+egs31kE43Nv3wi+rHJ3tPdDg6mrBKHTV8TmOikOKtUrmgZCgWt4dowZ8KieLGkFaCpNkWsnxS0kIXtvphZiG0iMklcRkKSx"
    "ZPhswwBA/iKuMAxJIYqpLgcwKrkMnmSj8yVn1MjTr8wUodkdXcTbhMHi4WDePqEOFoujhfbRIri7uwXg6KQPfV35c0V9ZmyCUINJ"
    "xBXoJr6zmTKdFXY0WuYVLIBQ8Kg5DZXpU1nh1J0Y54IFqjcTIOi4zQxMK5RXewDbNFlZymZd99VAzyr71iVBZdUvzxeykdmtwaLF"
    "UYO6xTfCxj+zCESzQptC7RZLYTwMoKoWTWbYutYm0eoQrEdJavbEETdQQ9lK5iJ1aihZyUUuEkWQMIDLcNSpiH5FXoHsqJk5LWul"
    "RgmP+ZqJhHQJ7Q61LtuuZG8s0bDyw7JqNgNeHaNCze/JFS/Z5PCWRbB8SIp9mh6+Ps3LbyyC5dWt5ckY748FNxmqsO4PK7NjwwQk"
    "gKA+wM8CnkSpnQnkG4PKrdHiIG2ujbB/oXEu+tyzEiEQu6rJmwbYgTFCEgYu1TMtbyOa4WK3TJXWjBjAhuRsgRaGfLHh0QhxBEkm"
    "L0dq6VTw+neRhamLkZwL4+Go9/XHi2BxUYqx4PmKzHm8+1RJfBwqgU85GgRh4iccYm7kOk3GHk2GgKoll4nTEcHXQ2pJtvSVUJX6"
    "QoIOXv3jy5CWPGLS0dibogUmx4cZUS6ZRopiVqymfVIr6db7/1gDi8sjc5Usn+8OHtDFVDssywGGnlfPVyXB8xg2LqcBmijzSX92"
    "EeSqkqTnzTZZXwdOdRzI0EVdu2ERjOyntTCe1aK7k7rIIcfyyVjPko+i0hsXd4drfJcti2qkYS/VTBGF6Jl9TVaGm6JddDFABmxs"
    "9/9YAum2Xx5iNCTNcWA8jHgCoAOGfUs/3eMOyUeLoOig5f3BeWxeBI+fCmKtfdOgF5nyextPCjP7LYisouAIwDIWidYUhMHQUnZA"
    "p/pod41XsgOooIzJKNOCRRV6NDzPOZMxdEwmpjQDFshjNnar715UH10PVvXcL82bAl4VejMCBqJv+64z6k9cBGyNuTwy/fIisAeL"
    "oKATnq/ijoZuy+rMIlD3j+HTkUxEVNJkA8dpjnhLc09CQeYo4xttCddwxOxoJQCO9Bg6Jy+tyaBOoJFAW0IMqyA9DVnMuFtyQZtw"
    "iPakvQjCMnZiBCrF62j6iaFowDiKAwDWkg4d5JBEkAIu8/8kpk/y+OHsuE//n3+kH+7Pffl06/kU9mJMKpycZvHwmeqXQeQq6TsD"
    "+lF3tApe0VGHTChqJYXCJEQjF6mU+BZeq7GjDpnC+oqh5SrXgyh9TKLWHwUs29x5d0N6lBLzGV3G3LJMC2CExKc4pEBHVunbT//X"
    "/gTL5TkL8XIo6Sz19fIo53v1XNOTLJfjd78WxEweIBOIBIjsj7e2oRqzfWMbRWlYNE6s32zKG5h/9UifWvhOfO5/fNxfd/hbQEWS"
    "VgB65valHKkZL2Mjre2laD1kOGyoJWXVqz9yjBPy/kuO+oOjUcCAAjx4rh2skUaxL8HklrF86c3INlZeMiSVAl3eH+FSyaIm/Jv6"
    "zLRxNxYwdKb3uhOOMithDV2a+kqRw9DnPKumv6w7xZ84bO5nYVE31HXN6+EjatGf2G4V6XONAsmZh+Xnrc94UPxGXCd/phcyn9nJ"
    "FrTF9T1BK9Gp/RHNnXpYkyUNJl86eFPavHdNsjW7duuj1EThzJ4n8aUOCcwodp7CHxul+OQCJVDupuuhIMgLBvW8KcUnxJxMKlE2"
    "UqgirltdF940dKM9ye6BkWwl7fmgC6r+DUXjVVJAYGRdeS3sa4IglaOQ0fJtnrNfsAgKlezZzUMlCjJDtdpUucZDZwiqqgk3IaoD"
    "pWJiLbgM+nDSoFf7+hT6G0k8X0tnRiwufpq7lQltjarK6UFDSmM1xHnKft42yBxlPTA78SRs2Xpph2Fyd93MgaYz41kBdRoThs67"
    "iDLnPhokRNPChTYGhbPSj9zNYx/2vhJRAZYxsq7poZkvWBUQlhZGtfU8Zz9vGazFLmDTnG2zLvXCJbTEHE3LlJGXxrs5TcT8jc/Y"
    "jvy79NQratv5TGpSnJ+npZjMyRtpuxZpKnbLgTWRfmvLImlKeovxI/5zwoE2mwG/bIgnU4kc1pU+zPlJ7AAtdcWZFzipF670KnOg"
    "1dmD9HooazQZe6Ir2xz4oNscekC0N9pylINARisOrBCiZdCKEwKEefzCEXVQGa5hz3mnWoCgbB9DVGidycEKB9rwJmeTfE75ZCrc"
    "d5sXXGvJmqKkbFpqqPv6GAvdSuce6cIjQanTJNQ8fnisheSsTvMx1IZMDISG5GbCURYy7VnGnGhpL5shqJmhhAgW2S0cHmsXdYaV"
    "ZsPTM9R4eDogy4ZevwgWhtrPc/SLx1MWxY6gknPkoe/jo+oKsLPAUzPU1LqjhuOT5rnjb0N3SssTxSsR7khrBSpgzy+/zxP1u9yF"
    "nFWouvbUDosqDLxgbK3qc+dVYAtN5jUjdZmwKR8ocQIBc7fbzNIujd4vyvnZE/i/+QtDA/tXiojDpRCG/nVmKCThT1sPxOnCr37S"
    "gjxK/5GMJGaGqQrz/f5Tw0cEcJwNy3Rj03tTuEJX5c0EVMKeuaHuBv8DhCSCQ5XIZMdKpzAvgj91vChBjsZTlyw20eWeu37SZ1fe"
    "quEzqTttXg4yxy0+FoAolANSAV0z3+0/N4AEE491BD4cmQaNxA2CdwFwkLo0UbJ4KYRl6etb1AUP1HZkF2OFk07agv2MolJz9OfP"
    "XgfbBv11VWVW7txM7VZ6pD2TzieF3WTojDuRARoNdClXcuOdME//H+8tPr2TY2TkCj3ZC1ykOfMkJgT5wwU6Ylbbg9gzbc3c0S5K"
    "T946zlLgrxiZb+q8J9+TkMoUWqpMG0XeMww1popYeqyBL7DN2kp7XGPmAPDfNCJBo73XZ7cJLRlp44gtBdt6DvWAbqjTsevxQWMJ"
    "EcJR7Ha+qX/fKghjcelEY8hwKyddZnKbnUMZD5ZKklhaCTD8ita08/gtI9R6db7GI7JQuCKYxJBM/MASQFPkTFbfKdSVBpLczdDA"
    "vzF4JBzX8UyrgNgN7XJ7qIzuiPazQ12DASChlUqE0M33869UByrzoAJt6tyhOEfcx25ty6rhSai/V9GjH0YyDbwF622ugv6lJBLz"
    "+OVroct41Es7TIN/rQSSUUMQxGSYb+C/MQj3zGGej/J70j4RxqTJjKbz/fsXxkvupqsmvQ2eLnsUYLmBBYAuutpMKQpmWMDfOxpV"
    "2E1IMHLJ1WvgBKIlrrCIpNfWqtkS+FfGGo2KXhVLDHyVo//2oEpuTcKJAJJqgYjWsxr4F4dtVgInbU/SwO/Ao6rK+Fo620ljnHn8"
    "e2tA9XXudXfcXepJeVfQobmlrq5n1Mi/OnbAm6BcxB6vEMJRIiAlKBJoo3qdk4X/6uilMzkLEsvCQDWZdNEl45maQUP/8sh+oc6N"
    "ceRRb9vA0MCGtKa2KtUksyD498a7fZd57axegQd9pTNi3EklSiUt90I1Zwv/G+O1a9aNsnspJBk4zdp2XgD/tWFyu1wkG9HtdjYG"
    "/oMjs5dtWIga6vmG/AdHGwprWTVthjuP/5a7mDulgseye9HzDflPjqjQ1gYZJTubA//hMYPG5jGMOTQ0j39y/A9t+C/RqenUbAAA"
    "AABJRU5ErkJggg=="
)


if __name__ == "__main__":
    main()
