"""
Shared helper functions for model training and evaluation.

This file is not executed directly.
It is imported by the baseline, Logistic Regression,
XGBoost and final model-selection scripts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# Column names used throughout the project

DATE_COLUMN = "Date"
BTC_CLOSE_COLUMN = "btc_close"

FUTURE_RETURN_COLUMN = "target_next_week_return"
TARGET_COLUMN = "target_next_week_up"

NON_FEATURE_COLUMNS = [
    DATE_COLUMN,
    BTC_CLOSE_COLUMN,
    FUTURE_RETURN_COLUMN,
    TARGET_COLUMN
]


def get_project_paths():
    """
    Returns the important project paths.

    model_utils.py is located in:
    experiment_1/scripts/06_modelling/
    """

    current_file = Path(__file__).resolve()

    experiment_dir = current_file.parents[2]
    project_dir = current_file.parents[3]

    processed_data_dir = project_dir / "data" / "processed"
    modelling_data_dir = project_dir / "data" / "modelling"

    report_dir = experiment_dir / "reports"
    image_dir = experiment_dir / "images" / "06_modelling"
    model_dir = experiment_dir / "models"

    modelling_data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "processed_data_dir": processed_data_dir,
        "modelling_data_dir": modelling_data_dir,
        "report_dir": report_dir,
        "image_dir": image_dir,
        "model_dir": model_dir
    }


def load_model_dataset(file_path, dataset_name):
    """
    Loads one prepared model dataset and checks whether
    the required columns exist.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file was not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    required_columns = [
        DATE_COLUMN,
        BTC_CLOSE_COLUMN,
        FUTURE_RETURN_COLUMN,
        TARGET_COLUMN
    ]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"{dataset_name} is missing required column: {column}"
            )

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])

    if data.isna().any().any():
        missing_values = int(data.isna().sum().sum())

        raise ValueError(
            f"{dataset_name} contains {missing_values} missing values."
        )

    print(
        f"Loaded {dataset_name}: {data.shape}, "
        f"{data[DATE_COLUMN].min().date()} to "
        f"{data[DATE_COLUMN].max().date()}"
    )

    return data


def get_feature_columns(data):
    """
    Returns all feature columns.

    Date, BTC close and target-related columns are excluded.
    """

    feature_columns = [
        column
        for column in data.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No feature columns were found.")

    return feature_columns


def evaluate_predictions(
    model_name,
    split_name,
    y_true,
    y_pred,
    y_probability
):
    """
    Calculates all main classification metrics.
    """

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_probability)
    else:
        roc_auc = np.nan

    metrics = {
        "Model": model_name,
        "Split": split_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),
        "Precision_Up": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall_Up": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1_Up": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "ROC_AUC": roc_auc,
        "True_Negative": true_negative,
        "False_Positive": false_positive,
        "False_Negative": false_negative,
        "True_Positive": true_positive
    }

    return metrics


def create_prediction_table(
    data,
    model_name,
    predictions,
    probabilities
):
    """
    Creates a table with dates, actual target,
    predicted target and probability of Up.
    """

    prediction_data = data[
        [
            DATE_COLUMN,
            BTC_CLOSE_COLUMN,
            FUTURE_RETURN_COLUMN,
            TARGET_COLUMN
        ]
    ].copy()

    prediction_data = prediction_data.rename(columns={
        TARGET_COLUMN: "Actual_Target"
    })

    prediction_data["Model"] = model_name
    prediction_data["Predicted_Target"] = predictions
    prediction_data["Probability_Up"] = probabilities

    return prediction_data


def save_csv(data, output_path):
    """
    Saves a DataFrame as CSV.
    """

    data.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return output_path


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    model_name,
    split_name,
    output_path
):
    """
    Creates and saves a confusion matrix.
    """

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))

    image = plt.imshow(matrix)
    plt.colorbar(image)

    plt.xticks(
        ticks=[0, 1],
        labels=["Predicted 0", "Predicted 1"]
    )

    plt.yticks(
        ticks=[0, 1],
        labels=["Actual 0", "Actual 1"]
    )

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center"
            )

    plt.title(f"{split_name} Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")