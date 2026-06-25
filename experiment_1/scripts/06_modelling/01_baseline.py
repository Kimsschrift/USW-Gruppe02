"""
Always-Up Baseline for Weekly BTC/USD Trading Bot

This script:
1. Loads the train and validation datasets.
2. Uses a simple baseline that predicts "Up" for every week.
3. Evaluates the baseline on validation data.
4. Saves validation metrics, predictions and a confusion matrix.

Input files:
- data/processed/05_train_selected_raw.csv
- data/processed/05_validation_selected_raw.csv

Output files:
- data/modelling/06_baseline_validation_metrics.csv
- data/modelling/06_baseline_validation_predictions.csv
- experiment_1/images/06_modelling/01_baseline_validation_confusion_matrix.png
- experiment_1/reports/06_baseline_report.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model_utils import (
    TARGET_COLUMN,
    create_prediction_table,
    evaluate_predictions,
    get_project_paths,
    load_model_dataset,
    save_confusion_matrix_plot,
    save_csv
)


# 1. Define paths

paths = get_project_paths()

processed_data_dir = paths["processed_data_dir"]
modelling_data_dir = paths["modelling_data_dir"]
report_dir = paths["report_dir"]
image_dir = paths["image_dir"]

train_file = (
    processed_data_dir / "05_train_selected_raw.csv"
)

validation_file = (
    processed_data_dir / "05_validation_selected_raw.csv"
)


# 2. Load datasets

print("Starting Always-Up baseline")

train_data = load_model_dataset(
    train_file,
    "Train dataset"
)

validation_data = load_model_dataset(
    validation_file,
    "Validation dataset"
)


# 3. Display train target distribution

train_up_rate = train_data[TARGET_COLUMN].mean()

print(
    f"Training target Up rate: {train_up_rate:.2%}"
)

print(
    "Baseline rule: Predict Up (1) for every validation week."
)


# 4. Create Always-Up predictions

y_validation = validation_data[TARGET_COLUMN]

baseline_predictions = np.ones(
    len(validation_data),
    dtype=int
)

baseline_probabilities = np.ones(
    len(validation_data),
    dtype=float
)

model_name = "Baseline_Always_Up"


# 5. Evaluate baseline on validation data

baseline_metrics = evaluate_predictions(
    model_name=model_name,
    split_name="Validation",
    y_true=y_validation,
    y_pred=baseline_predictions,
    y_probability=baseline_probabilities
)

metrics_data = pd.DataFrame([baseline_metrics])

metrics_file = modelling_data_dir / (
    "06_baseline_validation_metrics.csv"
)

save_csv(metrics_data, metrics_file)

print("\nBaseline validation metrics:")
print(metrics_data)


# 6. Save predictions

prediction_data = create_prediction_table(
    data=validation_data,
    model_name=model_name,
    predictions=baseline_predictions,
    probabilities=baseline_probabilities
)

predictions_file = modelling_data_dir / (
    "06_baseline_validation_predictions.csv"
)

save_csv(prediction_data, predictions_file)


# 7. Save confusion matrix

confusion_matrix_file = image_dir / (
    "01_baseline_validation_confusion_matrix.png"
)

save_confusion_matrix_plot(
    y_true=y_validation,
    y_pred=baseline_predictions,
    model_name=model_name,
    split_name="Validation",
    output_path=confusion_matrix_file
)


# 8. Create report

report_file = report_dir / "06_baseline_report.md"

report_lines = [
    "# 06 Baseline Report",
    "",
    "## Goal",
    "",
    (
        "Create a simple reference model that predicts "
        "`Up` for every validation week."
    ),
    "",
    "## Why a Baseline Is Needed",
    "",
    (
        "A later Logistic Regression or XGBoost model should "
        "perform better than this simple rule."
    ),
    "",
    "## Training Data Information",
    "",
    f"- Training target Up rate: {train_up_rate:.2%}",
    "",
    "## Validation Result",
    "",
    f"- Accuracy: {baseline_metrics['Accuracy']:.2%}",
    (
        "- Balanced Accuracy: "
        f"{baseline_metrics['Balanced_Accuracy']:.2%}"
    ),
    f"- Precision for Up: {baseline_metrics['Precision_Up']:.2%}",
    f"- Recall for Up: {baseline_metrics['Recall_Up']:.2%}",
    f"- F1 score for Up: {baseline_metrics['F1_Up']:.2%}",
    "",
    "## Interpretation",
    "",
    (
        "The baseline predicts only class 1. Therefore, it can "
        "identify Up weeks but cannot identify Not-Up weeks."
    ),
    "",
    "## Output Files",
    "",
    f"- `{metrics_file.name}`",
    f"- `{predictions_file.name}`",
    f"- `{confusion_matrix_file.name}`"
]

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(f"Saved report: {report_file}")

print("\nAlways-Up baseline completed successfully.")