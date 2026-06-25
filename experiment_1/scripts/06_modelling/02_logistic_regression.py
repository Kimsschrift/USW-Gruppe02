"""
Logistic Regression for Weekly BTC/USD Trading Bot

This script:
1. Loads the scaled train and validation datasets.
2. Trains Logistic Regression models with different C values.
3. Compares the models on validation data only.
4. Selects the best Logistic Regression configuration.
5. Saves metrics, predictions, coefficients and plots.

Important:
- The test dataset is intentionally NOT used here.
- The final model comparison and test evaluation happen later.

Input files:
- data/processed/05_train_selected_scaled.csv
- data/processed/05_validation_selected_scaled.csv

Output files:
- data/modelling/06_logistic_validation_metrics.csv
- data/modelling/06_logistic_validation_predictions.csv
- data/modelling/06_logistic_best_coefficients.csv
- experiment_1/models/06_logistic_best_validation_model.joblib
- experiment_1/images/06_modelling/02_logistic_validation_comparison.png
- experiment_1/images/06_modelling/03_logistic_best_validation_confusion_matrix.png
- experiment_1/images/06_modelling/04_logistic_best_coefficients.png
- experiment_1/reports/06_logistic_regression_report.md
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.linear_model import LogisticRegression

from model_utils import (
    TARGET_COLUMN,
    create_prediction_table,
    evaluate_predictions,
    get_feature_columns,
    get_project_paths,
    load_model_dataset,
    save_confusion_matrix_plot,
    save_csv
)


# 1. Define project paths

CURRENT_FILE = Path(__file__).resolve()

EXPERIMENT_DIR = CURRENT_FILE.parents[2]

PARAMS_PATH = EXPERIMENT_DIR / "conf" / "params.yaml"

paths = get_project_paths()

processed_data_dir = paths["processed_data_dir"]
modelling_data_dir = paths["modelling_data_dir"]
report_dir = paths["report_dir"]
image_dir = paths["image_dir"]
model_dir = paths["model_dir"]


# 2. Load modelling parameters

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

model_config = params.get("MODELLING", {})

RANDOM_STATE = model_config.get("RANDOM_STATE", 42)

PRIMARY_METRIC = model_config.get(
    "PRIMARY_METRIC",
    "Balanced_Accuracy"
)

LOGISTIC_C_VALUES = model_config.get(
    "LOGISTIC_C_VALUES",
    [0.01, 0.1, 1.0, 10.0]
)

VALID_METRICS = [
    "Accuracy",
    "Balanced_Accuracy",
    "Precision_Up",
    "Recall_Up",
    "F1_Up",
    "ROC_AUC"
]


# 3. Define input files

TRAIN_FILE = (
    processed_data_dir / "05_train_selected_scaled.csv"
)

VALIDATION_FILE = (
    processed_data_dir / "05_validation_selected_scaled.csv"
)


# 4. Helper functions

def validate_feature_columns(
    train_data,
    validation_data,
    feature_columns
):
    """
    Checks that validation data has the same feature columns
    as training data.
    """

    missing_features = (
        set(feature_columns) - set(validation_data.columns)
    )

    if missing_features:
        raise ValueError(
            "Validation data is missing feature columns: "
            f"{sorted(missing_features)}"
        )


def save_validation_comparison_plot(metrics_data):
    """
    Creates a bar chart comparing validation performance
    across the tested C values.
    """

    plot_data = metrics_data.sort_values("C_Value")

    plt.figure(figsize=(9, 5))

    plt.bar(
        plot_data["Model"],
        plot_data[PRIMARY_METRIC]
    )

    plt.title("Logistic Regression: Validation Comparison")
    plt.xlabel("Model Configuration")
    plt.ylabel(PRIMARY_METRIC.replace("_", " "))
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_file = (
        image_dir / "02_logistic_validation_comparison.png"
    )

    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot: {output_file}")

    return output_file


def save_coefficients_plot(coefficients_data):
    """
    Creates a plot of the best Logistic Regression coefficients.

    Positive coefficient:
    Higher feature value is associated with higher probability of Up.

    Negative coefficient:
    Higher feature value is associated with lower probability of Up.
    """

    plot_data = coefficients_data.sort_values(
        "Coefficient"
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_data["Feature"],
        plot_data["Coefficient"]
    )

    plt.title("Best Logistic Regression: Feature Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_file = (
        image_dir / "04_logistic_best_coefficients.png"
    )

    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot: {output_file}")

    return output_file


# 5. Validate configuration

if PRIMARY_METRIC not in VALID_METRICS:
    raise ValueError(
        f"PRIMARY_METRIC must be one of: {VALID_METRICS}"
    )

if len(LOGISTIC_C_VALUES) == 0:
    raise ValueError(
        "LOGISTIC_C_VALUES must contain at least one value."
    )


# 6. Load train and validation data

print("Starting Logistic Regression modelling")

train_data = load_model_dataset(
    TRAIN_FILE,
    "Train scaled dataset"
)

validation_data = load_model_dataset(
    VALIDATION_FILE,
    "Validation scaled dataset"
)

feature_columns = get_feature_columns(train_data)

validate_feature_columns(
    train_data,
    validation_data,
    feature_columns
)

print(f"Selected feature count: {len(feature_columns)}")


# 7. Prepare model matrices

X_train = train_data[feature_columns]
y_train = train_data[TARGET_COLUMN]

X_validation = validation_data[feature_columns]
y_validation = validation_data[TARGET_COLUMN]


# 8. Train Logistic Regression models

metric_rows = []
prediction_tables = []
trained_models = {}

for c_value in LOGISTIC_C_VALUES:

    model_name = f"LogisticRegression_C_{c_value}"

    logistic_model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    logistic_model.fit(X_train, y_train)

    predictions = logistic_model.predict(X_validation)

    probabilities = logistic_model.predict_proba(
        X_validation
    )[:, 1]

    metrics = evaluate_predictions(
        model_name=model_name,
        split_name="Validation",
        y_true=y_validation,
        y_pred=predictions,
        y_probability=probabilities
    )

    metrics["C_Value"] = float(c_value)

    metric_rows.append(metrics)

    prediction_tables.append(
        create_prediction_table(
            data=validation_data,
            model_name=model_name,
            predictions=predictions,
            probabilities=probabilities
        )
    )

    trained_models[model_name] = logistic_model

    print(
        f"Finished {model_name}: "
        f"{PRIMARY_METRIC} = "
        f"{metrics[PRIMARY_METRIC]:.2%}"
    )


# 9. Compare validation results

metrics_data = pd.DataFrame(metric_rows)

metrics_data = metrics_data.sort_values(
    by=[
        PRIMARY_METRIC,
        "F1_Up",
        "Accuracy"
    ],
    ascending=False
).reset_index(drop=True)

metrics_file = (
    modelling_data_dir / "06_logistic_validation_metrics.csv"
)

save_csv(metrics_data, metrics_file)

print("\nLogistic Regression validation results:")
print(
    metrics_data[
        [
            "Model",
            "C_Value",
            "Accuracy",
            "Balanced_Accuracy",
            "Precision_Up",
            "Recall_Up",
            "F1_Up",
            "ROC_AUC"
        ]
    ]
)

comparison_plot_file = save_validation_comparison_plot(
    metrics_data
)


# 10. Save validation predictions

prediction_data = pd.concat(
    prediction_tables,
    ignore_index=True
)

predictions_file = (
    modelling_data_dir
    / "06_logistic_validation_predictions.csv"
)

save_csv(prediction_data, predictions_file)


# 11. Select best Logistic Regression model

best_result = metrics_data.iloc[0]

best_model_name = best_result["Model"]

best_model = trained_models[best_model_name]

best_c_value = best_result["C_Value"]

print(
    f"\nBest Logistic Regression model: "
    f"{best_model_name}"
)

print(
    f"Selected using validation "
    f"{PRIMARY_METRIC}: "
    f"{best_result[PRIMARY_METRIC]:.2%}"
)


# 12. Save best validation model

best_model_file = (
    model_dir / "06_logistic_best_validation_model.joblib"
)

joblib.dump(best_model, best_model_file)

print(f"Saved model: {best_model_file}")


# 13. Save best model coefficients

coefficients_data = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": best_model.coef_[0]
})

coefficients_data["Absolute_Coefficient"] = np.abs(
    coefficients_data["Coefficient"]
)

coefficients_data["Direction"] = np.where(
    coefficients_data["Coefficient"] > 0,
    "Higher value associated with Up",
    "Higher value associated with Not Up"
)

coefficients_data = coefficients_data.sort_values(
    "Absolute_Coefficient",
    ascending=False
).reset_index(drop=True)

coefficients_file = (
    modelling_data_dir
    / "06_logistic_best_coefficients.csv"
)

save_csv(coefficients_data, coefficients_file)

coefficient_plot_file = save_coefficients_plot(
    coefficients_data
)


# 14. Create confusion matrix for best validation model

best_predictions = best_model.predict(X_validation)

best_probabilities = best_model.predict_proba(
    X_validation
)[:, 1]

confusion_matrix_file = (
    image_dir
    / "03_logistic_best_validation_confusion_matrix.png"
)

save_confusion_matrix_plot(
    y_true=y_validation,
    y_pred=best_predictions,
    model_name=best_model_name,
    split_name="Validation",
    output_path=confusion_matrix_file
)


# 15. Create report

report_file = (
    report_dir / "06_logistic_regression_report.md"
)

report_lines = [
    "# 06 Logistic Regression Report",
    "",
    "## Goal",
    "",
    (
        "Train several Logistic Regression configurations "
        "on the training split and compare them on the "
        "validation split."
    ),
    "",
    "## Why Scaled Data Is Used",
    "",
    (
        "Logistic Regression is sensitive to feature scales. "
        "Therefore, the scaled datasets from step 05 are used."
    ),
    "",
    "## Leakage Control",
    "",
    "- Only training data is used to fit the models.",
    "- Validation data is used only to compare C values.",
    "- Test data is not loaded or used in this script.",
    "",
    "## Tested Configurations",
    ""
]

for c_value in LOGISTIC_C_VALUES:
    report_lines.append(f"- Logistic Regression with C = {c_value}")

report_lines.extend([
    "",
    "## Best Validation Model",
    "",
    f"- Model: `{best_model_name}`",
    f"- Selected C value: {best_c_value}",
    (
        f"- {PRIMARY_METRIC}: "
        f"{best_result[PRIMARY_METRIC]:.2%}"
    ),
    f"- Accuracy: {best_result['Accuracy']:.2%}",
    f"- F1 Score for Up: {best_result['F1_Up']:.2%}",
    "",
    "## Output Files",
    "",
    f"- `{metrics_file.name}`",
    f"- `{predictions_file.name}`",
    f"- `{coefficients_file.name}`",
    f"- `{best_model_file.name}`",
    f"- `{comparison_plot_file.name}`",
    f"- `{confusion_matrix_file.name}`",
    f"- `{coefficient_plot_file.name}`"
])

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(f"Saved report: {report_file}")

print("\nLogistic Regression modelling completed successfully.")
print("Next step: compare the result with the baseline.")