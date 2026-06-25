"""
Model Visualisations for Weekly BTC/USD Trading Bot

This script creates visualisations from already saved model results.

It does NOT train a new model.
It does NOT change the test set.
It only reads existing validation and final test results.

Created outputs:
- data/modelling/06_best_validation_models_by_group.csv
- data/modelling/06_validation_test_comparison.csv
- data/modelling/06_top_10_logistic_coefficients.csv
- experiment_1/images/06_modelling/08_final_validation_model_comparison.png
- experiment_1/images/06_modelling/10_validation_vs_test_comparison.png
- experiment_1/images/06_modelling/11_top_10_logistic_coefficients.png
- experiment_1/reports/06_model_visualisations_report.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from model_utils import (
    TARGET_COLUMN,
    evaluate_predictions,
    get_project_paths,
    load_model_dataset,
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


# 2. Load modelling parameters

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

model_config = params.get("MODELLING", {})

PRIMARY_METRIC = model_config.get(
    "PRIMARY_METRIC",
    "Balanced_Accuracy"
)


# 3. Define input files

BASELINE_METRICS_FILE = (
    modelling_data_dir / "06_baseline_validation_metrics.csv"
)

LOGISTIC_METRICS_FILE = (
    modelling_data_dir / "06_logistic_validation_metrics.csv"
)

XGBOOST_METRICS_FILE = (
    modelling_data_dir / "06_xgboost_validation_metrics.csv"
)

FINAL_TEST_METRICS_FILE = (
    modelling_data_dir / "06_final_test_metrics.csv"
)

LOGISTIC_COEFFICIENTS_FILE = (
    modelling_data_dir / "06_logistic_best_coefficients.csv"
)

TEST_RAW_FILE = (
    processed_data_dir / "05_test_selected_raw.csv"
)


# 4. Helper functions

def load_metrics(file_path, model_group):
    """
    Loads one metrics CSV and adds a model group column.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    if PRIMARY_METRIC not in data.columns:
        raise ValueError(
            f"Column '{PRIMARY_METRIC}' is missing in "
            f"{file_path.name}"
        )

    if "Model" not in data.columns:
        raise ValueError(
            f"Column 'Model' is missing in {file_path.name}"
        )

    data["Model_Group"] = model_group

    return data


def select_best_model_per_group(metrics_data):
    """
    Selects the best model in each model group using validation data.

    Priority:
    1. Primary metric
    2. F1 score for Up
    3. Accuracy
    """

    sorted_data = metrics_data.sort_values(
        by=[
            PRIMARY_METRIC,
            "F1_Up",
            "Accuracy"
        ],
        ascending=False
    )

    best_by_group = (
        sorted_data
        .groupby("Model_Group", group_keys=False)
        .head(1)
        .copy()
    )

    group_order = {
        "Baseline": 1,
        "Logistic Regression": 2,
        "XGBoost": 3
    }

    best_by_group["Group_Order"] = best_by_group[
        "Model_Group"
    ].map(group_order)

    best_by_group = best_by_group.sort_values(
        "Group_Order"
    ).drop(columns="Group_Order")

    return best_by_group


def save_validation_model_comparison_plot(best_models):
    """
    Creates a bar chart comparing the best model from each group
    on the validation dataset.
    """

    plt.figure(figsize=(10, 6))

    plt.bar(
        best_models["Model"],
        best_models[PRIMARY_METRIC]
    )

    plt.axhline(
        y=0.50,
        linestyle="--",
        label="50 % Referenz"
    )

    plt.title("Modellvergleich auf dem Validation-Set")
    plt.xlabel("Modell")
    plt.ylabel(PRIMARY_METRIC.replace("_", " "))
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()

    output_file = (
        image_dir / "08_final_validation_model_comparison.png"
    )

    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot: {output_file}")

    return output_file


def create_test_baseline_metrics(test_data):
    """
    Creates an Always-Up baseline on the test dataset.

    This does not train or tune anything.
    It is only used as a transparent comparison for the
    final selected model.
    """

    y_test = test_data[TARGET_COLUMN]

    predictions = np.ones(
        len(test_data),
        dtype=int
    )

    probabilities = np.ones(
        len(test_data),
        dtype=float
    )

    baseline_metrics = evaluate_predictions(
        model_name="Baseline_Always_Up",
        split_name="Test",
        y_true=y_test,
        y_pred=predictions,
        y_probability=probabilities
    )

    return baseline_metrics


def save_validation_test_comparison_plot(comparison_data):
    """
    Creates a grouped bar chart comparing baseline and selected model
    between validation and test data.
    """

    baseline_validation = comparison_data[
        (
            comparison_data["Split"] == "Validation"
        )
        &
        (
            comparison_data["Comparison_Model"]
            == "Always-Up Baseline"
        )
    ][PRIMARY_METRIC].iloc[0]

    selected_validation = comparison_data[
        (
            comparison_data["Split"] == "Validation"
        )
        &
        (
            comparison_data["Comparison_Model"]
            == "Selected Model"
        )
    ][PRIMARY_METRIC].iloc[0]

    baseline_test = comparison_data[
        (
            comparison_data["Split"] == "Test"
        )
        &
        (
            comparison_data["Comparison_Model"]
            == "Always-Up Baseline"
        )
    ][PRIMARY_METRIC].iloc[0]

    selected_test = comparison_data[
        (
            comparison_data["Split"] == "Test"
        )
        &
        (
            comparison_data["Comparison_Model"]
            == "Selected Model"
        )
    ][PRIMARY_METRIC].iloc[0]

    split_labels = ["Validation", "Test"]

    x_positions = np.arange(len(split_labels))
    bar_width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar(
        x_positions - bar_width / 2,
        [baseline_validation, baseline_test],
        bar_width,
        label="Always-Up Baseline"
    )

    plt.bar(
        x_positions + bar_width / 2,
        [selected_validation, selected_test],
        bar_width,
        label="Ausgewähltes Modell"
    )

    plt.axhline(
        y=0.50,
        linestyle="--",
        label="50 % Referenz"
    )

    plt.title("Validation vs. Test: Generalisierungsvergleich")
    plt.xlabel("Datensatz")
    plt.ylabel(PRIMARY_METRIC.replace("_", " "))
    plt.xticks(x_positions, split_labels)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output_file = (
        image_dir / "10_validation_vs_test_comparison.png"
    )

    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot: {output_file}")

    return output_file


def load_and_prepare_top_coefficients():
    """
    Loads the Logistic Regression coefficients and keeps
    the 10 features with the largest absolute coefficients.
    """

    if not LOGISTIC_COEFFICIENTS_FILE.exists():
        raise FileNotFoundError(
            "Logistic coefficient file not found: "
            f"{LOGISTIC_COEFFICIENTS_FILE}"
        )

    coefficients = pd.read_csv(LOGISTIC_COEFFICIENTS_FILE)

    required_columns = [
        "Feature",
        "Coefficient"
    ]

    for column in required_columns:
        if column not in coefficients.columns:
            raise ValueError(
                f"Column '{column}' missing in coefficient file."
            )

    if "Absolute_Coefficient" not in coefficients.columns:
        coefficients["Absolute_Coefficient"] = np.abs(
            coefficients["Coefficient"]
        )

    top_coefficients = coefficients.nlargest(
        10,
        "Absolute_Coefficient"
    ).copy()

    top_coefficients["Direction"] = np.where(
        top_coefficients["Coefficient"] > 0,
        "Positive association with Up",
        "Negative association with Up"
    )

    return top_coefficients


def save_top_coefficients_plot(top_coefficients):
    """
    Creates a coefficient chart for the top 10 Logistic Regression
    features by absolute coefficient value.
    """

    plot_data = top_coefficients.sort_values(
        "Coefficient"
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_data["Feature"],
        plot_data["Coefficient"]
    )

    plt.axvline(
        x=0,
        linestyle="--"
    )

    plt.title(
        "Top-10 Merkmale der Logistic Regression"
    )
    plt.xlabel("Koeffizient")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_file = (
        image_dir / "11_top_10_logistic_coefficients.png"
    )

    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot: {output_file}")

    return output_file


# 5. Load all existing validation metrics

print("Starting model visualisations")

baseline_metrics = load_metrics(
    BASELINE_METRICS_FILE,
    "Baseline"
)

logistic_metrics = load_metrics(
    LOGISTIC_METRICS_FILE,
    "Logistic Regression"
)

xgboost_metrics = load_metrics(
    XGBOOST_METRICS_FILE,
    "XGBoost"
)

all_validation_metrics = pd.concat(
    [
        baseline_metrics,
        logistic_metrics,
        xgboost_metrics
    ],
    ignore_index=True
)


# 6. Select best model per group using validation only

best_validation_models = select_best_model_per_group(
    all_validation_metrics
)

best_validation_models_file = (
    modelling_data_dir
    / "06_best_validation_models_by_group.csv"
)

save_csv(
    best_validation_models,
    best_validation_models_file
)

print("\nBest validation model per group:")
print(
    best_validation_models[
        [
            "Model_Group",
            "Model",
            "Accuracy",
            "Balanced_Accuracy",
            "F1_Up",
            "ROC_AUC"
        ]
    ]
)

validation_comparison_plot = (
    save_validation_model_comparison_plot(
        best_validation_models
    )
)


# 7. Identify selected final model from validation data

best_overall_validation_model = (
    best_validation_models
    .sort_values(
        by=[
            PRIMARY_METRIC,
            "F1_Up",
            "Accuracy"
        ],
        ascending=False
    )
    .iloc[0]
)

selected_model_name = best_overall_validation_model["Model"]

selected_validation_result = best_overall_validation_model

print(
    f"\nSelected model based on validation: "
    f"{selected_model_name}"
)


# 8. Load final test result

final_test_metrics = load_metrics(
    FINAL_TEST_METRICS_FILE,
    "Selected Model"
)

selected_test_result = final_test_metrics.iloc[0]

if selected_test_result["Model"] != selected_model_name:
    print(
        "Warning: The model selected in the validation results "
        "does not match the final test model."
    )


# 9. Calculate Always-Up baseline for test comparison

test_data = load_model_dataset(
    TEST_RAW_FILE,
    "Test raw dataset"
)

baseline_test_metrics = create_test_baseline_metrics(
    test_data
)

baseline_validation_result = best_validation_models[
    best_validation_models["Model_Group"] == "Baseline"
].iloc[0]


# 10. Create Validation vs. Test comparison table

comparison_rows = [
    {
        "Split": "Validation",
        "Comparison_Model": "Always-Up Baseline",
        "Actual_Model_Name": baseline_validation_result["Model"],
        "Accuracy": baseline_validation_result["Accuracy"],
        "Balanced_Accuracy": baseline_validation_result[
            "Balanced_Accuracy"
        ],
        "F1_Up": baseline_validation_result["F1_Up"],
        "ROC_AUC": baseline_validation_result["ROC_AUC"]
    },
    {
        "Split": "Validation",
        "Comparison_Model": "Selected Model",
        "Actual_Model_Name": selected_model_name,
        "Accuracy": selected_validation_result["Accuracy"],
        "Balanced_Accuracy": selected_validation_result[
            "Balanced_Accuracy"
        ],
        "F1_Up": selected_validation_result["F1_Up"],
        "ROC_AUC": selected_validation_result["ROC_AUC"]
    },
    {
        "Split": "Test",
        "Comparison_Model": "Always-Up Baseline",
        "Actual_Model_Name": "Baseline_Always_Up",
        "Accuracy": baseline_test_metrics["Accuracy"],
        "Balanced_Accuracy": baseline_test_metrics[
            "Balanced_Accuracy"
        ],
        "F1_Up": baseline_test_metrics["F1_Up"],
        "ROC_AUC": baseline_test_metrics["ROC_AUC"]
    },
    {
        "Split": "Test",
        "Comparison_Model": "Selected Model",
        "Actual_Model_Name": selected_test_result["Model"],
        "Accuracy": selected_test_result["Accuracy"],
        "Balanced_Accuracy": selected_test_result[
            "Balanced_Accuracy"
        ],
        "F1_Up": selected_test_result["F1_Up"],
        "ROC_AUC": selected_test_result["ROC_AUC"]
    }
]

validation_test_comparison = pd.DataFrame(
    comparison_rows
)

validation_test_file = (
    modelling_data_dir
    / "06_validation_test_comparison.csv"
)

save_csv(
    validation_test_comparison,
    validation_test_file
)

validation_test_plot = (
    save_validation_test_comparison_plot(
        validation_test_comparison
    )
)


# 11. Create Top-10 Logistic Regression coefficient table and plot

top_coefficients = load_and_prepare_top_coefficients()

top_coefficients_file = (
    modelling_data_dir
    / "06_top_10_logistic_coefficients.csv"
)

save_csv(
    top_coefficients,
    top_coefficients_file
)

coefficient_plot = save_top_coefficients_plot(
    top_coefficients
)


# 12. Create report

report_file = (
    report_dir / "06_model_visualisations_report.md"
)

report_lines = [
    "# 06 Model Visualisations Report",
    "",
    "## Purpose",
    "",
    (
        "This script creates visualisations from already saved "
        "modelling results. No model is retrained and no test "
        "result is used to change model parameters."
    ),
    "",
    "## Validation Model Selection",
    "",
    (
        f"- Primary metric: `{PRIMARY_METRIC}`"
    ),
    (
        f"- Selected model: `{selected_model_name}`"
    ),
    (
        f"- Validation {PRIMARY_METRIC}: "
        f"{selected_validation_result[PRIMARY_METRIC]:.2%}"
    ),
    "",
    "## Generalisation Result",
    "",
    (
        f"- Selected model test {PRIMARY_METRIC}: "
        f"{selected_test_result[PRIMARY_METRIC]:.2%}"
    ),
    (
        f"- Always-Up baseline test {PRIMARY_METRIC}: "
        f"{baseline_test_metrics[PRIMARY_METRIC]:.2%}"
    ),
    "",
    "## Interpretation",
    "",
    (
        "The validation comparison shows which model was selected. "
        "The validation-versus-test comparison shows whether this "
        "advantage generalised to unseen data."
    ),
    "",
    "## Output Files",
    "",
    f"- `{best_validation_models_file.name}`",
    f"- `{validation_test_file.name}`",
    f"- `{top_coefficients_file.name}`",
    f"- `{validation_comparison_plot.name}`",
    f"- `{validation_test_plot.name}`",
    f"- `{coefficient_plot.name}`"
]

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(f"Saved report: {report_file}")

print("\nModel visualisations completed successfully.")

