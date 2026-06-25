"""
Final Model Selection and Test Evaluation
for Weekly BTC/USD Trading Bot

This script:
1. Loads validation metrics from baseline, Logistic Regression
   and XGBoost.
2. Selects the best model using validation data only.
3. Retrains the selected model on Train + Validation data.
4. Evaluates the final model once on the untouched test data.
5. Saves final test metrics, predictions, model and plots.

Important:
- Model selection is based only on validation results.
- The test dataset is used only after final model selection.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


# 2. Load parameters

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

model_config = params.get("MODELLING", {})

RANDOM_STATE = model_config.get("RANDOM_STATE", 42)

PRIMARY_METRIC = model_config.get(
    "PRIMARY_METRIC",
    "Balanced_Accuracy"
)

XGBOOST_CONFIGURATIONS = model_config.get(
    "XGBOOST_CONFIGURATIONS",
    []
)


# 3. Define input files

TRAIN_RAW_FILE = processed_data_dir / "05_train_selected_raw.csv"
VALIDATION_RAW_FILE = (
    processed_data_dir / "05_validation_selected_raw.csv"
)
TEST_RAW_FILE = processed_data_dir / "05_test_selected_raw.csv"

BASELINE_METRICS_FILE = (
    modelling_data_dir / "06_baseline_validation_metrics.csv"
)

LOGISTIC_METRICS_FILE = (
    modelling_data_dir / "06_logistic_validation_metrics.csv"
)

XGBOOST_METRICS_FILE = (
    modelling_data_dir / "06_xgboost_validation_metrics.csv"
)


# 4. Helper functions

def load_metrics(file_path, model_group):
    """
    Loads one validation metrics file and adds the model group.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    if PRIMARY_METRIC not in data.columns:
        raise ValueError(
            f"Missing primary metric '{PRIMARY_METRIC}' "
            f"in {file_path.name}"
        )

    data["Model_Group"] = model_group

    return data


def get_best_xgboost_configuration(model_name):
    """
    Finds the YAML configuration for the selected XGBoost model.
    """

    for config in XGBOOST_CONFIGURATIONS:
        if config["NAME"] == model_name:
            return config

    raise ValueError(
        f"No XGBoost configuration found for: {model_name}"
    )


def create_xgboost_model(config):
    """
    Creates one XGBoost classifier from YAML configuration.
    """

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(config["N_ESTIMATORS"]),
        max_depth=int(config["MAX_DEPTH"]),
        learning_rate=float(config["LEARNING_RATE"]),
        subsample=float(config["SUBSAMPLE"]),
        colsample_bytree=float(config["COLSAMPLE_BYTREE"]),
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        verbosity=0
    )


def save_validation_selection_plot(comparison_data):
    """
    Creates a plot of all validation results.
    """

    plot_data = comparison_data.sort_values(
        PRIMARY_METRIC,
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data["Model"],
        plot_data[PRIMARY_METRIC]
    )

    plt.title("Final Model Selection on Validation Data")
    plt.xlabel(PRIMARY_METRIC.replace("_", " "))
    plt.ylabel("Model")
    plt.tight_layout()

    output_file = (
        image_dir / "08_final_validation_model_selection.png"
    )

    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot: {output_file}")

    return output_file


# 5. Load validation metrics only

print("Starting final model selection")

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

comparison_data = pd.concat(
    [
        baseline_metrics,
        logistic_metrics,
        xgboost_metrics
    ],
    ignore_index=True
)

comparison_data = comparison_data.sort_values(
    by=[
        PRIMARY_METRIC,
        "F1_Up",
        "Accuracy"
    ],
    ascending=False
).reset_index(drop=True)

comparison_file = (
    modelling_data_dir / "06_all_validation_model_comparison.csv"
)

save_csv(comparison_data, comparison_file)

selection_plot_file = save_validation_selection_plot(
    comparison_data
)

print("\nValidation comparison:")
print(
    comparison_data[
        [
            "Model",
            "Model_Group",
            "Accuracy",
            "Balanced_Accuracy",
            "Precision_Up",
            "Recall_Up",
            "F1_Up",
            "ROC_AUC"
        ]
    ]
)


# 6. Select final model using validation only

best_result = comparison_data.iloc[0]

best_model_name = best_result["Model"]
best_model_group = best_result["Model_Group"]

print(
    f"\nSelected final model: {best_model_name}"
)

print(
    f"Selected with validation "
    f"{PRIMARY_METRIC}: "
    f"{best_result[PRIMARY_METRIC]:.2%}"
)


# 7. Load Train + Validation after model selection

train_raw = load_model_dataset(
    TRAIN_RAW_FILE,
    "Train raw dataset"
)

validation_raw = load_model_dataset(
    VALIDATION_RAW_FILE,
    "Validation raw dataset"
)

test_raw = load_model_dataset(
    TEST_RAW_FILE,
    "Test raw dataset"
)

feature_columns = get_feature_columns(train_raw)

train_validation_data = pd.concat(
    [train_raw, validation_raw],
    ignore_index=True
).sort_values("Date").reset_index(drop=True)

X_train_validation_raw = train_validation_data[
    feature_columns
]

y_train_validation = train_validation_data[
    TARGET_COLUMN
]

X_test_raw = test_raw[feature_columns]
y_test = test_raw[TARGET_COLUMN]


# 8. Retrain selected model on Train + Validation

final_scaler = None

if best_model_group == "Baseline":

    final_model = DummyClassifier(
        strategy="constant",
        constant=1
    )

    final_model.fit(
        X_train_validation_raw,
        y_train_validation
    )

    test_predictions = final_model.predict(X_test_raw)

    test_probabilities = final_model.predict_proba(
        X_test_raw
    )[:, 1]


elif best_model_group == "Logistic Regression":

    best_c_value = float(best_result["C_Value"])

    # Refit scaler on Train + Validation only.
    # Test data is not used for fitting.
    final_scaler = StandardScaler()

    final_scaler.fit(X_train_validation_raw)

    X_train_validation_scaled = final_scaler.transform(
        X_train_validation_raw
    )

    X_test_scaled = final_scaler.transform(X_test_raw)

    final_model = LogisticRegression(
        C=best_c_value,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    final_model.fit(
        X_train_validation_scaled,
        y_train_validation
    )

    test_predictions = final_model.predict(X_test_scaled)

    test_probabilities = final_model.predict_proba(
        X_test_scaled
    )[:, 1]


elif best_model_group == "XGBoost":

    best_xgboost_config = get_best_xgboost_configuration(
        best_model_name
    )

    final_model = create_xgboost_model(
        best_xgboost_config
    )

    final_model.fit(
        X_train_validation_raw,
        y_train_validation
    )

    test_predictions = final_model.predict(X_test_raw)

    test_probabilities = final_model.predict_proba(
        X_test_raw
    )[:, 1]


else:
    raise ValueError(
        f"Unknown model group: {best_model_group}"
    )


# 9. Final evaluation on untouched test data

test_metrics = evaluate_predictions(
    model_name=best_model_name,
    split_name="Test",
    y_true=y_test,
    y_pred=test_predictions,
    y_probability=test_probabilities
)

test_metrics_data = pd.DataFrame([test_metrics])

test_metrics_file = (
    modelling_data_dir / "06_final_test_metrics.csv"
)

save_csv(test_metrics_data, test_metrics_file)

print("\nFinal test metrics:")
print(test_metrics_data)


# 10. Save test predictions

test_predictions_data = create_prediction_table(
    data=test_raw,
    model_name=best_model_name,
    predictions=test_predictions,
    probabilities=test_probabilities
)

test_predictions_file = (
    modelling_data_dir / "06_final_test_predictions.csv"
)

save_csv(test_predictions_data, test_predictions_file)


# 11. Save final confusion matrix

confusion_matrix_file = (
    image_dir / "09_final_test_confusion_matrix.png"
)

save_confusion_matrix_plot(
    y_true=y_test,
    y_pred=test_predictions,
    model_name=best_model_name,
    split_name="Test",
    output_path=confusion_matrix_file
)


# 12. Save final model and scaler

final_model_file = (
    model_dir / "06_final_selected_model.joblib"
)

joblib.dump(final_model, final_model_file)

print(f"Saved final model: {final_model_file}")

final_scaler_file = None

if final_scaler is not None:
    final_scaler_file = (
        model_dir / "06_final_selected_logistic_scaler.joblib"
    )

    joblib.dump(final_scaler, final_scaler_file)

    print(f"Saved final scaler: {final_scaler_file}")


# 13. Create report

report_file = (
    report_dir / "06_final_model_selection_and_test_report.md"
)

report_lines = [
    "# 06 Final Model Selection and Test Report",
    "",
    "## Model Selection",
    "",
    (
        "The final model was selected only using validation "
        "results. The test set was not used for model selection."
    ),
    "",
    f"- Primary metric: `{PRIMARY_METRIC}`",
    f"- Selected model: `{best_model_name}`",
    (
        f"- Validation {PRIMARY_METRIC}: "
        f"{best_result[PRIMARY_METRIC]:.2%}"
    ),
    "",
    "## Final Test Result",
    "",
    f"- Accuracy: {test_metrics['Accuracy']:.2%}",
    (
        "- Balanced Accuracy: "
        f"{test_metrics['Balanced_Accuracy']:.2%}"
    ),
    f"- Precision for Up: {test_metrics['Precision_Up']:.2%}",
    f"- Recall for Up: {test_metrics['Recall_Up']:.2%}",
    f"- F1 Score for Up: {test_metrics['F1_Up']:.2%}",
    f"- ROC-AUC: {test_metrics['ROC_AUC']:.2%}",
    "",
    "## Leakage Control",
    "",
    "- Validation data selected the final model.",
    "- Test data was used only once for final evaluation.",
    (
        "- For Logistic Regression, the scaler was refitted "
        "only on Train + Validation data."
    ),
    "- Test data was never used to fit the scaler or the model.",
    "",
    "## Output Files",
    "",
    f"- `{comparison_file.name}`",
    f"- `{test_metrics_file.name}`",
    f"- `{test_predictions_file.name}`",
    f"- `{final_model_file.name}`",
    f"- `{confusion_matrix_file.name}`"
]

if final_scaler_file is not None:
    report_lines.append(
        f"- `{final_scaler_file.name}`"
    )

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(f"Saved report: {report_file}")

print("\nFinal model selection and test evaluation completed.")