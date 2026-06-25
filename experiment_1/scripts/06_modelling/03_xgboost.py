"""
XGBoost Modelling for Weekly BTC/USD Trading Bot

This script:
1. Loads the selected raw train and validation datasets.
2. Trains several small XGBoost configurations.
3. Compares all configurations on validation data only.
4. Selects the best XGBoost configuration.
5. Saves validation metrics, predictions, feature importance and plots.

Important:
- The test dataset is intentionally NOT used here.
- The final model comparison and test evaluation happen later.

Input files:
- data/processed/05_train_selected_raw.csv
- data/processed/05_validation_selected_raw.csv

Output files:
- data/modelling/06_xgboost_validation_metrics.csv
- data/modelling/06_xgboost_validation_predictions.csv
- data/modelling/06_xgboost_best_feature_importance.csv
- data/modelling/06_xgboost_best_validation_config.yaml
- experiment_1/models/06_xgboost_best_validation_model.joblib
- experiment_1/images/06_modelling/05_xgboost_validation_comparison.png
- experiment_1/images/06_modelling/06_xgboost_best_validation_confusion_matrix.png
- experiment_1/images/06_modelling/07_xgboost_best_feature_importance.png
- experiment_1/reports/06_xgboost_report.md
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import yaml

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


# 2. Load parameters from YAML

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
    processed_data_dir / "05_train_selected_raw.csv"
)

VALIDATION_FILE = (
    processed_data_dir / "05_validation_selected_raw.csv"
)


# 4. Helper functions

def validate_feature_columns(
    train_data,
    validation_data,
    feature_columns
):
    """
    Checks that validation data contains all feature columns
    used in training.
    """

    missing_features = (
        set(feature_columns) - set(validation_data.columns)
    )

    if missing_features:
        raise ValueError(
            "Validation data is missing feature columns: "
            f"{sorted(missing_features)}"
        )


def validate_xgboost_configurations():
    """
    Checks whether the XGBoost configurations are valid.
    """

    if not XGBOOST_CONFIGURATIONS:
        raise ValueError(
            "No XGBOOST_CONFIGURATIONS were found in params.yaml."
        )

    required_keys = [
        "NAME",
        "N_ESTIMATORS",
        "MAX_DEPTH",
        "LEARNING_RATE",
        "SUBSAMPLE",
        "COLSAMPLE_BYTREE"
    ]

    for config in XGBOOST_CONFIGURATIONS:
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"XGBoost configuration is missing key: {key}"
                )


def create_xgboost_model(config):
    """
    Creates one XGBClassifier from the YAML configuration.
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


def save_validation_comparison_plot(metrics_data):
    """
    Creates a bar chart comparing all XGBoost configurations
    using the selected primary validation metric.
    """

    plot_data = metrics_data.sort_values(
        PRIMARY_METRIC,
        ascending=False
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        plot_data["Model"],
        plot_data[PRIMARY_METRIC]
    )

    plt.title("XGBoost: Validation Comparison")
    plt.xlabel("XGBoost Configuration")
    plt.ylabel(PRIMARY_METRIC.replace("_", " "))
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_file = (
        image_dir / "05_xgboost_validation_comparison.png"
    )

    plt.savefig(output_file)
    plt.close()

    print(f"Saved plot: {output_file}")

    return output_file


def save_feature_importance_plot(importance_data):
    """
    Creates a horizontal bar chart for the feature importance
    of the best XGBoost model.
    """

    plot_data = importance_data.sort_values(
        "Feature_Importance"
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_data["Feature"],
        plot_data["Feature_Importance"]
    )

    plt.title("Best XGBoost Model: Feature Importance")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_file = (
        image_dir / "07_xgboost_best_feature_importance.png"
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

validate_xgboost_configurations()


# 6. Load train and validation data

print("Starting XGBoost modelling")

train_data = load_model_dataset(
    TRAIN_FILE,
    "Train raw dataset"
)

validation_data = load_model_dataset(
    VALIDATION_FILE,
    "Validation raw dataset"
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


# 8. Train and evaluate XGBoost configurations

metric_rows = []
prediction_tables = []
trained_models = {}
model_configurations = {}

for config in XGBOOST_CONFIGURATIONS:

    model_name = config["NAME"]

    xgboost_model = create_xgboost_model(config)

    xgboost_model.fit(X_train, y_train)

    predictions = xgboost_model.predict(X_validation)

    probabilities = xgboost_model.predict_proba(
        X_validation
    )[:, 1]

    metrics = evaluate_predictions(
        model_name=model_name,
        split_name="Validation",
        y_true=y_validation,
        y_pred=predictions,
        y_probability=probabilities
    )

    metrics["N_Estimators"] = int(config["N_ESTIMATORS"])
    metrics["Max_Depth"] = int(config["MAX_DEPTH"])
    metrics["Learning_Rate"] = float(config["LEARNING_RATE"])
    metrics["Subsample"] = float(config["SUBSAMPLE"])
    metrics["Colsample_Bytree"] = float(
        config["COLSAMPLE_BYTREE"]
    )

    metric_rows.append(metrics)

    prediction_tables.append(
        create_prediction_table(
            data=validation_data,
            model_name=model_name,
            predictions=predictions,
            probabilities=probabilities
        )
    )

    trained_models[model_name] = xgboost_model
    model_configurations[model_name] = config

    print(
        f"Finished {model_name}: "
        f"{PRIMARY_METRIC} = "
        f"{metrics[PRIMARY_METRIC]:.2%}"
    )


# 9. Save validation metrics and comparison plot

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
    modelling_data_dir / "06_xgboost_validation_metrics.csv"
)

save_csv(metrics_data, metrics_file)

print("\nXGBoost validation results:")
print(
    metrics_data[
        [
            "Model",
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
    / "06_xgboost_validation_predictions.csv"
)

save_csv(prediction_data, predictions_file)


# 11. Select best XGBoost model

best_result = metrics_data.iloc[0]

best_model_name = best_result["Model"]

best_model = trained_models[best_model_name]

best_configuration = model_configurations[best_model_name]

print(
    f"\nBest XGBoost model: {best_model_name}"
)

print(
    f"Selected using validation "
    f"{PRIMARY_METRIC}: "
    f"{best_result[PRIMARY_METRIC]:.2%}"
)


# 12. Save best validation model and configuration

best_model_file = (
    model_dir / "06_xgboost_best_validation_model.joblib"
)

joblib.dump(best_model, best_model_file)

print(f"Saved model: {best_model_file}")

best_config_file = (
    modelling_data_dir
    / "06_xgboost_best_validation_config.yaml"
)

with open(best_config_file, "w") as file:
    yaml.safe_dump(
        best_configuration,
        file,
        sort_keys=False
    )

print(f"Saved configuration: {best_config_file}")


# 13. Save feature importance

importance_data = pd.DataFrame({
    "Feature": feature_columns,
    "Feature_Importance": best_model.feature_importances_
})

importance_data = importance_data.sort_values(
    "Feature_Importance",
    ascending=False
).reset_index(drop=True)

importance_file = (
    modelling_data_dir
    / "06_xgboost_best_feature_importance.csv"
)

save_csv(importance_data, importance_file)

importance_plot_file = save_feature_importance_plot(
    importance_data
)


# 14. Save confusion matrix for best validation model

best_predictions = best_model.predict(X_validation)

best_probabilities = best_model.predict_proba(
    X_validation
)[:, 1]

confusion_matrix_file = (
    image_dir
    / "06_xgboost_best_validation_confusion_matrix.png"
)

save_confusion_matrix_plot(
    y_true=y_validation,
    y_pred=best_predictions,
    model_name=best_model_name,
    split_name="Validation",
    output_path=confusion_matrix_file
)


# 15. Create report

report_file = report_dir / "06_xgboost_report.md"

report_lines = [
    "# 06 XGBoost Report",
    "",
    "## Goal",
    "",
    (
        "Train several small XGBoost configurations on the "
        "training data and compare them on validation data."
    ),
    "",
    "## Why Raw Data Is Used",
    "",
    (
        "XGBoost is a tree-based model. Tree splits are not "
        "sensitive to feature scaling, so selected raw features "
        "are used."
    ),
    "",
    "## Leakage Control",
    "",
    "- Models are trained only on training data.",
    "- Validation data is used only for configuration selection.",
    "- Test data is not loaded or used in this script.",
    "",
    "## Tested Configurations",
    ""
]

for config in XGBOOST_CONFIGURATIONS:
    report_lines.append(
        f"- `{config['NAME']}`: "
        f"n_estimators={config['N_ESTIMATORS']}, "
        f"max_depth={config['MAX_DEPTH']}, "
        f"learning_rate={config['LEARNING_RATE']}"
    )

report_lines.extend([
    "",
    "## Best Validation Model",
    "",
    f"- Model: `{best_model_name}`",
    (
        f"- {PRIMARY_METRIC}: "
        f"{best_result[PRIMARY_METRIC]:.2%}"
    ),
    f"- Accuracy: {best_result['Accuracy']:.2%}",
    f"- F1 Score for Up: {best_result['F1_Up']:.2%}",
    f"- ROC-AUC: {best_result['ROC_AUC']:.2%}",
    "",
    "## Output Files",
    "",
    f"- `{metrics_file.name}`",
    f"- `{predictions_file.name}`",
    f"- `{importance_file.name}`",
    f"- `{best_config_file.name}`",
    f"- `{best_model_file.name}`",
    f"- `{comparison_plot_file.name}`",
    f"- `{confusion_matrix_file.name}`",
    f"- `{importance_plot_file.name}`"
])

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(f"Saved report: {report_file}")

print("\nXGBoost modelling completed successfully.")
print(
    "Next step: compare baseline, Logistic Regression and "
    "XGBoost before using the test data."
)
