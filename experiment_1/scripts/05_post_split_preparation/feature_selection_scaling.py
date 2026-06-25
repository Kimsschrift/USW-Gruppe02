"""
Post-Split Preparation for Weekly BTC/USD Trading Bot

This script:
1. Loads the chronological train, validation and test datasets.
2. Selects non-redundant features using TRAINING data only.
3. Fits a StandardScaler using TRAINING data only.
4. Applies the same scaler to validation and test data.
5. Saves selected raw and scaled datasets for later modelling.

Input files:
- data/processed/04_train_raw.csv
- data/processed/04_validation_raw.csv
- data/processed/04_test_raw.csv

Output files:
- data/processed/05_feature_selection.csv
- data/processed/05_train_selected_raw.csv
- data/processed/05_validation_selected_raw.csv
- data/processed/05_test_selected_raw.csv
- data/processed/05_train_selected_scaled.csv
- data/processed/05_validation_selected_scaled.csv
- data/processed/05_test_selected_scaled.csv
- data/processed/05_scaler.joblib
- data/processed/05_post_split_summary.csv
- experiment_1/reports/05_post_split_preparation_report.md
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


# 1. Define project paths

CURRENT_FILE = Path(__file__).resolve()

# Path to experiment_1 folder
EXPERIMENT_DIR = CURRENT_FILE.parents[2]

# Path to main project folder
PROJECT_DIR = CURRENT_FILE.parents[3]

# Path to params.yaml
PARAMS_PATH = EXPERIMENT_DIR / "conf" / "params.yaml"


# 2. Load parameters from YAML

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

pre_split_config = params["PRE_SPLIT_PREP"]
post_split_config = params["POST_SPLIT_PREPARATION"]

PROCESSED_DATA_DIR = (
    PROJECT_DIR / pre_split_config["PROCESSED_DATA_PATH"]
)

REPORT_DIR = EXPERIMENT_DIR / "reports"

CORRELATION_THRESHOLD = post_split_config[
    "CORRELATION_THRESHOLD"
]


# 3. Define input and output files

TRAIN_INPUT_FILE = PROCESSED_DATA_DIR / "04_train_raw.csv"
VALIDATION_INPUT_FILE = PROCESSED_DATA_DIR / "04_validation_raw.csv"
TEST_INPUT_FILE = PROCESSED_DATA_DIR / "04_test_raw.csv"

DATE_COLUMN = "Date"
BTC_CLOSE_COLUMN = "btc_close"

FUTURE_RETURN_COLUMN = "target_next_week_return"
TARGET_COLUMN = "target_next_week_up"

INFO_COLUMNS = [
    DATE_COLUMN,
    BTC_CLOSE_COLUMN
]

NON_FEATURE_COLUMNS = INFO_COLUMNS + [
    FUTURE_RETURN_COLUMN,
    TARGET_COLUMN
]

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 4. Helper functions

def load_split(file_path, split_name):
    """
    Loads one split file and performs basic checks.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{split_name} file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    if DATE_COLUMN not in data.columns:
        raise ValueError(
            f"{split_name} is missing required column: {DATE_COLUMN}"
        )

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"{split_name} is missing required column: {TARGET_COLUMN}"
        )

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])

    if data.isna().any().any():
        missing_count = int(data.isna().sum().sum())

        raise ValueError(
            f"{split_name} contains {missing_count} missing values. "
            "Please check the previous preparation step."
        )

    print(
        f"Loaded {split_name}: {data.shape}, "
        f"{data[DATE_COLUMN].min().date()} to "
        f"{data[DATE_COLUMN].max().date()}"
    )

    return data


def get_feature_columns(data):
    """
    Returns all model feature columns.

    The following columns are excluded:
    - Date
    - btc_close
    - target_next_week_return
    - target_next_week_up
    """

    feature_columns = [
        column
        for column in data.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if len(feature_columns) == 0:
        raise ValueError("No model feature columns were found.")

    return feature_columns


def validate_feature_columns(
    train_data,
    validation_data,
    test_data,
    feature_columns
):
    """
    Checks whether all three splits contain the same feature columns.
    """

    required_columns = set(feature_columns)

    for split_name, split_data in [
        ("Validation", validation_data),
        ("Test", test_data)
    ]:
        missing_columns = required_columns - set(split_data.columns)

        if missing_columns:
            raise ValueError(
                f"{split_name} is missing feature columns: "
                f"{sorted(missing_columns)}"
            )


def select_features_by_correlation(
    train_data,
    feature_columns
):
    """
    Removes features that are highly correlated with an earlier
    retained feature.

    Important:
    This uses TRAINING data only.
    """

    train_features = train_data[feature_columns]

    selected_features = []
    selection_rows = []

    for feature in feature_columns:

        # Keep first feature automatically
        if len(selected_features) == 0:
            selected_features.append(feature)

            selection_rows.append({
                "Feature": feature,
                "Selected": True,
                "Max_Abs_Correlation": 0.0,
                "Most_Correlated_Selected_Feature": None,
                "Reason": "First feature retained."
            })

            continue

        correlations = train_features[
            selected_features
        ].corrwith(train_features[feature]).abs()

        max_correlation = float(correlations.max())
        most_correlated_feature = correlations.idxmax()

        if max_correlation > CORRELATION_THRESHOLD:
            selection_rows.append({
                "Feature": feature,
                "Selected": False,
                "Max_Abs_Correlation": max_correlation,
                "Most_Correlated_Selected_Feature": (
                    most_correlated_feature
                ),
                "Reason": (
                    "Removed because correlation exceeds threshold "
                    f"{CORRELATION_THRESHOLD:.2f}."
                )
            })

        else:
            selected_features.append(feature)

            selection_rows.append({
                "Feature": feature,
                "Selected": True,
                "Max_Abs_Correlation": max_correlation,
                "Most_Correlated_Selected_Feature": (
                    most_correlated_feature
                ),
                "Reason": (
                    "Retained because correlation is below threshold."
                )
            })

    selection_table = pd.DataFrame(selection_rows)

    return selected_features, selection_table


def create_model_dataset(
    data,
    selected_features,
    scaler=None
):
    """
    Creates a clean dataset for later modelling.

    It keeps:
    - Date
    - btc_close
    - selected features
    - target_next_week_return
    - target_next_week_up

    If scaler is given, selected features are scaled.
    """

    output_columns = (
        INFO_COLUMNS
        + selected_features
        + [FUTURE_RETURN_COLUMN, TARGET_COLUMN]
    )

    model_data = data[output_columns].copy()

    if scaler is not None:
        model_data.loc[:, selected_features] = scaler.transform(
            data[selected_features]
        )

    return model_data


def create_summary_row(
    split_name,
    data,
    selected_features
):
    """
    Creates one summary row for train, validation or test.
    """

    return {
        "Split": split_name,
        "Rows": len(data),
        "Start_Date": data[DATE_COLUMN].min().date(),
        "End_Date": data[DATE_COLUMN].max().date(),
        "Target_Up_Rate": data[TARGET_COLUMN].mean(),
        "Selected_Feature_Count": len(selected_features)
    }


def save_csv(data, file_name):
    """
    Saves a DataFrame in the processed data folder.
    """

    output_path = PROCESSED_DATA_DIR / file_name

    data.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return output_path


# 5. Validate configuration

if not 0 < CORRELATION_THRESHOLD < 1:
    raise ValueError(
        "CORRELATION_THRESHOLD must be between 0 and 1."
    )


# 6. Load train, validation and test data

print("Starting post-split preparation")

train_data = load_split(TRAIN_INPUT_FILE, "Train")
validation_data = load_split(
    VALIDATION_INPUT_FILE,
    "Validation"
)
test_data = load_split(TEST_INPUT_FILE, "Test")


# 7. Identify and validate feature columns

feature_columns = get_feature_columns(train_data)

validate_feature_columns(
    train_data,
    validation_data,
    test_data,
    feature_columns
)

print(f"Initial feature count: {len(feature_columns)}")


# 8. Feature selection using training data only

selected_features, feature_selection = (
    select_features_by_correlation(
        train_data,
        feature_columns
    )
)

feature_selection_file = save_csv(
    feature_selection,
    "05_feature_selection.csv"
)

print(f"Selected feature count: {len(selected_features)}")

print("Selected features:")

for feature in selected_features:
    print(f"- {feature}")


# 9. Save selected but unscaled data

train_selected_raw = create_model_dataset(
    train_data,
    selected_features
)

validation_selected_raw = create_model_dataset(
    validation_data,
    selected_features
)

test_selected_raw = create_model_dataset(
    test_data,
    selected_features
)

train_raw_file = save_csv(
    train_selected_raw,
    "05_train_selected_raw.csv"
)

validation_raw_file = save_csv(
    validation_selected_raw,
    "05_validation_selected_raw.csv"
)

test_raw_file = save_csv(
    test_selected_raw,
    "05_test_selected_raw.csv"
)


# 10. Scaling using training data only

scaler = StandardScaler()

# Fit only on training data
scaler.fit(train_data[selected_features])

scaler_file = PROCESSED_DATA_DIR / "05_scaler.joblib"

joblib.dump(scaler, scaler_file)

print(f"Saved scaler: {scaler_file}")

# Apply same scaler to all splits
train_selected_scaled = create_model_dataset(
    train_data,
    selected_features,
    scaler=scaler
)

validation_selected_scaled = create_model_dataset(
    validation_data,
    selected_features,
    scaler=scaler
)

test_selected_scaled = create_model_dataset(
    test_data,
    selected_features,
    scaler=scaler
)

train_scaled_file = save_csv(
    train_selected_scaled,
    "05_train_selected_scaled.csv"
)

validation_scaled_file = save_csv(
    validation_selected_scaled,
    "05_validation_selected_scaled.csv"
)

test_scaled_file = save_csv(
    test_selected_scaled,
    "05_test_selected_scaled.csv"
)


# 11. Create summary table

summary_rows = [
    create_summary_row(
        "Train",
        train_data,
        selected_features
    ),
    create_summary_row(
        "Validation",
        validation_data,
        selected_features
    ),
    create_summary_row(
        "Test",
        test_data,
        selected_features
    )
]

summary_data = pd.DataFrame(summary_rows)

summary_file = save_csv(
    summary_data,
    "05_post_split_summary.csv"
)

print("\nPost-split summary:")
print(summary_data)


# 12. Create report

report_file = (
    REPORT_DIR / "05_post_split_preparation_report.md"
)

removed_features = feature_selection[
    feature_selection["Selected"] == False
]

report_lines = [
    "# 05 Post-Split Preparation Report",
    "",
    "## Goal",
    "",
    (
        "Select non-redundant features using training data only "
        "and scale the selected features using training data only."
    ),
    "",
    "## Leakage Control",
    "",
    "- Feature selection uses only the training split.",
    "- The StandardScaler is fitted only on the training split.",
    (
        "- The same fitted scaler is then applied to validation "
        "and test data."
    ),
    "- Validation and test data do not influence feature selection.",
    "",
    "## Feature Selection",
    "",
    f"- Initial feature count: {len(feature_columns)}",
    f"- Selected feature count: {len(selected_features)}",
    (
        "- Removed feature count: "
        f"{len(feature_columns) - len(selected_features)}"
    ),
    f"- Correlation threshold: {CORRELATION_THRESHOLD:.2f}",
    "",
    "## Selected Features",
    ""
]

for feature in selected_features:
    report_lines.append(f"- `{feature}`")

report_lines.extend([
    "",
    "## Removed Features",
    ""
])

if removed_features.empty:
    report_lines.append("- No features were removed.")

else:
    for _, row in removed_features.iterrows():
        report_lines.append(
            f"- `{row['Feature']}` removed because of high "
            f"correlation with "
            f"`{row['Most_Correlated_Selected_Feature']}` "
            f"({row['Max_Abs_Correlation']:.2f})."
        )

report_lines.extend([
    "",
    "## Split Summary",
    ""
])

for _, row in summary_data.iterrows():
    report_lines.append(
        f"- **{row['Split']}**: "
        f"{row['Rows']} rows, "
        f"{row['Start_Date']} to {row['End_Date']}, "
        f"Target Up Rate: {row['Target_Up_Rate']:.2%}"
    )

report_lines.extend([
    "",
    "## Output Files",
    "",
    f"- `{feature_selection_file.name}`",
    f"- `{train_raw_file.name}`",
    f"- `{validation_raw_file.name}`",
    f"- `{test_raw_file.name}`",
    f"- `{train_scaled_file.name}`",
    f"- `{validation_scaled_file.name}`",
    f"- `{test_scaled_file.name}`",
    f"- `{scaler_file.name}`",
    f"- `{summary_file.name}`"
])

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(f"\nSaved report: {report_file}")


# 13. Final overview

print("\nPost-split preparation completed successfully.")
print("Next step: baseline and model training.")