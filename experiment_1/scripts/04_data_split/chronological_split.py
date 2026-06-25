"""
Chronological Data Split for Weekly BTC/USD Trading Bot

This script:
1. Loads the pre-split feature-target dataset.
2. Sorts the observations by date.
3. Creates chronological training, validation and test datasets.
4. Leaves one embargo week between splits because the target
   uses the BTC close price of the following week.
5. Saves the raw datasets for the next preparation step.

Input file:
- data/processed/03_features_targets.csv

Output files:
- data/processed/04_train_raw.csv
- data/processed/04_validation_raw.csv
- data/processed/04_test_raw.csv
- data/processed/04_embargo_train_validation.csv
- data/processed/04_embargo_validation_test.csv
- data/processed/04_split_summary.csv
- experiment_1/reports/04_data_split_report.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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
split_config = params["DATA_SPLIT"]

PROCESSED_DATA_DIR = (
    PROJECT_DIR / pre_split_config["PROCESSED_DATA_PATH"]
)

REPORT_DIR = EXPERIMENT_DIR / "reports"

INPUT_FILE = PROCESSED_DATA_DIR / "03_features_targets.csv"

TRAIN_RATIO = split_config["TRAIN_RATIO"]
VALIDATION_RATIO = split_config["VALIDATION_RATIO"]
TEST_RATIO = split_config["TEST_RATIO"]

EMBARGO_WEEKS = split_config["EMBARGO_WEEKS"]

DATE_COLUMN = "Date"
TARGET_COLUMN = "target_next_week_up"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 3. Helper functions

def validate_split_parameters():
    """
    Checks whether the split configuration is valid.
    """

    ratio_sum = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "TRAIN_RATIO, VALIDATION_RATIO and TEST_RATIO "
            f"must add up to 1.0. Current value: {ratio_sum}"
        )

    if EMBARGO_WEEKS < 0:
        raise ValueError(
            "EMBARGO_WEEKS must be zero or a positive number."
        )


def save_csv(data, file_name):
    """
    Saves a DataFrame in the processed data folder.
    """

    output_path = PROCESSED_DATA_DIR / file_name

    data.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return output_path


def create_split_summary(split_name, split_data):
    """
    Creates one summary row for a split.
    """

    return {
        "Split": split_name,
        "Rows": len(split_data),
        "Start_Date": split_data[DATE_COLUMN].min().date(),
        "End_Date": split_data[DATE_COLUMN].max().date(),
        "Target_Up_Rate": split_data[TARGET_COLUMN].mean(),
        "Up_Weeks": int(split_data[TARGET_COLUMN].sum()),
        "Not_Up_Weeks": int(
            (split_data[TARGET_COLUMN] == 0).sum()
        )
    }


def create_chronological_split(data):
    """
    Creates chronological train, validation and test splits.

    One embargo week is skipped between train and validation
    and between validation and test.
    """

    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)

    total_rows = len(data)

    # Two embargo gaps:
    # one after training and one after validation
    rows_available_for_splits = (
        total_rows - (2 * EMBARGO_WEEKS)
    )

    if rows_available_for_splits <= 0:
        raise ValueError(
            "Not enough rows after applying embargo weeks."
        )

    train_rows = int(rows_available_for_splits * TRAIN_RATIO)

    validation_rows = int(
        rows_available_for_splits * VALIDATION_RATIO
    )

    # Test receives all remaining rows.
    test_rows = (
        rows_available_for_splits
        - train_rows
        - validation_rows
    )

    if min(train_rows, validation_rows, test_rows) <= 0:
        raise ValueError(
            "At least one split would be empty."
        )

    # Training data
    train_end = train_rows

    train_data = data.iloc[:train_end].copy()

    # First embargo week
    validation_start = train_end + EMBARGO_WEEKS

    embargo_train_validation = data.iloc[
        train_end:validation_start
    ].copy()

    # Validation data
    validation_end = validation_start + validation_rows

    validation_data = data.iloc[
        validation_start:validation_end
    ].copy()

    # Second embargo week
    test_start = validation_end + EMBARGO_WEEKS

    embargo_validation_test = data.iloc[
        validation_end:test_start
    ].copy()

    # Test data
    test_data = data.iloc[test_start:].copy()

    if len(test_data) != test_rows:
        raise RuntimeError(
            "Unexpected number of rows in the test dataset."
        )

    return (
        train_data,
        validation_data,
        test_data,
        embargo_train_validation,
        embargo_validation_test
    )


# 4. Validate setup

validate_split_parameters()

print("Starting chronological data split")
print(f"Loading input file: {INPUT_FILE}")


# 5. Load pre-split dataset

data = pd.read_csv(INPUT_FILE)

if DATE_COLUMN not in data.columns:
    raise ValueError(
        f"Missing required column: {DATE_COLUMN}"
    )

if TARGET_COLUMN not in data.columns:
    raise ValueError(
        f"Missing required column: {TARGET_COLUMN}"
    )

data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])

data = data.sort_values(DATE_COLUMN).reset_index(drop=True)


# 6. Safety checks

if data[DATE_COLUMN].duplicated().any():
    raise ValueError(
        "Duplicate dates found in the input dataset."
    )

if data.isna().any().any():
    missing_values = int(data.isna().sum().sum())

    raise ValueError(
        f"The input dataset contains {missing_values} missing values. "
        "Please clean the pre-split dataset first."
    )

if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
    raise ValueError(
        "Target column must only contain 0 and 1."
    )

print(f"Input shape: {data.shape}")
print(
    f"Input period: "
    f"{data[DATE_COLUMN].min().date()} "
    f"to {data[DATE_COLUMN].max().date()}"
)


# 7. Create chronological splits

(
    train_data,
    validation_data,
    test_data,
    embargo_train_validation,
    embargo_validation_test
) = create_chronological_split(data)

print(
    "Split sizes: "
    f"Train={len(train_data)}, "
    f"Validation={len(validation_data)}, "
    f"Test={len(test_data)}, "
    f"Embargo={len(embargo_train_validation) + len(embargo_validation_test)}"
)


# 8. Save raw split datasets

train_file = save_csv(
    train_data,
    "04_train_raw.csv"
)

validation_file = save_csv(
    validation_data,
    "04_validation_raw.csv"
)

test_file = save_csv(
    test_data,
    "04_test_raw.csv"
)

embargo_train_validation_file = save_csv(
    embargo_train_validation,
    "04_embargo_train_validation.csv"
)

embargo_validation_test_file = save_csv(
    embargo_validation_test,
    "04_embargo_validation_test.csv"
)


# 9. Create and save split summary

summary_rows = [
    create_split_summary("Train", train_data),
    create_split_summary("Validation", validation_data),
    create_split_summary("Test", test_data)
]

summary_data = pd.DataFrame(summary_rows)

summary_file = save_csv(
    summary_data,
    "04_split_summary.csv"
)

print("\nSplit summary:")
print(summary_data)


# 10. Create report

report_file = REPORT_DIR / "04_data_split_report.md"

report_lines = [
    "# 04 Data Split Report",
    "",
    "## Goal",
    "",
    (
        "The feature-target dataset was split chronologically into "
        "training, validation and test data."
    ),
    "",
    "## Split Parameters",
    "",
    f"- Training ratio: {TRAIN_RATIO:.0%}",
    f"- Validation ratio: {VALIDATION_RATIO:.0%}",
    f"- Test ratio: {TEST_RATIO:.0%}",
    f"- Embargo weeks between splits: {EMBARGO_WEEKS}",
    "",
    "## Leakage Control",
    "",
    "- The data was sorted chronologically.",
    "- No random shuffle was used.",
    (
        "- One embargo week was excluded between training and validation."
    ),
    (
        "- One embargo week was excluded between validation and test."
    ),
    (
        "- Feature selection and scaling are not performed here. "
        "They will be done later using training data only."
    ),
    "",
    "## Split Summary",
    ""
]

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
    f"- `{train_file.name}`",
    f"- `{validation_file.name}`",
    f"- `{test_file.name}`",
    f"- `{embargo_train_validation_file.name}`",
    f"- `{embargo_validation_test_file.name}`",
    f"- `{summary_file.name}`"
])

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(f"\nSaved report: {report_file}")


# 11. Final overview

print("\nChronological split completed successfully.")
print("Next step: feature selection and scaling in step 05.")