# 04 Data Split Report

## Goal

The feature-target dataset was split chronologically into training, validation and test data.

## Split Parameters

- Training ratio: 60%
- Validation ratio: 20%
- Test ratio: 20%
- Embargo weeks between splits: 1

## Leakage Control

- The data was sorted chronologically.
- No random shuffle was used.
- One embargo week was excluded between training and validation.
- One embargo week was excluded between validation and test.
- Feature selection and scaling are not performed here. They will be done later using training data only.

## Split Summary

- **Train**: 147 rows, 2020-04-10 to 2023-01-27, Target Up Rate: 52.38%
- **Validation**: 49 rows, 2023-02-10 to 2024-01-12, Target Up Rate: 48.98%
- **Test**: 49 rows, 2024-01-26 to 2024-12-27, Target Up Rate: 51.02%

## Output Files

- `04_train_raw.csv`
- `04_validation_raw.csv`
- `04_test_raw.csv`
- `04_embargo_train_validation.csv`
- `04_embargo_validation_test.csv`
- `04_split_summary.csv`