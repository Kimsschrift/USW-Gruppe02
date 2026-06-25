# 06 Logistic Regression Report

## Goal

Train several Logistic Regression configurations on the training split and compare them on the validation split.

## Why Scaled Data Is Used

Logistic Regression is sensitive to feature scales. Therefore, the scaled datasets from step 05 are used.

## Leakage Control

- Only training data is used to fit the models.
- Validation data is used only to compare C values.
- Test data is not loaded or used in this script.

## Tested Configurations

- Logistic Regression with C = 0.01
- Logistic Regression with C = 0.1
- Logistic Regression with C = 1.0
- Logistic Regression with C = 10.0

## Best Validation Model

- Model: `LogisticRegression_C_0.1`
- Selected C value: 0.1
- Balanced_Accuracy: 54.50%
- Accuracy: 55.10%
- F1 Score for Up: 35.29%

## Output Files

- `06_logistic_validation_metrics.csv`
- `06_logistic_validation_predictions.csv`
- `06_logistic_best_coefficients.csv`
- `06_logistic_best_validation_model.joblib`
- `02_logistic_validation_comparison.png`
- `03_logistic_best_validation_confusion_matrix.png`
- `04_logistic_best_coefficients.png`