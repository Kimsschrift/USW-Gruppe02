# 06 Model Visualisations Report

## Purpose

This script creates visualisations from already saved modelling results. No model is retrained and no test result is used to change model parameters.

## Validation Model Selection

- Primary metric: `Balanced_Accuracy`
- Selected model: `LogisticRegression_C_0.1`
- Validation Balanced_Accuracy: 54.50%

## Generalisation Result

- Selected model test Balanced_Accuracy: 49.58%
- Always-Up baseline test Balanced_Accuracy: 50.00%

## Interpretation

The validation comparison shows which model was selected. The validation-versus-test comparison shows whether this advantage generalised to unseen data.

## Output Files

- `06_best_validation_models_by_group.csv`
- `06_validation_test_comparison.csv`
- `06_top_10_logistic_coefficients.csv`
- `08_final_validation_model_comparison.png`
- `10_validation_vs_test_comparison.png`
- `11_top_10_logistic_coefficients.png`