# 06 Final Model Selection and Test Report

## Model Selection

The final model was selected only using validation results. The test set was not used for model selection.

- Primary metric: `Balanced_Accuracy`
- Selected model: `LogisticRegression_C_0.1`
- Validation Balanced_Accuracy: 54.50%

## Final Test Result

- Accuracy: 48.98%
- Balanced Accuracy: 49.58%
- Precision for Up: 50.00%
- Recall for Up: 20.00%
- F1 Score for Up: 28.57%
- ROC-AUC: 61.33%

## Leakage Control

- Validation data selected the final model.
- Test data was used only once for final evaluation.
- For Logistic Regression, the scaler was refitted only on Train + Validation data.
- Test data was never used to fit the scaler or the model.

## Output Files

- `06_all_validation_model_comparison.csv`
- `06_final_test_metrics.csv`
- `06_final_test_predictions.csv`
- `06_final_selected_model.joblib`
- `09_final_test_confusion_matrix.png`
- `06_final_selected_logistic_scaler.joblib`