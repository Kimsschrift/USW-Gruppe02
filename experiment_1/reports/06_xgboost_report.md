# 06 XGBoost Report

## Goal

Train several small XGBoost configurations on the training data and compare them on validation data.

## Why Raw Data Is Used

XGBoost is a tree-based model. Tree splits are not sensitive to feature scaling, so selected raw features are used.

## Leakage Control

- Models are trained only on training data.
- Validation data is used only for configuration selection.
- Test data is not loaded or used in this script.

## Tested Configurations

- `XGBoost_Shallow_50`: n_estimators=50, max_depth=2, learning_rate=0.05
- `XGBoost_Shallow_100`: n_estimators=100, max_depth=2, learning_rate=0.03
- `XGBoost_Depth3_100`: n_estimators=100, max_depth=3, learning_rate=0.03

## Best Validation Model

- Model: `XGBoost_Shallow_100`
- Balanced_Accuracy: 48.83%
- Accuracy: 48.98%
- F1 Score for Up: 44.44%
- ROC-AUC: 44.17%

## Output Files

- `06_xgboost_validation_metrics.csv`
- `06_xgboost_validation_predictions.csv`
- `06_xgboost_best_feature_importance.csv`
- `06_xgboost_best_validation_config.yaml`
- `06_xgboost_best_validation_model.joblib`
- `05_xgboost_validation_comparison.png`
- `06_xgboost_best_validation_confusion_matrix.png`
- `07_xgboost_best_feature_importance.png`