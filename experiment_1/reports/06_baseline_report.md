# 06 Baseline Report

## Goal

Create a simple reference model that predicts `Up` for every validation week.

## Why a Baseline Is Needed

A later Logistic Regression or XGBoost model should perform better than this simple rule.

## Training Data Information

- Training target Up rate: 52.38%

## Validation Result

- Accuracy: 48.98%
- Balanced Accuracy: 50.00%
- Precision for Up: 48.98%
- Recall for Up: 100.00%
- F1 score for Up: 65.75%

## Interpretation

The baseline predicts only class 1. Therefore, it can identify Up weeks but cannot identify Not-Up weeks.

## Output Files

- `06_baseline_validation_metrics.csv`
- `06_baseline_validation_predictions.csv`
- `01_baseline_validation_confusion_matrix.png`