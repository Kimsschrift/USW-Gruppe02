# 05 Post-Split Preparation Report

## Goal

Select non-redundant features using training data only and scale the selected features using training data only.

## Leakage Control

- Feature selection uses only the training split.
- The StandardScaler is fitted only on the training split.
- The same fitted scaler is then applied to validation and test data.
- Validation and test data do not influence feature selection.

## Feature Selection

- Initial feature count: 28
- Selected feature count: 21
- Removed feature count: 7
- Correlation threshold: 0.90

## Selected Features

- `btc_return_1w`
- `btc_return_2w`
- `btc_return_4w`
- `btc_range_pct`
- `btc_volume_change_1w`
- `btc_vwap_distance`
- `btc_volatility_4w`
- `btc_volatility_8w`
- `btc_sma_ratio_12w`
- `btc_volatility_12w`
- `btc_rsi_14w`
- `btc_macd`
- `market_GLD_return_1w`
- `market_QQQ_return_1w`
- `macro_US10Y_close`
- `macro_VIX_close`
- `macro_US10Y_change_1w`
- `macro_US20Y_change_1w`
- `macro_VIX_change_1w`
- `fear_greed_value`
- `fear_greed_change_1w`

## Removed Features

- `btc_sma_ratio_4w` removed because of high correlation with `btc_return_2w` (0.96).
- `btc_sma_ratio_8w` removed because of high correlation with `btc_return_4w` (0.93).
- `btc_macd_signal` removed because of high correlation with `btc_macd` (0.95).
- `market_SPY_return_1w` removed because of high correlation with `market_QQQ_return_1w` (0.91).
- `macro_US20Y_close` removed because of high correlation with `macro_US10Y_close` (0.99).
- `macro_US30Y_close` removed because of high correlation with `macro_US10Y_close` (0.99).
- `macro_US30Y_change_1w` removed because of high correlation with `macro_US20Y_change_1w` (0.97).

## Split Summary

- **Train**: 147 rows, 2020-04-10 to 2023-01-27, Target Up Rate: 52.38%
- **Validation**: 49 rows, 2023-02-10 to 2024-01-12, Target Up Rate: 48.98%
- **Test**: 49 rows, 2024-01-26 to 2024-12-27, Target Up Rate: 51.02%

## Output Files

- `05_feature_selection.csv`
- `05_train_selected_raw.csv`
- `05_validation_selected_raw.csv`
- `05_test_selected_raw.csv`
- `05_train_selected_scaled.csv`
- `05_validation_selected_scaled.csv`
- `05_test_selected_scaled.csv`
- `05_scaler.joblib`
- `05_post_split_summary.csv`