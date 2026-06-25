# Experiment 1: Weekly BTC/USD Trading Bot

## Goal

This experiment builds the first complete data science pipeline for a weekly
BTC/USD trading bot.

The task is a binary classification problem:

- `1`: BTC/USD close price rises next week
- `0`: BTC/USD close price does not rise next week

The experiment currently focuses on problem definition, data acquisition, data
understanding, and pre-split data preparation.

## Current Status

| Step | Folder | Status |
| --- | --- | --- |
| 01 Data Acquisition | `scripts/01_data_acquisition/` | done |
| 02 Data Understanding | `scripts/02_data_understanding/` | done |
| 03 Pre-Split Data Preparation | `scripts/03_pre_split_prep/` | done |
| 04 Split Data | `scripts/04_split_data/` | planned |
| 05 Post-Split Preparation | `scripts/05_preparation/` | planned |
| 06 Modelling | `scripts/06_modelling/` | planned |
| 07 Deployment / Backtesting | `scripts/07_deployment/` | planned |

## Data

Raw data is stored in `../../data/raw/`.

| File | Content |
| --- | --- |
| `BTC_USD.csv` | weekly BTC/USD OHLCV and VWAP |
| `STOCKS.csv` | weekly QQQ, SPY, and GLD data |
| `INTEREST.csv` | weekly VIX, US10Y, US20Y, and US30Y data |
| `SENTIMENT.csv` | weekly Crypto Fear & Greed Index data |

The data covers the period from `2020-01-03` to `2025-01-03`.
Each symbol has 262 weekly observations.

## 01 Data Acquisition

Script:

- `scripts/01_data_acquisition/data_acquisition.py`

The script downloads daily data with `yfinance`, aggregates it to weekly data
with week ending on Friday (`W-FRI`), and stores the result as CSV files.

Configuration:

- `conf/params.yaml`

Main parameters:

- start date: `2020-01-01`
- end date: `2025-01-01`
- interval from API: daily (`1d`)
- target frequency: weekly (`W-FRI`)

## 02 Data Understanding

Scripts:

- `scripts/02_data_understanding/statistics_report.py`
- `scripts/02_data_understanding/plotter.py`

Outputs:

- `reports/02_data_understanding_report.md`
- `../../data/processed/02_data_understanding_summary.csv`
- `images/02_raw_data_overview.png`
- `images/02_performance_comparison.png`
- `images/02_correlation_matrix.png`
- `images/02_sentiment_overview.png`

This step covers the data understanding requirements:

| Requirement | Output |
| --- | --- |
| Explain relevant data columns | report with column overview |
| Show descriptive statistics | report with `describe()` tables |
| Show relevant plots | raw data, performance, and correlation plots |
| Present findings | findings listed below |

### Findings

1. All datasets cover the same weekly period from `2020-01-03` to `2025-01-03`.
2. Each symbol has 262 weekly observations, so the datasets can be merged by date.
3. No missing values were found in the raw CSV files.
4. No duplicate `Date` and `Symbol` rows were found.
5. BTC/USD shows much stronger volatility than QQQ, SPY, and GLD.
6. BTC weekly returns have a weak positive correlation with QQQ and SPY.
7. BTC weekly returns have a negative correlation with VIX.
8. The Fear & Greed Index gives an additional weekly sentiment signal from 0 to 100.
9. Because absolute price levels are very different, returns and normalized
   indicators should be used for modelling.

## 03 Pre-Split Data Preparation

Scripts:

- `scripts/03_pre_split_prep/feature_target_engineering.py`
- `scripts/03_pre_split_prep/feature_target_report.py`

Outputs:

- `../../data/processed/03_features_targets.csv`
- `../../data/processed/03_feature_target_sample.csv`
- `../../data/processed/03_feature_target_summary.csv`
- `reports/03_pre_split_prep_report.md`
- `images/03_feature_overview.png`
- `images/03_target_distribution.png`

This step prepares the modelling table before the chronological train,
validation, and test split.

### Process

1. Load weekly raw CSV files from `../../data/raw/`.
2. Use BTC/USD as the main weekly table.
3. Pivot market and macro data so each symbol becomes one feature column.
4. Merge BTC, market, macro, and sentiment data by `Date`.
5. Create feature columns from current and past weekly data only.
6. Create the binary target from the next week's BTC close price.
7. Drop rows with missing values from rolling windows and the final unknown
   target.
8. Save one clean feature and target dataset.

### Parameters

The parameters are stored in `conf/params.yaml`.

| Parameter | Value | Purpose |
| --- | --- | --- |
| `TARGET_HORIZON_WEEKS` | `1` | Predict one week ahead |
| `BTC_RETURN_WINDOWS` | `1, 2, 4` | Short-term BTC momentum |
| `ROLLING_WINDOWS` | `4, 8, 12` | Rolling trend and volatility |
| `RSI_WINDOW` | `14` | RSI momentum indicator |
| `MACD_FAST_WINDOW` | `12` | Fast EMA for MACD |
| `MACD_SLOW_WINDOW` | `26` | Slow EMA for MACD |
| `MACD_SIGNAL_WINDOW` | `9` | MACD signal line |

### Designed Features

BTC price features:

- weekly returns over 1, 2, and 4 weeks
- weekly high-low range
- volume change
- distance between close price and VWAP
- SMA ratios over 4, 8, and 12 weeks
- rolling volatility over 4, 8, and 12 weeks
- RSI and MACD indicators

Market features:

- QQQ, SPY, and GLD weekly returns

Macro features:

- VIX, US10Y, US20Y, and US30Y weekly levels
- weekly changes in those macro indicators

Sentiment features:

- Crypto Fear & Greed value
- weekly change in Fear & Greed value

### Target

The target is `target_next_week_up`.

```text
target_next_week_return = BTC_Close[t + 1] / BTC_Close[t] - 1
target_next_week_up = 1 if target_next_week_return > 0 else 0
```

This means:

- `1`: BTC/USD rises next week
- `0`: BTC/USD does not rise next week

The target uses future information, but it is only used as the label. It is not
used as a model feature.

### Feature Selection Intuition

The selected features are simple and explainable:

1. BTC returns describe short-term momentum.
2. Rolling volatility describes market risk.
3. SMA ratios, RSI, and MACD describe trend and overbought or oversold states.
4. Volume change and VWAP distance describe trading activity and price pressure.
5. QQQ, SPY, and GLD returns describe related market movements.
6. VIX and bond indicators describe macro risk and interest-rate pressure.
7. Fear & Greed adds crypto sentiment information.

### Presentation Outputs

| Requirement | Output |
| --- | --- |
| Clearly describe the process | `reports/03_pre_split_prep_report.md` |
| Which parameters did you use? | `conf/params.yaml` and report |
| Show samples of features and targets | `03_feature_target_sample.csv` and report |
| Show descriptive statistics | `03_feature_target_summary.csv` and report |
| Show relevant plots | `03_feature_overview.png`, `03_target_distribution.png` |
| Explain feature selection | README and report |

### Findings

1. Rows are removed at the beginning because rolling indicators need past weeks.
2. The last row is removed because the next week's target is unknown.
3. The target distribution must be checked before modelling because class
   imbalance can make accuracy misleading.
4. Scaling is intentionally not done here because scalers must be fit only on
   training data after the chronological split.

## How To Run

From the project root:

```bash
python experiment_1/scripts/01_data_acquisition/data_acquisition.py
python experiment_1/scripts/02_data_understanding/statistics_report.py
python experiment_1/scripts/02_data_understanding/plotter.py
python experiment_1/scripts/03_pre_split_prep/feature_target_engineering.py
python experiment_1/scripts/03_pre_split_prep/feature_target_report.py
```

If `python` is not available on your system path, use the Python interpreter
from your virtual environment or IDE.

## 04 Chronological Data Split

Script:

* `scripts/04_data_split/chronological_split.py`

### Goal

The clean feature-target dataset is split chronologically into training,
validation, and test data. Random shuffling is not used because the observations
form a time series.

### Split Setup

| Parameter        | Value                 |
| ---------------- | --------------------- |
| Training ratio   | 60%                   |
| Validation ratio | 20%                   |
| Test ratio       | 20%                   |
| Embargo gap      | 1 week between splits |

An embargo week is removed after the training period and after the validation
period. This prevents the one-week-ahead target of the final observation in a
split from overlapping with the following split.

### Split Result

| Split      | Rows | Period                   | Target Up Rate |
| ---------- | ---: | ------------------------ | -------------: |
| Train      |  147 | 2020-04-10 to 2023-01-27 |         52.38% |
| Validation |   49 | 2023-02-10 to 2024-01-12 |         48.98% |
| Test       |   49 | 2024-01-26 to 2024-12-27 |         51.02% |

### Outputs

* `../data/processed/04_train_raw.csv`
* `../data/processed/04_validation_raw.csv`
* `../data/processed/04_test_raw.csv`
* `../data/processed/04_embargo_train_validation.csv`
* `../data/processed/04_embargo_validation_test.csv`
* `../data/processed/04_split_summary.csv`
* `reports/04_data_split_report.md`

### Leakage Control

* Data is sorted by date before splitting.
* No random shuffle is used.
* One embargo week is excluded between Train and Validation.
* One embargo week is excluded between Validation and Test.
* Feature selection and scaling are intentionally performed only after this
  split, using training data only.

## 05 Post-Split Preparation

Script:

* `scripts/05_post_split_preparation/feature_selection_scaling.py`

### Goal

This step selects non-redundant features and scales the selected features
without using information from the validation or test periods during fitting.

### Process

1. Load the chronological Train, Validation, and Test datasets.
2. Use only the Train split to detect highly correlated features.
3. Remove a feature when its absolute correlation with an already retained
   feature is above `0.90`.
4. Fit `StandardScaler` only on the selected training features.
5. Apply the same fitted scaler to Validation and Test data.
6. Save selected raw datasets and scaled datasets.

### Feature Selection Result

| Result                 | Value |
| ---------------------- | ----: |
| Initial feature count  |    28 |
| Selected feature count |    21 |
| Removed feature count  |     7 |
| Correlation threshold  |  0.90 |

Examples of removed redundant features include highly correlated SMA ratios,
MACD signal, SPY return, and long-term Treasury proxy features.

### Outputs

* `../data/processed/05_feature_selection.csv`
* `../data/processed/05_train_selected_raw.csv`
* `../data/processed/05_validation_selected_raw.csv`
* `../data/processed/05_test_selected_raw.csv`
* `../data/processed/05_train_selected_scaled.csv`
* `../data/processed/05_validation_selected_scaled.csv`
* `../data/processed/05_test_selected_scaled.csv`
* `../data/processed/05_scaler.joblib`
* `../data/processed/05_post_split_summary.csv`
* `reports/05_post_split_preparation_report.md`

### Usage in Modelling

* Logistic Regression uses the selected and scaled datasets.
* XGBoost uses the selected raw datasets because tree-based models do not need
  feature scaling.

### Leakage Control

* Feature selection is based only on the training split.
* The scaler is fitted only on the training split.
* Validation and Test data are transformed with the same fitted scaler.
* Validation and Test data do not influence feature selection or scaler fitting.

## 06 Modelling, Validation and Test Evaluation

Scripts:

* `scripts/06_modelling/01_baseline.py`
* `scripts/06_modelling/02_logistic_regression.py`
* `scripts/06_modelling/03_xgboost.py`
* `scripts/06_modelling/04_model_selection_and_test.py`
* `scripts/06_modelling/05_model_visualisations.py`
* `scripts/06_modelling/model_utils.py`

### Goal

Several classification approaches are compared on the Validation split. The best
model is selected using `Balanced_Accuracy`, retrained on Train + Validation,
and then evaluated once on the untouched Test split.

### Models Compared on the Validation Split

| Model group         | Variants                                          |
| ------------------- | ------------------------------------------------- |
| Always-Up Baseline  | Always predicts `Up`                              |
| Logistic Regression | `C = 0.01, 0.1, 1.0, 10.0`                        |
| XGBoost             | Three small tree configurations with depth 2 or 3 |

`Balanced_Accuracy` is used as the main metric because it gives equal weight to
both classes: `Up` and `Not Up`.

### Validation Results

| Model                          |   Accuracy | Balanced Accuracy | ROC-AUC |
| ------------------------------ | ---------: | ----------------: | ------: |
| Always-Up Baseline             |     48.98% |            50.00% |  50.00% |
| Logistic Regression, `C = 0.1` | **55.10%** |        **54.50%** |  52.17% |
| Best XGBoost: Shallow 100      |     48.98% |            48.83% |  44.17% |

The Logistic Regression with `C = 0.1` achieved the best validation Balanced
Accuracy and was selected as the final model.

### Final Test Result

After selection, the Logistic Regression was retrained on Train + Validation
only. The Test split was then used once for final evaluation.

| Metric            | Final Test Result |
| ----------------- | ----------------: |
| Accuracy          |            48.98% |
| Balanced Accuracy |            49.58% |
| Precision for Up  |            50.00% |
| Recall for Up     |            20.00% |
| F1 Score for Up   |            28.57% |
| ROC-AUC           |            61.33% |

The final model did not outperform the Always-Up baseline on the Test split.
It correctly identified many `Not Up` weeks but missed many actual `Up` weeks.
This means that the current feature set and models do not yet provide a robust
weekly BTC direction prediction.

### Modelling Outputs

Metrics, predictions, selected-model files, and model-specific reports are
stored in `../data/modelling/`, `models/`, `reports/`, and
`images/06_modelling/`.

Key outputs:

* `../data/modelling/06_all_validation_model_comparison.csv`
* `../data/modelling/06_final_test_metrics.csv`
* `../data/modelling/06_final_test_predictions.csv`
* `models/06_final_selected_model.joblib`
* `models/06_final_selected_logistic_scaler.joblib`
* `images/06_modelling/08_final_validation_model_comparison.png`
* `images/06_modelling/09_final_test_confusion_matrix.png`
* `images/06_modelling/10_validation_vs_test_comparison.png`
* `images/06_modelling/11_top_10_logistic_coefficients.png`
* `reports/06_final_model_selection_and_test_report.md`

### Leakage Control

* Models are trained on the Train split only while comparing configurations.
* The Validation split is used only to select the final model.
* The Test split is not used for model selection or parameter tuning.
* For the final Logistic Regression, the scaler is refitted only on Train +
  Validation data.
* The Test split is used once for final evaluation and should not be used to
  tune the model further.

## How To Run

Run the scripts from the project root in this order:

```bash
python experiment_1/scripts/01_data_acquisition/data_acquisition.py
python experiment_1/scripts/02_data_understanding/statistics_report.py
python experiment_1/scripts/02_data_understanding/plotter.py
python experiment_1/scripts/03_pre_split_prep/feature_target_engineering.py
python experiment_1/scripts/03_pre_split_prep/feature_target_report.py
python experiment_1/scripts/04_data_split/chronological_split.py
python experiment_1/scripts/05_post_split_preparation/feature_selection_scaling.py
python experiment_1/scripts/06_modelling/01_baseline.py
python experiment_1/scripts/06_modelling/02_logistic_regression.py
python experiment_1/scripts/06_modelling/03_xgboost.py
python experiment_1/scripts/06_modelling/04_model_selection_and_test.py
python experiment_1/scripts/06_modelling/05_model_visualisations.py
```

## Next Step

The next planned step is `07_backtesting`.

The final test predictions can be used without retraining the model to compare:

* model strategy: hold BTC only when the predicted target is `Up`
* BTC buy-and-hold strategy
* cash strategy

Possible backtesting outputs include total return, an equity curve, number of
weeks invested, maximum drawdown, and optional risk-adjusted performance
metrics.
