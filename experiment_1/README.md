# Experiment 1: Weekly BTC/USD Trading Bot

## Goal

This experiment builds the first complete data science pipeline for a weekly
BTC/USD trading bot.

The task is a binary classification problem:

- `1`: BTC/USD close price rises next week
- `0`: BTC/USD close price does not rise next week

The experiment currently focuses on problem definition, data acquisition, and
data understanding.

## Current Status

| Step | Folder | Status |
| --- | --- | --- |
| 01 Data Acquisition | `scripts/01_data_acquisition/` | done |
| 02 Data Understanding | `scripts/02_data_understanding/` | done |
| 03 Pre-Split Data Preparation | `scripts/03_pre_split_prep/` | planned |
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

## How To Run

From the project root:

```bash
python experiment_1/scripts/01_data_acquisition/data_acquisition.py
python experiment_1/scripts/02_data_understanding/statistics_report.py
python experiment_1/scripts/02_data_understanding/plotter.py
```

If `python` is not available on your system path, use the Python interpreter
from your virtual environment or IDE.

## Next Step

The next planned step is `03_pre_split_prep`.

Planned tasks:

- merge BTC, market, and macro data by date
- create weekly return features
- create rolling volatility features
- create SMA, EMA, RSI, and MACD indicators
- include Fear & Greed sentiment as an additional feature
- create the binary target variable for next week
- save the feature and target dataset in `../../data/processed/`
