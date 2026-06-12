# USW-Gruppe02: BTC/USD Algorithmic Trading Bot

HTW Berlin | Unternehmenssoftware | Group Project

## Members

- Kim, Dongwoo
- Cu, Kevin Nhat Tien

## Project Overview

This project develops a simple and understandable machine learning pipeline for
a weekly BTC/USD trading bot.

The main research question is:

Can a machine learning model predict whether the BTC/USD price will rise or fall
in the next week, and can this prediction be converted into trading performance?

## Experiments

| Experiment | Focus | Status | Link |
| --- | --- | --- | --- |
| Experiment 1 | Weekly BTC/USD baseline pipeline | in progress | [Details](experiment_1/README.md) |

At the moment, the project intentionally has only one experiment. Additional
experiments should only be added when the first complete pipeline is finished
and a new modelling idea is tested.

## Data Science Process

The project follows the process from the course:

1. Problem Definition
2. Data Acquisition
3. Data Understanding
4. Pre-Split Data Preparation
5. Train / Validation / Test Split
6. Post-Split Preparation
7. Modelling
8. Validation and Testing
9. Deployment / Backtesting

## Problem Definition

**Problem description:**

Based on weekly BTC/USD price data and macroeconomic indicators, the model
predicts whether the BTC price will rise or fall in the next week.

**Target variable:**

- `1`: next week's BTC close price is higher than the current close price
- `0`: next week's BTC close price is not higher than the current close price

**Input variables:**

| Type | Data |
| --- | --- |
| Raw | BTC/USD OHLCV, VWAP |
| Market | QQQ, SPY, GLD |
| Macro | VIX, US bonds (10Y, 20Y, 30Y) |
| Sentiment | Crypto Fear & Greed Index |
| Derived | SMA, EMA, RSI, MACD, volatility |

## Data Acquisition

Raw data is downloaded with `yfinance` and stored as CSV files in `data/raw/`.

| API | Data | File |
| --- | --- | --- |
| yfinance | weekly BTC/USD OHLCV | `BTC_USD.csv` |
| yfinance | QQQ, SPY, GLD | `STOCKS.csv` |
| yfinance | VIX, US10Y, US20Y, US30Y | `INTEREST.csv` |
| alternative.me | Crypto Fear & Greed Index | `SENTIMENT.csv` |

Parameters:

- `START_DATE`: `2020-01-01`
- `END_DATE`: `2025-01-01`
- API interval: daily (`1d`)
- project frequency: weekly (`W-FRI`)

## Repository Structure

```text
USW-Gruppe02/
  data/
    raw/
    processed/
  experiment_1/
    conf/
    images/
    reports/
    scripts/
      01_data_acquisition/
      02_data_understanding/
      03_pre_split_prep/
  README.md
```

## Coding Principle

The code should stay simple enough to explain in the graded presentation:

> Write simple Python code that a beginner can easily understand. Use basic
> Python features, minimal exception handling, very little abstraction, no
> unnecessary classes, and no overengineering. Keep it short, readable, and easy
> to modify.
