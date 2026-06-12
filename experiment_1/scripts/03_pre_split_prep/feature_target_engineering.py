"""
Pre-Split Data Preparation: feature and target engineering

This script:
1. Reads the raw weekly CSV files
2. Merges BTC, market, macro, and sentiment data by Date
3. Creates features from current and past weekly values only
4. Creates the target for next week's BTC price direction
5. Saves one prepared feature and target dataset

Input files:
- data/raw/BTC_USD.csv
- data/raw/STOCKS.csv
- data/raw/INTEREST.csv
- data/raw/SENTIMENT.csv

Output file:
- data/processed/03_features_targets.csv

How to run:
python experiment_1/scripts/03_pre_split_prep/feature_target_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# 1. Define project paths

CURRENT_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = CURRENT_FILE.parents[2]
PROJECT_DIR = CURRENT_FILE.parents[3]
PARAMS_PATH = EXPERIMENT_DIR / "conf" / "params.yaml"


# 2. Load parameters

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

config = params["PRE_SPLIT_PREP"]

RAW_DATA_DIR = PROJECT_DIR / config["RAW_DATA_PATH"]
PROCESSED_DATA_DIR = PROJECT_DIR / config["PROCESSED_DATA_PATH"]

TARGET_HORIZON_WEEKS = config["TARGET_HORIZON_WEEKS"]
BTC_RETURN_WINDOWS = config["BTC_RETURN_WINDOWS"]
ROLLING_WINDOWS = config["ROLLING_WINDOWS"]
RSI_WINDOW = config["RSI_WINDOW"]
MACD_FAST_WINDOW = config["MACD_FAST_WINDOW"]
MACD_SLOW_WINDOW = config["MACD_SLOW_WINDOW"]
MACD_SIGNAL_WINDOW = config["MACD_SIGNAL_WINDOW"]

OUTPUT_PATH = PROCESSED_DATA_DIR / "03_features_targets.csv"


# 3. Helper functions

def load_csv(file_path):
    """Load one CSV file and convert Date to datetime."""
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    return data.sort_values("Date")


def pivot_close(data, prefix):
    """Create one close-price table with Date as rows and Symbol as columns."""
    close_data = data.pivot(index="Date", columns="Symbol", values="Close")
    close_data = close_data.sort_index()
    close_data.columns = [f"{prefix}_{column}_close" for column in close_data.columns]
    return close_data


def calculate_rsi(close_price, window):
    """Calculate a simple RSI indicator from past price changes."""
    price_change = close_price.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(window).mean()
    average_loss = losses.rolling(window).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


# 4. Load raw weekly data

btc_data = load_csv(RAW_DATA_DIR / "BTC_USD.csv")
stocks_data = load_csv(RAW_DATA_DIR / "STOCKS.csv")
interest_data = load_csv(RAW_DATA_DIR / "INTEREST.csv")
sentiment_data = load_csv(RAW_DATA_DIR / "SENTIMENT.csv")


# 5. Build the base table

btc = btc_data[btc_data["Symbol"] == "BTC_USD"].copy()
btc = btc.sort_values("Date")

features = pd.DataFrame()
features["Date"] = btc["Date"]
features["btc_close"] = btc["Close"]

btc_close = btc["Close"]
btc_high = btc["High"]
btc_low = btc["Low"]
btc_volume = btc["Volume"]
btc_vwap = btc["VWAP"]


# 6. Create BTC price and volume features

for window in BTC_RETURN_WINDOWS:
    features[f"btc_return_{window}w"] = btc_close.pct_change(window)

features["btc_range_pct"] = (btc_high - btc_low) / btc_close
features["btc_volume_change_1w"] = btc_volume.pct_change()
features["btc_vwap_distance"] = (btc_close / btc_vwap) - 1

for window in ROLLING_WINDOWS:
    rolling_mean = btc_close.rolling(window).mean()
    rolling_std = features["btc_return_1w"].rolling(window).std()

    features[f"btc_sma_ratio_{window}w"] = (btc_close / rolling_mean) - 1
    features[f"btc_volatility_{window}w"] = rolling_std

features[f"btc_rsi_{RSI_WINDOW}w"] = calculate_rsi(btc_close, RSI_WINDOW)

fast_ema = btc_close.ewm(span=MACD_FAST_WINDOW, adjust=False).mean()
slow_ema = btc_close.ewm(span=MACD_SLOW_WINDOW, adjust=False).mean()
macd = fast_ema - slow_ema
macd_signal = macd.ewm(span=MACD_SIGNAL_WINDOW, adjust=False).mean()

features["btc_macd"] = macd
features["btc_macd_signal"] = macd_signal


# 7. Add market return features

market_close = pivot_close(stocks_data, "market")
market_returns = market_close.pct_change()
market_returns.columns = [
    column.replace("_close", "_return_1w") for column in market_returns.columns
]

features = features.merge(
    market_returns.reset_index(),
    on="Date",
    how="left"
)


# 8. Add macro level and change features

macro_close = pivot_close(interest_data, "macro")
macro_change = macro_close.diff()
macro_change.columns = [
    column.replace("_close", "_change_1w") for column in macro_change.columns
]

features = features.merge(
    macro_close.reset_index(),
    on="Date",
    how="left"
)

features = features.merge(
    macro_change.reset_index(),
    on="Date",
    how="left"
)


# 9. Add sentiment features

sentiment = sentiment_data[["Date", "Fear_Greed_Value"]].copy()
sentiment = sentiment.sort_values("Date")
sentiment["fear_greed_change_1w"] = sentiment["Fear_Greed_Value"].diff()
sentiment = sentiment.rename(columns={"Fear_Greed_Value": "fear_greed_value"})

features = features.merge(sentiment, on="Date", how="left")


# 10. Create target for next week

features["target_next_week_return"] = (
    features["btc_close"].shift(-TARGET_HORIZON_WEEKS) / features["btc_close"] - 1
)

features["target_next_week_up"] = np.where(
    features["target_next_week_return"] > 0,
    1,
    0
)

features.loc[
    features["target_next_week_return"].isna(),
    "target_next_week_up"
] = np.nan


# 11. Clean final data

features = features.replace([np.inf, -np.inf], np.nan)
features = features.dropna()
features["target_next_week_up"] = features["target_next_week_up"].astype(int)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT_PATH, index=False)


# 12. Final overview

print("Pre-split feature and target engineering finished.")
print(f"Saved dataset to: {OUTPUT_PATH}")
print(f"Shape: {features.shape}")
print("Sample:")
print(features.head())
