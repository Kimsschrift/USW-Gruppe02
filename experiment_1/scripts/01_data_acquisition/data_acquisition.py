"""
Data Acquisition for Weekly BTC/USD Trading Bot

Project idea:
Predict whether the BTC/USD price will rise or fall in the next week.

This script:
1. Downloads daily raw data from yfinance
2. Converts the data into weekly data
3. Saves the data as CSV files

Output files:
- BTC_USD.csv
- STOCKS.csv
- INTEREST.csv
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml


# 1. Define paths

# Current file:
# experiment_1/scripts/01_data_acquisition/data_acquisition.py
CURRENT_FILE = Path(__file__).resolve()

# experiment_1 folder
EXPERIMENT_DIR = CURRENT_FILE.parents[2]

# config file
PARAMS_PATH = EXPERIMENT_DIR / "conf" / "params.yaml"


# 2. Load parameters

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

config = params["DATA_ACQUISITION"]

DATA_PATH = EXPERIMENT_DIR / config["DATA_PATH"]
START_DATE = config["START_DATE"]
END_DATE = config["END_DATE"]
WEEKLY_FREQUENCY = config["WEEKLY_FREQUENCY"]

BTC_TICKERS = config["BTC_TICKERS"]
STOCK_TICKERS = config["STOCK_TICKERS"]
INTEREST_TICKERS = config["INTEREST_TICKERS"]

# Create raw data folder if it does not exist
DATA_PATH.mkdir(parents=True, exist_ok=True)


# 3. Function to download daily data from yfinance

def download_daily_data(symbol_name, ticker):
    """
    Downloads daily OHLCV data for one ticker from yfinance.
    """

    print(f"Downloading daily data for {symbol_name} ({ticker})...")

    data = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print(f"Warning: No data downloaded for {symbol_name} ({ticker})")
        return pd.DataFrame()

    # Sometimes yfinance returns multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Add identifier columns
    data["Symbol"] = symbol_name
    data["Ticker"] = ticker

    return data


# 4. Function to convert daily data to weekly data

def convert_to_weekly(daily_data):
    """
    Converts daily OHLCV data into weekly OHLCV data.

    Weekly aggregation:
    - Open: first value of the week
    - High: maximum value of the week
    - Low: minimum value of the week
    - Close: last value of the week
    - Adj Close: last adjusted close value of the week
    - Volume: sum of volume during the week
    - VWAP: volume-weighted average price approximation
    """

    if daily_data.empty:
        return pd.DataFrame()

    symbol_name = daily_data["Symbol"].iloc[0]
    ticker = daily_data["Ticker"].iloc[0]

    daily_data["Date"] = pd.to_datetime(daily_data["Date"])
    daily_data = daily_data.set_index("Date")

    # Make sure required columns exist
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in required_columns:
        if column not in daily_data.columns:
            raise ValueError(f"Missing column {column} for {symbol_name}")

    weekly_data = pd.DataFrame()

    weekly_data["Open"] = daily_data["Open"].resample(WEEKLY_FREQUENCY).first()
    weekly_data["High"] = daily_data["High"].resample(WEEKLY_FREQUENCY).max()
    weekly_data["Low"] = daily_data["Low"].resample(WEEKLY_FREQUENCY).min()
    weekly_data["Close"] = daily_data["Close"].resample(WEEKLY_FREQUENCY).last()

    if "Adj Close" in daily_data.columns:
        weekly_data["Adj Close"] = daily_data["Adj Close"].resample(WEEKLY_FREQUENCY).last()

    weekly_data["Volume"] = daily_data["Volume"].resample(WEEKLY_FREQUENCY).sum()

    # VWAP approximation:
    # typical price = average of High, Low and Close
    typical_price = (
        daily_data["High"] + daily_data["Low"] + daily_data["Close"]
    ) / 3

    volume_sum = daily_data["Volume"].resample(WEEKLY_FREQUENCY).sum()

    vwap_numerator = (
        typical_price * daily_data["Volume"]
    ).resample(WEEKLY_FREQUENCY).sum()

    weekly_data["VWAP"] = vwap_numerator / volume_sum

    # If volume is missing or zero, VWAP cannot be calculated.
    # In that case, use the weekly typical price as fallback.
    weekly_typical_price = (
        weekly_data["High"] + weekly_data["Low"] + weekly_data["Close"]
    ) / 3

    weekly_data["VWAP"] = weekly_data["VWAP"].fillna(weekly_typical_price)

    # Remove rows without price information
    weekly_data = weekly_data.dropna(subset=["Open", "High", "Low", "Close"])

    weekly_data = weekly_data.reset_index()

    weekly_data["Symbol"] = symbol_name
    weekly_data["Ticker"] = ticker

    return weekly_data


# 5. Function to download and prepare a group of tickers

def download_group(ticker_dictionary):
    """
    Downloads and converts multiple tickers into one combined weekly DataFrame.
    """

    weekly_data_list = []

    for symbol_name, ticker in ticker_dictionary.items():
        daily_data = download_daily_data(symbol_name, ticker)
        weekly_data = convert_to_weekly(daily_data)

        if not weekly_data.empty:
            weekly_data_list.append(weekly_data)

    if len(weekly_data_list) == 0:
        return pd.DataFrame()

    combined_data = pd.concat(weekly_data_list, ignore_index=True)

    return combined_data


# 6. Download BTC/USD data

print("-" * 60)
print("Starting BTC/USD data acquisition")

btc_data = download_group(BTC_TICKERS)

btc_file = DATA_PATH / "BTC_USD.csv"
btc_data.to_csv(btc_file, index=False)

print(f"Saved BTC/USD data to: {btc_file}")
print(f"BTC/USD shape: {btc_data.shape}")


# 7. Download market data: QQQ, SPY, GLD

print("-" * 60)
print("Starting market data acquisition")

stocks_data = download_group(STOCK_TICKERS)

stocks_file = DATA_PATH / "STOCKS.csv"
stocks_data.to_csv(stocks_file, index=False)

print(f"Saved market data to: {stocks_file}")
print(f"STOCKS shape: {stocks_data.shape}")


# 8. Download macro / interest data: VIX and US yields

print("-" * 60)
print("Starting macro / interest data acquisition")

interest_data = download_group(INTEREST_TICKERS)

interest_file = DATA_PATH / "INTEREST.csv"
interest_data.to_csv(interest_file, index=False)

print(f"Saved macro / interest data to: {interest_file}")
print(f"INTEREST shape: {interest_data.shape}")


# 9. Final overview

print("-" * 60)
print("Data Acquisition completed successfully.")
print("Created files:")
print(f"- {btc_file}")
print(f"- {stocks_file}")
print(f"- {interest_file}")
print("-" * 60)

print("BTC/USD sample:")
print(btc_data.head())

print("STOCKS sample:")
print(stocks_data.head())

print("INTEREST sample:")
print(interest_data.head())