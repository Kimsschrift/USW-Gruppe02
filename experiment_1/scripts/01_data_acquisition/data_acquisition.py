"""
Data Acquisition for Weekly BTC/USD Trading Bot

Project idea:
Predict whether the BTC/USD price will rise or fall in the next week.

This script:
1. Downloads daily raw data from yfinance
2. Converts daily data into weekly data
3. Downloads Crypto Fear & Greed Index data
4. Converts sentiment data into weekly data
5. Saves the data as CSV files

Output files:
- data/raw/BTC_USD.csv
- data/raw/STOCKS.csv
- data/raw/INTEREST.csv
- data/raw/SENTIMENT.csv
"""

from pathlib import Path

import requests
import pandas as pd
import yfinance as yf
import yaml


# 1. Define project paths

CURRENT_FILE = Path(__file__).resolve()

# Path to experiment_1 folder
EXPERIMENT_DIR = CURRENT_FILE.parents[2]

# Path to main project folder, for example USW-Gruppe02
PROJECT_DIR = CURRENT_FILE.parents[3]

# Path to params.yaml
PARAMS_PATH = EXPERIMENT_DIR / "conf" / "params.yaml"


# 2. Load parameters from YAML

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

config = params["DATA_ACQUISITION"]

DATA_PATH = PROJECT_DIR / config["DATA_PATH"]

START_DATE = config["START_DATE"]
END_DATE = config["END_DATE"]
WEEKLY_FREQUENCY = config["WEEKLY_FREQUENCY"]

BTC_TICKERS = config["BTC_TICKERS"]
STOCK_TICKERS = config["STOCK_TICKERS"]
INTEREST_TICKERS = config["INTEREST_TICKERS"]

SENTIMENT_CONFIG = config["SENTIMENT"]
FEAR_GREED_API_URL = SENTIMENT_CONFIG["FEAR_GREED_API_URL"]

# Create data/raw folder if it does not exist
DATA_PATH.mkdir(parents=True, exist_ok=True)


# 3. Download daily data from yfinance

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

    # yfinance sometimes returns multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data["Symbol"] = symbol_name
    data["Ticker"] = ticker

    return data


# 4. Convert daily data to weekly data

def convert_to_weekly(daily_data):
    """
    Converts daily OHLCV data into weekly OHLCV data.

    Weekly aggregation:
    - Open: first value of the week
    - High: highest value of the week
    - Low: lowest value of the week
    - Close: last value of the week
    - Adj Close: last adjusted close of the week
    - Volume: sum of weekly volume
    - VWAP: calculated from daily High, Low, Close and Volume
    """

    if daily_data.empty:
        return pd.DataFrame()

    symbol_name = daily_data["Symbol"].iloc[0]
    ticker = daily_data["Ticker"].iloc[0]

    daily_data["Date"] = pd.to_datetime(daily_data["Date"])
    daily_data = daily_data.set_index("Date")

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

    # VWAP approximation
    typical_price = (
        daily_data["High"] + daily_data["Low"] + daily_data["Close"]
    ) / 3

    volume_sum = daily_data["Volume"].resample(WEEKLY_FREQUENCY).sum()

    vwap_numerator = (
        typical_price * daily_data["Volume"]
    ).resample(WEEKLY_FREQUENCY).sum()

    weekly_data["VWAP"] = vwap_numerator / volume_sum

    # If Volume is zero or missing, VWAP cannot be calculated.
    # In this case, use weekly typical price as fallback.
    weekly_typical_price = (
        weekly_data["High"] + weekly_data["Low"] + weekly_data["Close"]
    ) / 3

    weekly_data["VWAP"] = weekly_data["VWAP"].fillna(weekly_typical_price)

    weekly_data = weekly_data.dropna(subset=["Open", "High", "Low", "Close"])

    weekly_data = weekly_data.reset_index()

    weekly_data["Symbol"] = symbol_name
    weekly_data["Ticker"] = ticker

    return weekly_data


# 5. Download a group of tickers

def download_group(ticker_dictionary):
    """
    Downloads several tickers and combines them into one weekly DataFrame.
    """

    all_weekly_data = []

    for symbol_name, ticker in ticker_dictionary.items():
        daily_data = download_daily_data(symbol_name, ticker)
        weekly_data = convert_to_weekly(daily_data)

        if not weekly_data.empty:
            all_weekly_data.append(weekly_data)

    if len(all_weekly_data) == 0:
        return pd.DataFrame()

    combined_data = pd.concat(all_weekly_data, ignore_index=True)

    return combined_data


# 6. Download Crypto Fear & Greed Index

def download_fear_greed_index():
    """
    Downloads historical Crypto Fear & Greed Index data
    and converts it into weekly sentiment data.
    """

    print("Downloading Crypto Fear & Greed Index data...")

    response = requests.get(FEAR_GREED_API_URL, timeout=30)
    response.raise_for_status()

    json_data = response.json()
    records = json_data["data"]

    sentiment_data = pd.DataFrame(records)

    sentiment_data["Date"] = pd.to_datetime(
        sentiment_data["timestamp"].astype(int),
        unit="s"
    )

    sentiment_data["Fear_Greed_Value"] = sentiment_data["value"].astype(int)
    sentiment_data["Fear_Greed_Class"] = sentiment_data["value_classification"]

    sentiment_data = sentiment_data[
        ["Date", "Fear_Greed_Value", "Fear_Greed_Class"]
    ]

    sentiment_data = sentiment_data.sort_values("Date")

    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)

    sentiment_data = sentiment_data[
        (sentiment_data["Date"] >= start_date) &
        (sentiment_data["Date"] < end_date)
    ]

    sentiment_data = sentiment_data.set_index("Date")

    weekly_sentiment = pd.DataFrame()

    weekly_sentiment["Fear_Greed_Value"] = (
        sentiment_data["Fear_Greed_Value"]
        .resample(WEEKLY_FREQUENCY)
        .mean()
    )

    weekly_sentiment["Fear_Greed_Class"] = (
        sentiment_data["Fear_Greed_Class"]
        .resample(WEEKLY_FREQUENCY)
        .last()
    )

    weekly_sentiment = weekly_sentiment.dropna(
        subset=["Fear_Greed_Value"]
    )

    weekly_sentiment = weekly_sentiment.reset_index()

    return weekly_sentiment


# 7. BTC/USD data

print("Starting BTC/USD data acquisition")

btc_data = download_group(BTC_TICKERS)

btc_file = DATA_PATH / "BTC_USD.csv"
btc_data.to_csv(btc_file, index=False)

print(f"Saved BTC/USD data to: {btc_file}")
print(f"Shape: {btc_data.shape}")


# 8. Market data: QQQ, SPY, GLD

print("Starting market data acquisition")

stocks_data = download_group(STOCK_TICKERS)

stocks_file = DATA_PATH / "STOCKS.csv"
stocks_data.to_csv(stocks_file, index=False)

print(f"Saved market data to: {stocks_file}")
print(f"Shape: {stocks_data.shape}")


# 9. Macro / interest data: VIX, US10Y, US20Y proxy, US30Y

print("Starting macro / interest data acquisition")

interest_data = download_group(INTEREST_TICKERS)

interest_file = DATA_PATH / "INTEREST.csv"
interest_data.to_csv(interest_file, index=False)

print(f"Saved macro / interest data to: {interest_file}")
print(f"Shape: {interest_data.shape}")


# 10. Sentiment data: Crypto Fear & Greed Index

print("Starting sentiment data acquisition")

sentiment_data = download_fear_greed_index()

sentiment_file = DATA_PATH / "SENTIMENT.csv"
sentiment_data.to_csv(sentiment_file, index=False)

print(f"Saved sentiment data to: {sentiment_file}")
print(f"Shape: {sentiment_data.shape}")


# 11. Final overview

print("Data Acquisition completed successfully.")
print("Created files:")
print(f"- {btc_file}")
print(f"- {stocks_file}")
print(f"- {interest_file}")
print(f"- {sentiment_file}")

print("BTC/USD sample:")
print(btc_data.head())

print("STOCKS sample:")
print(stocks_data.head())

print("INTEREST sample:")
print(interest_data.head())

print("SENTIMENT sample:")
print(sentiment_data.head())