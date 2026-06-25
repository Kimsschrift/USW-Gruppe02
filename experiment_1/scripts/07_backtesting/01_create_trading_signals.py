"""
Create Trading Signals for Weekly BTC/USD Backtesting

This script:
1. Loads the final test predictions from step 06.
2. Converts the predicted class into a simple trading signal.
3. Calculates weekly strategy returns.
4. Saves one backtesting table.

Trading rule:
- Predicted_Target = 1 -> hold BTC for the next week
- Predicted_Target = 0 -> stay in cash for the next week

Input files:
- data/modelling/06_final_test_predictions.csv

Output files:
- data/processed/07_trading_signals.csv
"""

from pathlib import Path

import pandas as pd
import yaml


# 1. Define project paths

CURRENT_FILE = Path(__file__).resolve()

EXPERIMENT_DIR = CURRENT_FILE.parents[2]
PROJECT_DIR = EXPERIMENT_DIR.parent

PARAMS_PATH = EXPERIMENT_DIR / "conf" / "params.yaml"

modelling_data_dir = PROJECT_DIR / "data" / "modelling"
processed_data_dir = PROJECT_DIR / "data" / "processed"


# 2. Load backtesting parameters

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

backtesting_config = params.get("BACKTESTING", {})

TRANSACTION_COST = float(
    backtesting_config.get("TRANSACTION_COST", 0.0)
)


# 3. Define input and output files

PREDICTIONS_FILE = (
    modelling_data_dir / "06_final_test_predictions.csv"
)

OUTPUT_FILE = processed_data_dir / "07_trading_signals.csv"


# 4. Load predictions

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {PREDICTIONS_FILE}"
    )

predictions = pd.read_csv(PREDICTIONS_FILE)

required_columns = [
    "Date",
    "btc_close",
    "target_next_week_return",
    "Actual_Target",
    "Predicted_Target",
    "Probability_Up"
]

for column in required_columns:
    if column not in predictions.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

predictions["Date"] = pd.to_datetime(predictions["Date"])

predictions = predictions.sort_values(
    "Date"
).reset_index(drop=True)


# 5. Create trading signals and returns

signals = predictions.copy()

signals["Trading_Signal"] = signals["Predicted_Target"].astype(int)

signals["Signal_Label"] = signals["Trading_Signal"].map({
    1: "Hold BTC",
    0: "Stay in Cash"
})

# The model decides at the current week. The return belongs to
# the next week and is only used to evaluate the decision.
signals["BTC_Next_Week_Return"] = signals[
    "target_next_week_return"
]

signals["Buy_And_Hold_Return"] = signals[
    "BTC_Next_Week_Return"
]

signals["Position_Change"] = signals[
    "Trading_Signal"
].diff().fillna(signals["Trading_Signal"]).abs()

signals["Transaction_Cost"] = (
    signals["Position_Change"] * TRANSACTION_COST
)

signals["Strategy_Return"] = (
    signals["Trading_Signal"]
    * signals["BTC_Next_Week_Return"]
    - signals["Transaction_Cost"]
)

signals["Cash_Return"] = 0.0

signals["Strategy_Equity"] = (
    1 + signals["Strategy_Return"]
).cumprod()

signals["Buy_And_Hold_Equity"] = (
    1 + signals["Buy_And_Hold_Return"]
).cumprod()

signals["Cash_Equity"] = 1.0


# 6. Keep only the columns needed for backtesting

output_columns = [
    "Date",
    "btc_close",
    "Actual_Target",
    "Predicted_Target",
    "Probability_Up",
    "Trading_Signal",
    "Signal_Label",
    "BTC_Next_Week_Return",
    "Strategy_Return",
    "Buy_And_Hold_Return",
    "Cash_Return",
    "Position_Change",
    "Transaction_Cost",
    "Strategy_Equity",
    "Buy_And_Hold_Equity",
    "Cash_Equity"
]

signals[output_columns].to_csv(
    OUTPUT_FILE,
    index=False
)


# 7. Print a short console summary

print("Trading signal creation finished.")
print(f"Saved signals to: {OUTPUT_FILE}")
print(f"Rows: {len(signals)}")
print(f"Transaction cost per position change: {TRANSACTION_COST:.4%}")
print("Signal counts:")
print(signals["Signal_Label"].value_counts())
print("\nSample:")
print(signals[output_columns].head())
