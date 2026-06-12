"""
Pre-Split Data Preparation: report and plots

This script:
1. Reads the feature and target dataset
2. Saves a small sample of designed features and targets
3. Saves descriptive statistics for features and targets
4. Creates plots for selected features and the target
5. Writes a markdown report for the presentation

Input file:
- data/processed/03_features_targets.csv

Output files:
- experiment_1/reports/03_pre_split_prep_report.md
- data/processed/03_feature_target_sample.csv
- data/processed/03_feature_target_summary.csv
- experiment_1/images/03_feature_overview.png
- experiment_1/images/03_target_distribution.png

How to run:
python experiment_1/scripts/03_pre_split_prep/feature_target_report.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
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

PROCESSED_DATA_DIR = PROJECT_DIR / config["PROCESSED_DATA_PATH"]
REPORT_DIR = EXPERIMENT_DIR / "reports"
IMAGE_DIR = EXPERIMENT_DIR / "images"

FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "03_features_targets.csv"
SAMPLE_PATH = PROCESSED_DATA_DIR / "03_feature_target_sample.csv"
SUMMARY_PATH = PROCESSED_DATA_DIR / "03_feature_target_summary.csv"
REPORT_PATH = REPORT_DIR / "03_pre_split_prep_report.md"

FEATURE_PLOT_PATH = IMAGE_DIR / "03_feature_overview.png"
TARGET_PLOT_PATH = IMAGE_DIR / "03_target_distribution.png"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# 3. Load data

data = pd.read_csv(FEATURE_DATA_PATH)
data["Date"] = pd.to_datetime(data["Date"])

target_columns = ["target_next_week_return", "target_next_week_up"]
info_columns = ["Date", "btc_close"]
feature_columns = [
    column for column in data.columns
    if column not in info_columns + target_columns
]


# 4. Save sample rows

sample_columns = [
    "Date",
    "btc_close",
    "btc_return_1w",
    "btc_return_4w",
    "btc_volatility_4w",
    "btc_sma_ratio_4w",
    "btc_rsi_14w",
    "market_QQQ_return_1w",
    "macro_VIX_close",
    "fear_greed_value",
    "target_next_week_return",
    "target_next_week_up",
]

sample_columns = [column for column in sample_columns if column in data.columns]
sample = data[sample_columns].head(10)
sample.to_csv(SAMPLE_PATH, index=False)


# 5. Save descriptive statistics

summary_rows = []

for column in feature_columns + target_columns:
    summary_rows.append(
        {
            "column": column,
            "missing_values": int(data[column].isna().sum()),
            "mean": data[column].mean(),
            "std": data[column].std(),
            "min": data[column].min(),
            "median": data[column].median(),
            "max": data[column].max(),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)


# 6. Create feature overview plot

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(data["Date"], data["btc_return_1w"], label="BTC return 1w")
plt.axhline(0, color="black", linewidth=1)
plt.title("BTC weekly return")
plt.ylabel("Return")
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 2)
plt.plot(data["Date"], data["btc_volatility_4w"], color="orange", label="BTC volatility 4w")
plt.title("BTC rolling volatility")
plt.ylabel("Volatility")
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 3)
plt.plot(data["Date"], data["btc_rsi_14w"], color="green", label="RSI 14w")
plt.axhline(70, color="red", linestyle="--", linewidth=1)
plt.axhline(30, color="blue", linestyle="--", linewidth=1)
plt.title("BTC RSI")
plt.ylabel("RSI")
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 4)
plt.plot(data["Date"], data["fear_greed_value"], color="purple", label="Fear & Greed")
plt.title("Crypto Fear & Greed Index")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FEATURE_PLOT_PATH, dpi=150)
plt.close()


# 7. Create target distribution plot

target_counts = data["target_next_week_up"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["0: not up", "1: up"],
    [target_counts.get(0, 0), target_counts.get(1, 0)],
    color=["#d95f02", "#1b9e77"],
)

plt.title("Target distribution")
plt.ylabel("Number of weeks")
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(TARGET_PLOT_PATH, dpi=150)
plt.close()


# 8. Write markdown report

target_rate = data["target_next_week_up"].mean()

report_lines = [
    "# 03 Pre-Split Data Preparation Report",
    "",
    "## Goal",
    "",
    "This step prepares one modelling table before the chronological split.",
    "All features use only information from the current week or earlier weeks.",
    "The target looks one week into the future and is not used as a feature.",
    "",
    "## Input and Output Files",
    "",
    "Input files:",
    "",
    "- `data/raw/BTC_USD.csv`",
    "- `data/raw/STOCKS.csv`",
    "- `data/raw/INTEREST.csv`",
    "- `data/raw/SENTIMENT.csv`",
    "",
    "Output files:",
    "",
    "- `data/processed/03_features_targets.csv`",
    "- `data/processed/03_feature_target_sample.csv`",
    "- `data/processed/03_feature_target_summary.csv`",
    "- `experiment_1/reports/03_pre_split_prep_report.md`",
    "- `experiment_1/images/03_feature_overview.png`",
    "- `experiment_1/images/03_target_distribution.png`",
    "",
    "## Parameters",
    "",
    f"- Target horizon: {config['TARGET_HORIZON_WEEKS']} week",
    f"- BTC return windows: {config['BTC_RETURN_WINDOWS']} weeks",
    f"- Rolling windows: {config['ROLLING_WINDOWS']} weeks",
    f"- RSI window: {config['RSI_WINDOW']} weeks",
    (
        "- MACD windows: "
        f"fast={config['MACD_FAST_WINDOW']}, "
        f"slow={config['MACD_SLOW_WINDOW']}, "
        f"signal={config['MACD_SIGNAL_WINDOW']}"
    ),
    "",
    "## Process",
    "",
    "1. Load weekly raw CSV files.",
    "2. Keep BTC as the main row table with one row per week.",
    "3. Pivot market and macro data so each symbol becomes a feature column.",
    "4. Merge all datasets by `Date`.",
    "5. Create BTC return, volatility, trend, volume, VWAP, RSI, and MACD features.",
    "6. Create market return features from QQQ, SPY, and GLD.",
    "7. Create macro level and weekly change features from VIX and bond indicators.",
    "8. Add Fear & Greed sentiment level and weekly change.",
    "9. Create the binary target for next week's BTC direction.",
    "10. Drop rows with missing values from rolling windows and the final unknown target.",
    "",
    "## Target Definition",
    "",
    "`target_next_week_up` is `1` if next week's BTC close is higher than the current BTC close.",
    "Otherwise it is `0`.",
    "",
    "Formula:",
    "",
    "```text",
    "target_next_week_return = BTC_Close[t + 1] / BTC_Close[t] - 1",
    "target_next_week_up = 1 if target_next_week_return > 0 else 0",
    "```",
    "",
    "## Feature Selection Intuition",
    "",
    "- BTC returns describe short-term momentum.",
    "- Rolling volatility describes market risk.",
    "- SMA ratios, RSI, and MACD describe trend and overbought or oversold situations.",
    "- Volume change and VWAP distance describe trading activity and price pressure.",
    "- QQQ, SPY, and GLD returns describe related market movements.",
    "- VIX and bond indicators describe macro risk and interest-rate pressure.",
    "- Fear & Greed describes crypto market sentiment.",
    "",
    "## Dataset Overview",
    "",
    f"- Rows after cleaning: {len(data)}",
    f"- Feature columns: {len(feature_columns)}",
    f"- Start date: {data['Date'].min().date()}",
    f"- End date: {data['Date'].max().date()}",
    f"- Target up rate: {target_rate:.2%}",
    "",
    "## Sample of Designed Features and Target",
    "",
    "```text",
    sample.round(4).to_string(index=False),
    "```",
    "",
    "## Descriptive Statistics",
    "",
    "```text",
    summary.round(4).to_string(index=False),
    "```",
    "",
    "## Findings",
    "",
    "1. The final dataset is smaller than the raw dataset because rolling features need past weeks.",
    "2. The target is not perfectly balanced, so baseline models are important in the modelling step.",
    "3. BTC return and volatility features are useful because BTC has strong weekly movements.",
    "4. Market, macro, and sentiment features add context beyond BTC price history.",
    "5. All scaling must happen after the chronological split to avoid data leakage.",
]

REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


# 9. Final overview

print("Pre-split feature and target report finished.")
print(f"Saved report to: {REPORT_PATH}")
print(f"Saved sample to: {SAMPLE_PATH}")
print(f"Saved summary to: {SUMMARY_PATH}")
print(f"Saved feature plot to: {FEATURE_PLOT_PATH}")
print(f"Saved target plot to: {TARGET_PLOT_PATH}")
