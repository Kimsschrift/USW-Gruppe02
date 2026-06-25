"""
Backtest Evaluation for Weekly BTC/USD Trading Bot

This script:
1. Loads the trading signals from step 07.
2. Compares the model strategy with Buy-and-Hold and Cash.
3. Calculates simple backtesting metrics.
4. Saves summary tables, plots, and a report.

Input files:
- data/processed/07_trading_signals.csv

Output files:
- data/processed/07_backtest_summary.csv
- experiment_1/images/07_backtest_equity_curve.png
- experiment_1/images/07_backtest_weekly_returns.png
- experiment_1/reports/07_backtesting_report.md
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# 1. Define project paths

CURRENT_FILE = Path(__file__).resolve()

EXPERIMENT_DIR = CURRENT_FILE.parents[2]
PROJECT_DIR = EXPERIMENT_DIR.parent

PARAMS_PATH = EXPERIMENT_DIR / "conf" / "params.yaml"

processed_data_dir = PROJECT_DIR / "data" / "processed"
image_dir = EXPERIMENT_DIR / "images"
report_dir = EXPERIMENT_DIR / "reports"


# 2. Load backtesting parameters

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

backtesting_config = params.get("BACKTESTING", {})

INITIAL_CAPITAL = float(
    backtesting_config.get("INITIAL_CAPITAL", 1.0)
)

TRANSACTION_COST = float(
    backtesting_config.get("TRANSACTION_COST", 0.0)
)


# 3. Define input and output files

SIGNALS_FILE = processed_data_dir / "07_trading_signals.csv"

SUMMARY_FILE = processed_data_dir / "07_backtest_summary.csv"

EQUITY_PLOT_FILE = image_dir / "07_backtest_equity_curve.png"

RETURNS_PLOT_FILE = image_dir / "07_backtest_weekly_returns.png"

REPORT_FILE = report_dir / "07_backtesting_report.md"


# 4. Helper functions

def calculate_max_drawdown(equity_curve):
    """
    Calculates the worst percentage drop from a previous peak.
    """

    equity_with_start = pd.concat(
        [
            pd.Series([INITIAL_CAPITAL]),
            equity_curve.reset_index(drop=True)
        ],
        ignore_index=True
    )

    running_peak = equity_with_start.cummax()
    drawdown = equity_with_start / running_peak - 1

    return drawdown.min()


def calculate_strategy_metrics(
    strategy_name,
    return_column,
    equity_column,
    signals
):
    """
    Calculates simple backtesting metrics for one strategy.
    """

    weekly_returns = signals[return_column]
    equity_curve = signals[equity_column]

    invested_weeks = len(signals)
    number_of_trades = 1

    if strategy_name == "Model Strategy":
        invested_weeks = int(signals["Trading_Signal"].sum())
        number_of_trades = int(signals["Position_Change"].sum())

    if strategy_name == "Cash":
        invested_weeks = 0
        number_of_trades = 0

    total_return = equity_curve.iloc[-1] / INITIAL_CAPITAL - 1

    average_weekly_return = weekly_returns.mean()

    weekly_volatility = weekly_returns.std()

    max_drawdown = calculate_max_drawdown(equity_curve)

    positive_weeks = int((weekly_returns > 0).sum())

    if strategy_name == "Model Strategy" and invested_weeks > 0:
        win_rate = positive_weeks / invested_weeks
    elif strategy_name != "Model Strategy" and len(weekly_returns) > 0:
        win_rate = positive_weeks / len(weekly_returns)
    else:
        win_rate = 0.0

    exposure_rate = invested_weeks / len(signals)

    return {
        "Strategy": strategy_name,
        "Total_Return": total_return,
        "Final_Equity": equity_curve.iloc[-1],
        "Average_Weekly_Return": average_weekly_return,
        "Weekly_Volatility": weekly_volatility,
        "Max_Drawdown": max_drawdown,
        "Positive_Weeks": positive_weeks,
        "Total_Weeks": len(signals),
        "Invested_Weeks": invested_weeks,
        "Exposure_Rate": exposure_rate,
        "Number_Of_Trades": number_of_trades,
        "Win_Rate": win_rate
    }


def save_equity_curve_plot(signals):
    """
    Saves a line chart of the strategy equity curves.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        signals["Date"],
        signals["Strategy_Equity"],
        label="Model Strategy"
    )

    plt.plot(
        signals["Date"],
        signals["Buy_And_Hold_Equity"],
        label="Buy and Hold"
    )

    plt.plot(
        signals["Date"],
        signals["Cash_Equity"],
        label="Cash"
    )

    plt.title("Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity, start = 1.0")
    plt.legend()
    plt.tight_layout()

    plt.savefig(EQUITY_PLOT_FILE)
    plt.close()

    return EQUITY_PLOT_FILE


def save_weekly_returns_plot(signals):
    """
    Saves a bar chart comparing weekly model and buy-and-hold returns.
    """

    plt.figure(figsize=(11, 6))

    plt.bar(
        signals["Date"],
        signals["Buy_And_Hold_Return"],
        label="Buy and Hold",
        alpha=0.5
    )

    plt.bar(
        signals["Date"],
        signals["Strategy_Return"],
        label="Model Strategy",
        alpha=0.8
    )

    plt.axhline(
        y=0,
        color="black",
        linewidth=1
    )

    plt.title("Weekly Returns: Model Strategy vs. Buy and Hold")
    plt.xlabel("Date")
    plt.ylabel("Weekly return")
    plt.legend()
    plt.tight_layout()

    plt.savefig(RETURNS_PLOT_FILE)
    plt.close()

    return RETURNS_PLOT_FILE


# 5. Load trading signals

if not SIGNALS_FILE.exists():
    raise FileNotFoundError(
        f"Trading signal file not found: {SIGNALS_FILE}"
    )

signals = pd.read_csv(SIGNALS_FILE)

signals["Date"] = pd.to_datetime(signals["Date"])

required_columns = [
    "Date",
    "Trading_Signal",
    "Strategy_Return",
    "Buy_And_Hold_Return",
    "Cash_Return",
    "Strategy_Equity",
    "Buy_And_Hold_Equity",
    "Cash_Equity",
    "Position_Change"
]

for column in required_columns:
    if column not in signals.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# 6. Scale equity curves by initial capital

signals["Strategy_Equity"] = (
    signals["Strategy_Equity"] * INITIAL_CAPITAL
)

signals["Buy_And_Hold_Equity"] = (
    signals["Buy_And_Hold_Equity"] * INITIAL_CAPITAL
)

signals["Cash_Equity"] = (
    signals["Cash_Equity"] * INITIAL_CAPITAL
)


# 7. Calculate metrics

summary_rows = [
    calculate_strategy_metrics(
        strategy_name="Model Strategy",
        return_column="Strategy_Return",
        equity_column="Strategy_Equity",
        signals=signals
    ),
    calculate_strategy_metrics(
        strategy_name="Buy and Hold",
        return_column="Buy_And_Hold_Return",
        equity_column="Buy_And_Hold_Equity",
        signals=signals
    ),
    calculate_strategy_metrics(
        strategy_name="Cash",
        return_column="Cash_Return",
        equity_column="Cash_Equity",
        signals=signals
    )
]

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# 8. Save plots

save_equity_curve_plot(signals)

save_weekly_returns_plot(signals)


# 9. Create report

model_result = summary[
    summary["Strategy"] == "Model Strategy"
].iloc[0]

buy_hold_result = summary[
    summary["Strategy"] == "Buy and Hold"
].iloc[0]

report_lines = [
    "# 07 Backtesting Report",
    "",
    "## Goal",
    "",
    (
        "Evaluate whether the final model predictions from the "
        "test period can be converted into useful trading signals."
    ),
    "",
    "## Trading Rule",
    "",
    "- Predicted target `1`: hold BTC for the next week.",
    "- Predicted target `0`: stay in cash for the next week.",
    (
        f"- Transaction cost per position change: "
        f"{TRANSACTION_COST:.2%}"
    ),
    "",
    "## Test Period",
    "",
    (
        f"- Start date: `{signals['Date'].min().date()}`"
    ),
    (
        f"- End date: `{signals['Date'].max().date()}`"
    ),
    f"- Number of weeks: `{len(signals)}`",
    "",
    "## Main Result",
    "",
    (
        f"- Model strategy total return: "
        f"{model_result['Total_Return']:.2%}"
    ),
    (
        f"- Buy-and-Hold total return: "
        f"{buy_hold_result['Total_Return']:.2%}"
    ),
    (
        f"- Model strategy max drawdown: "
        f"{model_result['Max_Drawdown']:.2%}"
    ),
    (
        f"- Buy-and-Hold max drawdown: "
        f"{buy_hold_result['Max_Drawdown']:.2%}"
    ),
    "",
    "## Interpretation",
    "",
    (
        "The backtest translates classification predictions into "
        "a simple trading strategy. It does not improve or retrain "
        "the model. It only checks whether the saved test predictions "
        "would have produced useful trading performance."
    ),
    "",
    "## Output Files",
    "",
    f"- `{SUMMARY_FILE.name}`",
    f"- `{EQUITY_PLOT_FILE.name}`",
    f"- `{RETURNS_PLOT_FILE.name}`"
]

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# 10. Print short console summary

print("Backtest evaluation finished.")
print(f"Saved summary to: {SUMMARY_FILE}")
print(f"Saved equity plot to: {EQUITY_PLOT_FILE}")
print(f"Saved returns plot to: {RETURNS_PLOT_FILE}")
print(f"Saved report to: {REPORT_FILE}")
print("\nSummary:")
print(summary)
