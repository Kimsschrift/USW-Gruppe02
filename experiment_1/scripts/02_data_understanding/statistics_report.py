"""
Data Understanding: statistics report

Input files:
- data/raw/BTC_USD.csv
- data/raw/STOCKS.csv
- data/raw/INTEREST.csv

Output files:
- experiment_1/reports/02_data_understanding_report.md
- data/processed/02_data_understanding_summary.csv
"""

from pathlib import Path

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = CURRENT_FILE.parents[2]
PROJECT_DIR = CURRENT_FILE.parents[3]

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = EXPERIMENT_DIR / "reports"

INPUT_FILES = {
    "BTC": RAW_DATA_DIR / "BTC_USD.csv",
    "MARKET": RAW_DATA_DIR / "STOCKS.csv",
    "MACRO": RAW_DATA_DIR / "INTEREST.csv",
}

REPORT_PATH = REPORT_DIR / "02_data_understanding_report.md"
SUMMARY_PATH = PROCESSED_DATA_DIR / "02_data_understanding_summary.csv"


def load_csv(file_path):
    """Load one CSV file and make sure Date is a datetime column."""
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Symbol", "Date"])
    return data


def build_overview(name, data):
    """Create one overview row for each symbol in one data file."""
    rows = []

    for symbol, symbol_data in data.groupby("Symbol"):
        date_diff = symbol_data["Date"].diff().dt.days
        large_gaps = date_diff[date_diff > 10].count()

        rows.append(
            {
                "dataset": name,
                "symbol": symbol,
                "ticker": symbol_data["Ticker"].iloc[0],
                "rows": len(symbol_data),
                "start_date": symbol_data["Date"].min().date(),
                "end_date": symbol_data["Date"].max().date(),
                "missing_values": int(symbol_data.isna().sum().sum()),
                "duplicate_dates": int(symbol_data.duplicated(["Date", "Symbol"]).sum()),
                "large_date_gaps": int(large_gaps),
                "close_min": symbol_data["Close"].min(),
                "close_mean": symbol_data["Close"].mean(),
                "close_max": symbol_data["Close"].max(),
                "volume_mean": symbol_data["Volume"].mean(),
            }
        )

    return rows


def add_dataset_section(report_lines, name, data):
    """Add readable statistics for one dataset to the markdown report."""
    report_lines.append(f"## {name}")
    report_lines.append("")
    report_lines.append(f"- Rows: {len(data)}")
    report_lines.append(f"- Columns: {', '.join(data.columns)}")
    report_lines.append(f"- Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    report_lines.append(f"- Symbols: {', '.join(sorted(data['Symbol'].unique()))}")
    report_lines.append(f"- Total missing values: {int(data.isna().sum().sum())}")
    report_lines.append(f"- Duplicate Date/Symbol rows: {int(data.duplicated(['Date', 'Symbol']).sum())}")
    report_lines.append("")

    important_columns = ["Open", "High", "Low", "Close", "Volume", "VWAP"]
    stats = data[important_columns].describe().round(2)

    report_lines.append("### Descriptive statistics")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append(stats.to_string())
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("### Sample rows")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append(data.head(5).to_string(index=False))
    report_lines.append("```")
    report_lines.append("")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = ["# Data Understanding Report", ""]
    overview_rows = []

    for name, file_path in INPUT_FILES.items():
        print(f"Reading {file_path}")
        data = load_csv(file_path)

        overview_rows.extend(build_overview(name, data))
        add_dataset_section(report_lines, name, data)

    overview = pd.DataFrame(overview_rows)
    overview.to_csv(SUMMARY_PATH, index=False)

    report_lines.insert(2, "## Overview by symbol")
    report_lines.insert(3, "")
    report_lines.insert(4, "```text\n" + overview.round(2).to_string(index=False) + "\n```")
    report_lines.insert(5, "")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Data Understanding statistics finished.")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
