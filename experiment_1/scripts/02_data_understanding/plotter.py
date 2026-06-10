"""
Data Understanding: plots for raw weekly data

This script uses pandas and Pillow. It avoids complex plotting code so that the
steps are easy to explain in the presentation.

Input files:
- data/raw/BTC_USD.csv
- data/raw/STOCKS.csv
- data/raw/INTEREST.csv

Output files:
- experiment_1/images/02_raw_data_overview.png
- experiment_1/images/02_performance_comparison.png
- experiment_1/images/02_correlation_matrix.png
"""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CURRENT_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = CURRENT_FILE.parents[2]
PROJECT_DIR = CURRENT_FILE.parents[3]

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
IMAGE_DIR = EXPERIMENT_DIR / "images"

BTC_FILE = RAW_DATA_DIR / "BTC_USD.csv"
STOCKS_FILE = RAW_DATA_DIR / "STOCKS.csv"
INTEREST_FILE = RAW_DATA_DIR / "INTEREST.csv"

COLORS = [
    (30, 30, 30),
    (31, 119, 180),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (255, 127, 14),
    (140, 86, 75),
    (23, 190, 207),
]


def load_csv(file_path):
    """Load one CSV file and make sure Date is a datetime column."""
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Symbol", "Date"])
    return data


def close_prices_by_symbol(data):
    """Create a table with Date as rows, Symbol as columns, and Close as values."""
    close_prices = data.pivot(index="Date", columns="Symbol", values="Close")
    close_prices = close_prices.sort_index()
    return close_prices


def scale_value(value, min_value, max_value, top, bottom):
    """Scale a numeric value to a y position in the chart."""
    if max_value == min_value:
        return (top + bottom) / 2

    return bottom - (value - min_value) / (max_value - min_value) * (bottom - top)


def draw_line_panel(draw, box, title, series_dict, font):
    """Draw one line chart panel."""
    left, top, right, bottom = box
    chart_left = left + 70
    chart_top = top + 35
    chart_right = right - 25
    chart_bottom = bottom - 45

    all_values = []
    all_dates = []

    for series in series_dict.values():
        clean_series = series.dropna()
        all_values.extend(clean_series.tolist())
        all_dates.extend(clean_series.index.tolist())

    if len(all_values) == 0:
        draw.text((left + 10, top + 10), title + " (no data)", fill=(0, 0, 0), font=font)
        return

    min_value = min(all_values)
    max_value = max(all_values)
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = max((max_date - min_date).days, 1)

    draw.text((left + 10, top + 8), title, fill=(0, 0, 0), font=font)
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline=(180, 180, 180))

    for grid_step in range(1, 4):
        y = chart_top + grid_step * (chart_bottom - chart_top) / 4
        draw.line((chart_left, y, chart_right, y), fill=(230, 230, 230))

    draw.text((left + 10, chart_top), f"{max_value:.2f}", fill=(80, 80, 80), font=font)
    draw.text((left + 10, chart_bottom - 10), f"{min_value:.2f}", fill=(80, 80, 80), font=font)
    draw.text((chart_left, chart_bottom + 12), str(min_date.date()), fill=(80, 80, 80), font=font)
    draw.text((chart_right - 80, chart_bottom + 12), str(max_date.date()), fill=(80, 80, 80), font=font)

    for color_index, (label, series) in enumerate(series_dict.items()):
        clean_series = series.dropna()
        points = []

        for date, value in clean_series.items():
            x = chart_left + (date - min_date).days / date_range * (chart_right - chart_left)
            y = scale_value(value, min_value, max_value, chart_top, chart_bottom)
            points.append((x, y))

        if len(points) > 1:
            color = COLORS[color_index % len(COLORS)]
            draw.line(points, fill=color, width=2)
            legend_x = chart_left + 120 * (color_index % 4)
            legend_y = bottom - 25 + 15 * (color_index // 4)
            draw.rectangle((legend_x, legend_y + 3, legend_x + 10, legend_y + 13), fill=color)
            draw.text((legend_x + 15, legend_y), label, fill=(0, 0, 0), font=font)


def save_raw_data_overview(btc_data, stocks_data, interest_data):
    """Save one overview image with BTC, market, and macro data."""
    btc = btc_data[btc_data["Symbol"] == "BTC_USD"].set_index("Date")
    stocks = close_prices_by_symbol(stocks_data)
    interest = close_prices_by_symbol(interest_data)

    image = Image.new("RGB", (1200, 1400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw_line_panel(
        draw,
        (20, 20, 1180, 350),
        "BTC/USD weekly close price",
        {"BTC_USD": btc["Close"]},
        font,
    )

    draw_line_panel(
        draw,
        (20, 365, 1180, 695),
        "BTC/USD weekly volume",
        {"BTC volume": btc["Volume"]},
        font,
    )

    draw_line_panel(
        draw,
        (20, 710, 1180, 1040),
        "Market indicators: QQQ, SPY, GLD",
        {column: stocks[column] for column in stocks.columns},
        font,
    )

    draw_line_panel(
        draw,
        (20, 1055, 1180, 1385),
        "Macro indicators: VIX and US bonds",
        {column: interest[column] for column in interest.columns},
        font,
    )

    save_path = IMAGE_DIR / "02_raw_data_overview.png"
    image.save(save_path)
    print(f"Saved: {save_path}")


def save_performance_comparison(btc_data, stocks_data):
    """Save a normalized performance comparison image."""
    btc_close = close_prices_by_symbol(btc_data)
    stocks_close = close_prices_by_symbol(stocks_data)

    prices = pd.concat([btc_close, stocks_close], axis=1)
    prices = prices[["BTC_USD", "QQQ", "SPY", "GLD"]]
    prices = prices.dropna()

    normalized = prices / prices.iloc[0] * 100

    image = Image.new("RGB", (1200, 650), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    series_dict = {column: normalized[column] for column in normalized.columns}
    draw_line_panel(
        draw,
        (20, 20, 1180, 630),
        "Performance comparison, start = 100",
        series_dict,
        font,
    )

    save_path = IMAGE_DIR / "02_performance_comparison.png"
    image.save(save_path)
    print(f"Saved: {save_path}")


def color_for_correlation(value):
    """Return a simple blue-white-red color for a correlation value."""
    value = max(min(value, 1), -1)

    if value >= 0:
        red = 255
        green = int(255 * (1 - value))
        blue = int(255 * (1 - value))
    else:
        red = int(255 * (1 + value))
        green = int(255 * (1 + value))
        blue = 255

    return red, green, blue


def save_correlation_matrix(btc_data, stocks_data, interest_data):
    """Save a heatmap of weekly return correlations."""
    btc_close = close_prices_by_symbol(btc_data)
    stocks_close = close_prices_by_symbol(stocks_data)
    interest_close = close_prices_by_symbol(interest_data)

    prices = pd.concat([btc_close, stocks_close, interest_close], axis=1)
    prices = prices.dropna()

    returns = prices.pct_change().dropna()
    corr = returns.corr()

    labels = list(corr.columns)
    cell_size = 75
    left_margin = 150
    top_margin = 100
    width = left_margin + cell_size * len(labels) + 40
    height = top_margin + cell_size * len(labels) + 40

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((20, 20), "Correlation matrix of weekly returns", fill=(0, 0, 0), font=font)

    for index, label in enumerate(labels):
        x = left_margin + index * cell_size + 5
        y = top_margin - 25
        draw.text((x, y), label, fill=(0, 0, 0), font=font)

        x = 20
        y = top_margin + index * cell_size + 25
        draw.text((x, y), label, fill=(0, 0, 0), font=font)

    for row_index, row_label in enumerate(labels):
        for col_index, col_label in enumerate(labels):
            value = corr.loc[row_label, col_label]
            x1 = left_margin + col_index * cell_size
            y1 = top_margin + row_index * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            draw.rectangle((x1, y1, x2, y2), fill=color_for_correlation(value), outline=(180, 180, 180))
            draw.text((x1 + 22, y1 + 28), f"{value:.2f}", fill=(0, 0, 0), font=font)

    save_path = IMAGE_DIR / "02_correlation_matrix.png"
    image.save(save_path)
    print(f"Saved: {save_path}")


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    btc_data = load_csv(BTC_FILE)
    stocks_data = load_csv(STOCKS_FILE)
    interest_data = load_csv(INTEREST_FILE)

    save_raw_data_overview(btc_data, stocks_data, interest_data)
    save_performance_comparison(btc_data, stocks_data)
    save_correlation_matrix(btc_data, stocks_data, interest_data)

    print("Data Understanding plots finished.")


if __name__ == "__main__":
    main()
