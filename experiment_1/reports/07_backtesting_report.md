# 07 Backtesting Report

## Goal

Evaluate whether the final model predictions from the test period can be converted into useful trading signals.

## Trading Rule

- Predicted target `1`: hold BTC for the next week.
- Predicted target `0`: stay in cash for the next week.
- Transaction cost per position change: 0.00%

## Test Period

- Start date: `2024-01-26`
- End date: `2024-12-27`
- Number of weeks: `49`

## Main Result

- Model strategy total return: 16.48%
- Buy-and-Hold total return: 123.42%
- Model strategy max drawdown: -10.53%
- Buy-and-Hold max drawdown: -22.81%

## Interpretation

The backtest translates classification predictions into a simple trading strategy. It does not improve or retrain the model. It only checks whether the saved test predictions would have produced useful trading performance.

## Output Files

- `07_backtest_summary.csv`
- `07_backtest_equity_curve.png`
- `07_backtest_weekly_returns.png`