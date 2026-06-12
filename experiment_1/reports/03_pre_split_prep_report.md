# 03 Pre-Split Data Preparation Report

## Goal

This step prepares one modelling table before the chronological split.
All features use only information from the current week or earlier weeks.
The target looks one week into the future and is not used as a feature.

## Input and Output Files

Input files:

- `data/raw/BTC_USD.csv`
- `data/raw/STOCKS.csv`
- `data/raw/INTEREST.csv`
- `data/raw/SENTIMENT.csv`

Output files:

- `data/processed/03_features_targets.csv`
- `data/processed/03_feature_target_sample.csv`
- `data/processed/03_feature_target_summary.csv`
- `experiment_1/reports/03_pre_split_prep_report.md`
- `experiment_1/images/03_feature_overview.png`
- `experiment_1/images/03_target_distribution.png`

## Parameters

- Target horizon: 1 week
- BTC return windows: [1, 2, 4] weeks
- Rolling windows: [4, 8, 12] weeks
- RSI window: 14 weeks
- MACD windows: fast=12, slow=26, signal=9

## Process

1. Load weekly raw CSV files.
2. Keep BTC as the main row table with one row per week.
3. Pivot market and macro data so each symbol becomes a feature column.
4. Merge all datasets by `Date`.
5. Create BTC return, volatility, trend, volume, VWAP, RSI, and MACD features.
6. Create market return features from QQQ, SPY, and GLD.
7. Create macro level and weekly change features from VIX and bond indicators.
8. Add Fear & Greed sentiment level and weekly change.
9. Create the binary target for next week's BTC direction.
10. Drop rows with missing values from rolling windows and the final unknown target.

## Target Definition

`target_next_week_up` is `1` if next week's BTC close is higher than the current BTC close.
Otherwise it is `0`.

Formula:

```text
target_next_week_return = BTC_Close[t + 1] / BTC_Close[t] - 1
target_next_week_up = 1 if target_next_week_return > 0 else 0
```

## Feature Selection Intuition

- BTC returns describe short-term momentum.
- Rolling volatility describes market risk.
- SMA ratios, RSI, and MACD describe trend and overbought or oversold situations.
- Volume change and VWAP distance describe trading activity and price pressure.
- QQQ, SPY, and GLD returns describe related market movements.
- VIX and bond indicators describe macro risk and interest-rate pressure.
- Fear & Greed describes crypto market sentiment.

## Dataset Overview

- Rows after cleaning: 247
- Feature columns: 28
- Start date: 2020-04-10
- End date: 2024-12-27
- Target up rate: 51.42%

## Sample of Designed Features and Target

```text
      Date  btc_close  btc_return_1w  btc_return_4w  btc_volatility_4w  btc_sma_ratio_4w  btc_rsi_14w  market_QQQ_return_1w  macro_VIX_close  fear_greed_value  target_next_week_return  target_next_week_up
2020-04-10  6865.4932         0.0196         0.2340             0.0411            0.0455      47.7979                0.0954            41.67           16.2857                   0.0336                    1
2020-04-17  7096.1846         0.0336         0.1448             0.0107            0.0449      44.8009                0.0718            38.15           13.8571                   0.0641                    1
2020-04-24  7550.9009         0.0641         0.1671             0.0186            0.0693      43.0997               -0.0067            35.93           17.7143                   0.1740                    1
2020-05-01  8864.7666         0.1740         0.3165             0.0700            0.1673      51.9384               -0.0051            37.19           29.8571                   0.1103                    1
2020-05-08  9842.6660         0.1103         0.4336             0.0611            0.1804      52.2598                0.0570            27.98           45.0000                  -0.0523                    0
2020-05-15  9328.1973        -0.0523         0.3145             0.0955            0.0485      47.8658               -0.0071            31.89           44.0000                  -0.0156                    0
2020-05-22  9182.5771        -0.0156         0.2161             0.1060           -0.0131      44.6658                0.0286            28.16           46.2857                   0.0279                    1
2020-05-29  9439.1240         0.0279         0.0648             0.0700           -0.0010      48.7899                0.0161            27.51           41.5714                   0.0240                    1
2020-06-05  9665.5332         0.0240        -0.0180             0.0377            0.0278      55.2650                0.0271            24.52           51.4286                  -0.0191                    0
2020-06-12  9480.8438        -0.0191         0.0164             0.0251            0.0041      51.9546               -0.0159            36.09           51.0000                  -0.0203                    0
```

## Descriptive Statistics

```text
                 column  missing_values      mean       std        min    median        max
          btc_return_1w               0    0.0146    0.0888    -0.2961    0.0041     0.3889
          btc_return_2w               0    0.0302    0.1330    -0.3496    0.0114     0.6541
          btc_return_4w               0    0.0637    0.2032    -0.3819    0.0215     1.2591
          btc_range_pct               0    0.1216    0.0692     0.0252    0.1047     0.5350
   btc_volume_change_1w               0    0.0283    0.2389    -0.5492   -0.0087     0.8576
      btc_vwap_distance               0    0.0051    0.0453    -0.1260    0.0014     0.1673
       btc_sma_ratio_4w               0    0.0174    0.0855    -0.2624    0.0081     0.3833
      btc_volatility_4w               0    0.0751    0.0435     0.0056    0.0700     0.2395
       btc_sma_ratio_8w               0    0.0421    0.1455    -0.3205    0.0215     0.7136
      btc_volatility_8w               0    0.0821    0.0337     0.0208    0.0741     0.1937
      btc_sma_ratio_12w               0    0.0649    0.1963    -0.4007    0.0363     0.9675
     btc_volatility_12w               0    0.0844    0.0295     0.0346    0.0754     0.1725
            btc_rsi_14w               0   56.3834   18.9109    11.5573   57.6981    93.7625
               btc_macd               0 1746.3840 4027.0726 -6214.7219 1347.4167 11210.8983
        btc_macd_signal               0 1624.4006 3658.8091 -5378.2686 1429.8574 10090.5580
   market_GLD_return_1w               0    0.0021    0.0198    -0.0615    0.0019     0.0569
   market_QQQ_return_1w               0    0.0047    0.0302    -0.0745    0.0049     0.0954
   market_SPY_return_1w               0    0.0038    0.0241    -0.0614    0.0057     0.1209
      macro_US10Y_close               0    2.7627    1.3911     0.5360    3.1560     4.9240
      macro_US20Y_close               0  121.3955   27.0641    83.2400  112.5000   171.0000
      macro_US30Y_close               0    3.1094    1.1667     1.1770    3.2690     5.0890
        macro_VIX_close               0   20.4972    6.1260    11.9300   19.3200    41.6700
  macro_US10Y_change_1w               0    0.0163    0.1290    -0.4080    0.0180     0.3440
  macro_US20Y_change_1w               0   -0.3296    2.5034    -7.4800   -0.2400     6.1900
  macro_US30Y_change_1w               0    0.0146    0.1175    -0.3470    0.0060     0.3270
    macro_VIX_change_1w               0   -0.1249    3.2431   -13.1600   -0.2200    11.5700
       fear_greed_value               0   50.8355   22.0709     9.0000   51.0000    93.4286
   fear_greed_change_1w               0    0.2504    9.0624   -35.7143    0.2857    36.7143
target_next_week_return               0    0.0144    0.0888    -0.2961    0.0032     0.3889
    target_next_week_up               0    0.5142    0.5008     0.0000    1.0000     1.0000
```

## Findings

1. The final dataset is smaller than the raw dataset because rolling features need past weeks.
2. The target is not perfectly balanced, so baseline models are important in the modelling step.
3. BTC return and volatility features are useful because BTC has strong weekly movements.
4. Market, macro, and sentiment features add context beyond BTC price history.
5. All scaling must happen after the chronological split to avoid data leakage.