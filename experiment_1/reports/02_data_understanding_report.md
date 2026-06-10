# Data Understanding Report

## Overview by symbol

```text
dataset  symbol  ticker  rows start_date   end_date  missing_values  duplicate_dates  large_date_gaps  close_min  close_mean  close_max  volume_mean
    BTC BTC_USD BTC-USD   262 2020-01-03 2025-01-03               0                0                0    5563.71    36472.53  101459.26 2.313645e+11
 MARKET     GLD     GLD   262 2020-01-03 2025-01-03               0                0                0     140.11      181.13     253.32 4.138891e+07
 MARKET     QQQ     QQQ   262 2020-01-03 2025-01-03               0                0                0     170.70      345.43     530.53 2.380664e+08
 MARKET     SPY     SPY   262 2020-01-03 2025-01-03               0                0                0     228.80      425.56     607.81 3.920702e+08
  MACRO   US10Y    ^TNX   262 2020-01-03 2025-01-03               0                0                0       0.54        2.69       4.92 0.000000e+00
  MACRO   US20Y     TLT   262 2020-01-03 2025-01-03               0                0                0      83.24      122.84     171.00 1.137476e+08
  MACRO   US30Y    ^TYX   262 2020-01-03 2025-01-03               0                0                0       1.18        3.05       5.09 0.000000e+00
  MACRO     VIX    ^VIX   262 2020-01-03 2025-01-03               0                0                0      11.93       21.06      66.04 0.000000e+00
```

## BTC

- Rows: 262
- Columns: Date, Open, High, Low, Close, Adj Close, Volume, VWAP, Symbol, Ticker
- Date range: 2020-01-03 to 2025-01-03
- Symbols: BTC_USD
- Total missing values: 0
- Duplicate Date/Symbol rows: 0

### Descriptive statistics

```text
            Open       High       Low      Close        Volume       VWAP
count     262.00     262.00    262.00     262.00  2.620000e+02     262.00
mean    36144.41   38487.64  33936.13   36472.53  2.313645e+11   36314.32
std     21167.52   22414.04  19905.13   21384.25  1.128657e+11   21212.27
min      5573.08    6844.26   4106.98    5563.71  6.747923e+10    5538.80
25%     19604.00   21005.99  18665.76   19806.83  1.652682e+11   19826.81
50%     31495.99   35535.14  29519.09   32321.77  2.099274e+11   32846.69
75%     50554.61   55418.41  46819.60   50799.63  2.682699e+11   50172.24
max    101451.44  108268.45  94321.26  101459.26  7.873680e+11  101571.77
```

### Sample rows

```text
      Date        Open        High         Low       Close   Adj Close       Volume        VWAP  Symbol  Ticker
2020-01-03 7194.892090 7413.715332 6914.996094 7344.884277 7344.884277  67479229494 7164.921956 BTC_USD BTC-USD
2020-01-10 7345.375488 8396.738281 7309.514160 8166.554199 8166.554199 174646031870 7853.243838 BTC_USD BTC-USD
2020-01-17 8162.190918 8958.122070 8009.059082 8929.038086 8929.038086 223538254162 8544.084995 BTC_USD BTC-USD
2020-01-24 8927.211914 9164.362305 8266.840820 8445.434570 8445.434570 189843685144 8686.221602 BTC_USD BTC-USD
2020-01-31 8440.119141 9553.125977 8296.218750 9350.529297 9350.529297 197364973826 9070.836197 BTC_USD BTC-USD
```

## MARKET

- Rows: 786
- Columns: Date, Open, High, Low, Close, Adj Close, Volume, VWAP, Symbol, Ticker
- Date range: 2020-01-03 to 2025-01-03
- Symbols: GLD, QQQ, SPY
- Total missing values: 0
- Duplicate Date/Symbol rows: 0

### Descriptive statistics

```text
         Open    High     Low   Close        Volume    VWAP
count  786.00  786.00  786.00  786.00  7.860000e+02  786.00
mean   316.42  322.22  311.14  317.37  2.238418e+08  316.85
std    120.46  122.05  118.80  120.81  1.887508e+08  120.46
min    137.56  146.20  136.12  140.11  6.044600e+06  140.70
25%    184.72  187.42  181.53  184.87  4.771485e+07  184.53
50%    320.18  325.76  312.68  321.32  2.113479e+08  319.13
75%    412.33  417.84  405.90  412.43  3.305508e+08  412.32
max    607.69  609.07  602.34  607.81  1.562964e+09  605.89
```

### Sample rows

```text
      Date       Open       High        Low      Close  Adj Close   Volume       VWAP Symbol Ticker
2020-01-03 143.860001 146.320007 143.399994 145.860001 145.860001 20006600 145.084298    GLD    GLD
2020-01-10 148.440002 148.610001 145.440002 146.910004 146.910004 61105900 147.121038    GLD    GLD
2020-01-17 146.350006 146.990005 145.080002 146.580002 146.580002 37883700 146.191566    GLD    GLD
2020-01-24 145.770004 148.380005 145.550003 147.979996 147.979996 29204000 147.105211    GLD    GLD
2020-01-31 149.240005 149.679993 147.529999 149.330002 149.330002 46452100 148.730177    GLD    GLD
```

## MACRO

- Rows: 1048
- Columns: Date, Open, High, Low, Close, Adj Close, Volume, VWAP, Symbol, Ticker
- Date range: 2020-01-03 to 2025-01-03
- Symbols: US10Y, US20Y, US30Y, VIX
- Total missing values: 0
- Duplicate Date/Symbol rows: 0

### Descriptive statistics

```text
          Open     High      Low    Close        Volume     VWAP
count  1048.00  1048.00  1048.00  1048.00  1.048000e+03  1048.00
mean     37.62    38.87    36.39    37.41  2.843690e+07    37.55
std      51.87    52.51    51.16    51.89  5.914364e+07    51.82
min       0.48     0.57     0.40     0.54  0.000000e+00     0.55
25%       3.05     3.19     2.96     3.09  0.000000e+00     3.06
50%       8.71     9.15     7.79     8.51  0.000000e+00     8.66
75%      76.26    85.38    63.67    70.34  5.850025e+06    73.26
max     179.10   179.70   168.98   171.00  3.086069e+08   170.58
```

### Sample rows

```text
      Date  Open  High   Low  Close  Adj Close  Volume     VWAP Symbol Ticker
2020-01-03 1.903 1.903 1.786  1.788      1.788       0 1.825667  US10Y   ^TNX
2020-01-10 1.785 1.900 1.766  1.825      1.825       0 1.830333  US10Y   ^TNX
2020-01-17 1.834 1.858 1.780  1.836      1.836       0 1.824667  US10Y   ^TNX
2020-01-24 1.797 1.797 1.670  1.681      1.681       0 1.716000  US10Y   ^TNX
2020-01-31 1.612 1.655 1.512  1.520      1.520       0 1.562333  US10Y   ^TNX
```
