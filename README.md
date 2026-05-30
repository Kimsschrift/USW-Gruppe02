# USW-Gruppe02

# BTC/USD Algorithmischer Trading Bot
HTW Berlin | Unternehmenssoftware | Gruppenprojekt

---

## Projektstruktur
- 4 benotete Präsentationen (jeweils 20%)
- Bewertungskriterien pro Präsentation:
  - DS-Schritt-Kriterien (40%)
  - Code-Verständnis / Pseudocode / Flussdiagramme (30%)
  - Präsentationsform (30%)

---

## Präsentation 1: Problemdefinition & Datenbeschaffung

### Problemdefinition

**Problembeschreibung:**
- Basierend auf wöchentlichen BTC/USD-Kursdaten und makroökonomischen Indikatoren soll vorhergesagt werden, ob der BTC-Preis in der nächsten Woche steigt oder fällt. Ziel ist es zu überprüfen, ob die Vorhersagegenauigkeit eines ML-Modells in reale Trading-Gewinne umgewandelt werden kann.

**Zielvariable:**
- Binär: Schlusskurs (nächste Woche) > Schlusskurs (diese Woche) → 1 (Aufwärts) / 0 (Abwärts)

**Eingabevariablen:**

| Typ | Daten |
|-----|-------|
| Roh | BTC/USD OHLCV, VWAP |
| Markt | S&P 500 (SPY), QQQ, GLD |
| Makro | VIX, US-Anleihen (10J, 20J, 30J) |
| Abgeleitet | SMA, EMA, RSI, MACD, Volatilität |

---

### Datenbeschaffung

**Vorgehensweise:**
Alle Rohdaten werden über yfinance bezogen und als CSV-Dateien im Verzeichnis `data/raw/` gespeichert.

**Verwendete APIs:**

| API | Daten | Datei |
|-----|-------|-------|
| yfinance | BTC/USD wöchentliche OHLCV | BTC_USD.csv |
| yfinance | S&P 500 (SPY), QQQ, GLD | STOCKS.csv |
| yfinance | VIX (^VIX), US-Anleihen (^TNX, TLT, ^TYX) | INTEREST.csv |

**Parameter:**
- START_DATE: 2020-01-01
- END_DATE: 2025-01-01
- Intervall: Täglich (1d), aggregiert auf wöchentlich (W-FRI)

**Speicherung:**
- Format: CSV
- Speicherort: `data/raw/`

---
