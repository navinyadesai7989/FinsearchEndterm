# Nifty50 Black-Scholes Accuracy Backtest

Backtesting how accurately the **Black-Scholes-Merton model** prices real
**Nifty50 index options**, plus a Monte Carlo cross-validation of the
pricing engine.

## Structure

```
.
├── data/
│   ├── raw/                          # original NSE downloads (CE + PE, separate files)
│   └── nifty50_options_real.csv      # cleaned dataset used for the backtest
├── src/
│   ├── black_scholes.py              # BS pricing engine, Greeks, implied vol solver
│   ├── monte_carlo.py                # GBM Monte Carlo pricer + convergence study
│   ├── prepare_real_data.py          # cleans the raw NSE CE/PE CSVs into the backtest schema
│   ├── accuracy_analysis.py          # main backtest: BS price vs market price
│   └── test_black_scholes.py         # unit tests (put-call parity, MC vs BS, etc.)
├── outputs/                          # plots + result CSVs from the backtest
├── requirements.txt
└── README.md
```

## The data

`data/nifty50_options_real.csv` is real Nifty50 option chain data, downloaded
from NSE India's historical data portal:

> nseindia.com → Reports → Historical Data → F&O → Option Chain
> Instrument = Index Options, Symbol = NIFTY, past 3 months

Raw files (`data/raw/`) cover **28 April 2026 to 27 July 2026**, split into
separate CE and PE downloads (~110k rows combined). `src/prepare_real_data.py`
cleans this into the schema `accuracy_analysis.py` expects:
- drops rows with no actual trade that day (NSE marks these with `-` in the
  contracts-traded column — a stale settlement mark, not a real quote)
- drops NSE's multi-year "long dated" option series (expiries out to 2031)
  since they're a different, barely-traded product — capped at 90 days to expiry
- computes T (time to expiry, years) and attaches a representative risk-free
  rate and dividend yield (not in NSE's bhavcopy)

After cleaning: **56,719 real option quotes across 62 trading days and 19
expiries**.

## How to run

```bash
pip install -r requirements.txt

python src/prepare_real_data.py                                   # rebuild the cleaned dataset (optional, already included)
python src/accuracy_analysis.py --data data/nifty50_options_real.csv   # run the backtest
python src/monte_carlo.py                                          # Monte Carlo cross-validation
python -m pytest src/test_black_scholes.py -v                      # unit tests
```

## Method

1. Estimate a single day-level implied volatility from the ATM options
   traded that day.
2. Price the entire option chain that day with Black-Scholes using that
   one volatility.
3. Compare BS price to the actual market price for every quote — MAE,
   RMSE, MAPE, mean bias, R² — overall and by moneyness/maturity bucket.
4. Cross-validate the Black-Scholes implementation with an independently
   built Monte Carlo pricer (`outputs/mc_convergence.png`).

## Key finding

Black-Scholes explains 99.6%+ of the variation in real market prices
(R² = 0.9965) with negligible bias, but the average percentage error is
~35% — driven almost entirely by deep out-of-the-money and longer-dated
contracts, where a single flat volatility stops matching the real
volatility surface (skew + term structure). Near the money, error stays
in the 6-19% MAPE range across maturities. See
`outputs/error_vs_moneyness.png`.
