# turns the raw NSE historical option chain downloads (CE and PE, separate
# files, that's just how their download tool splits it) into the
# date/spot/strike/expiry/T/option_type/r/q/market_price/volume/oi format
# accuracy_analysis.py expects.
#
# raw files: data/raw/OPTIDX_NIFTY_CE_28-Apr-2026_TO_28-Jul-2026.csv
#            data/raw/OPTIDX_NIFTY_PE_28-Apr-2026_TO_28-Jul-2026.csv
# (downloaded from nseindia.com -> Reports -> Historical Data -> F&O ->
#  Option Chain, Instrument = Index Options, Symbol = NIFTY, past 3M)

import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
CE_FILE = f"{RAW_DIR}/OPTIDX_NIFTY_CE_28-Apr-2026_TO_28-Jul-2026.csv"
PE_FILE = f"{RAW_DIR}/OPTIDX_NIFTY_PE_28-Apr-2026_TO_28-Jul-2026.csv"
OUT_FILE = "data/nifty50_options_real.csv"

# rough 91-day T-bill yield and Nifty dividend yield for this window --
# NSE's bhavcopy doesn't include these, so using representative constants
# rather than day-by-day values (same simplification as the sample dataset)
RISK_FREE_RATE = 0.065
DIVIDEND_YIELD = 0.013


def load_one(path, option_type):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%b-%Y")
    df["expiry"] = pd.to_datetime(df["Expiry"].str.strip(), format="%d-%b-%Y")
    df["No. of contracts"] = df["No. of contracts"].astype(str).str.strip()
    df["Open Int"] = df["Open Int"].astype(str).str.strip()

    # rows with '-' for contracts traded = no actual trade that day, just
    # a carried/theoretical settlement value -- drop these, we want real
    # traded prices for the backtest, not stale marks
    df = df[df["No. of contracts"] != "-"].copy()
    df["volume"] = pd.to_numeric(df["No. of contracts"], errors="coerce")
    df["oi"] = pd.to_numeric(df["Open Int"], errors="coerce").fillna(0)

    df["spot"] = df["Underlying Value"]
    df["strike"] = df["Strike Price"]
    df["market_price"] = df["Close"]
    df["option_type"] = option_type

    df["T"] = (df["expiry"] - df["date"]).dt.days / 365
    df = df[df["T"] > 0]  # drop same-day-expiry rows, T=0 breaks BS

    # NSE also lists a handful of multi-year "long dated" Nifty option
    # series (expiries out to 2031!) alongside the regular weekly/monthly
    # ones. They're barely traded and behave completely differently, so
    # they don't belong in the same backtest as near-term options -- cap
    # at 90 days, which covers weeklies + the current month/quarter.
    df = df[df["T"] <= 90 / 365]

    df["r"] = RISK_FREE_RATE
    df["q"] = DIVIDEND_YIELD

    cols = ["date", "spot", "strike", "expiry", "T", "option_type",
            "r", "q", "market_price", "volume", "oi"]
    return df[cols]


def main():
    ce = load_one(CE_FILE, "CE")
    pe = load_one(PE_FILE, "PE")
    combined = pd.concat([ce, pe], ignore_index=True)
    combined = combined.sort_values(["date", "expiry", "strike", "option_type"])

    combined["date"] = combined["date"].dt.date.astype(str)
    combined["expiry"] = combined["expiry"].dt.date.astype(str)

    combined.to_csv(OUT_FILE, index=False)
    print(f"{len(combined)} quotes across {combined['date'].nunique()} trading days -> {OUT_FILE}")
    print(f"date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"expiries: {combined['expiry'].nunique()}")


if __name__ == "__main__":
    main()
