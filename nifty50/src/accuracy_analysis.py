"""
accuracy_analysis.py
---------------------
Backtests the Black-Scholes model against real (or, here, realistically
simulated -- see generate_sample_data.py) Nifty50 index option prices.

For every quote in the dataset:
  1. Back out the market-implied volatility (IV) for a *reference* window
     (rolling median IV of ATM options on that date) -- this is the single
     volatility estimate a trader could plausibly have used going INTO the
     Black-Scholes formula that day (i.e. we don't use each option's own
     IV, or BS would trivially reprice it perfectly).
  2. Price every option that day with Black-Scholes using that single
     volatility estimate.
  3. Compare BS price to the actual market price.

Error metrics reported: MAE, RMSE, MAPE, Mean Bias, and R^2 -- overall,
and broken down by moneyness bucket and time-to-expiry bucket, since BS
accuracy is well known to vary systematically across the smile.

Usage
-----
    python src/accuracy_analysis.py --data data/nifty50_options_sample.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from black_scholes import BSInputs, bs_price, implied_volatility


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "expiry"])
    return df


def estimate_daily_atm_vol(df: pd.DataFrame) -> pd.Series:
    """
    For each date, compute the market-implied vol of the options closest
    to at-the-money (|ln(K/S)| smallest), and take the median across CE/PE.
    This is the single volatility input BS uses to price the WHOLE day's
    option chain -- the realistic backtest scenario.
    """
    df = df.copy()
    df["moneyness"] = np.log(df["strike"] / df["spot"])
    df["abs_moneyness"] = df["moneyness"].abs()

    atm_vols = {}
    for date, grp in df.groupby("date"):
        atm_rows = grp.nsmallest(6, "abs_moneyness")
        ivs = []
        for _, row in atm_rows.iterrows():
            inp = BSInputs(S=row["spot"], K=row["strike"], T=row["T"],
                            r=row["r"], q=row["q"], sigma=0.15)
            iv = implied_volatility(row["option_type"], row["market_price"], inp)
            if not np.isnan(iv):
                ivs.append(iv)
        atm_vols[date] = np.median(ivs) if ivs else np.nan
    return pd.Series(atm_vols, name="atm_vol")


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    atm_vol = estimate_daily_atm_vol(df)
    df = df.copy()
    df["atm_vol"] = df["date"].map(atm_vol)
    df = df.dropna(subset=["atm_vol"])

    bs_prices = []
    for _, row in df.iterrows():
        inp = BSInputs(S=row["spot"], K=row["strike"], T=row["T"],
                        r=row["r"], q=row["q"], sigma=row["atm_vol"])
        bs_prices.append(bs_price(row["option_type"], inp))
    df["bs_price"] = bs_prices

    df["error"] = df["bs_price"] - df["market_price"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["error"] / df["market_price"] * 100
    df["moneyness"] = np.log(df["strike"] / df["spot"])
    return df


def error_metrics(df: pd.DataFrame) -> dict:
    err = df["error"]
    mkt = df["market_price"]
    ss_res = (err ** 2).sum()
    ss_tot = ((mkt - mkt.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "n_obs": len(df),
        "MAE": df["abs_error"].mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAPE_%": df["pct_error"].abs().mean(),
        "Mean_Bias": err.mean(),
        "R2": r2,
    }


def bucketed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["moneyness_bucket"] = pd.cut(
        df["moneyness"],
        bins=[-np.inf, -0.03, -0.01, 0.01, 0.03, np.inf],
        labels=["Deep OTM Put / ITM Call", "OTM Put", "ATM", "OTM Call", "Deep OTM Call / ITM Put"],
    )
    df["T_days"] = (df["T"] * 365).round()
    df["maturity_bucket"] = pd.cut(
        df["T_days"], bins=[0, 3, 7, 14, np.inf],
        labels=["0-3d", "4-7d", "8-14d", "15d+"],
    )

    rows = []
    for (mb, tb), grp in df.groupby(["moneyness_bucket", "maturity_bucket"], observed=True):
        if len(grp) < 3:
            continue
        m = error_metrics(grp)
        m["moneyness_bucket"] = mb
        m["maturity_bucket"] = tb
        rows.append(m)
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # 1. BS vs Market scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["market_price"], df["bs_price"], s=8, alpha=0.4, color="#2563eb")
    lims = [0, max(df["market_price"].max(), df["bs_price"].max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect agreement (y=x)")
    ax.set_xlabel("Market Price (₹)")
    ax.set_ylabel("Black-Scholes Price (₹)")
    ax.set_title("Black-Scholes vs Market Price — Nifty50 Options")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bs_vs_market_scatter.png"), dpi=150)
    plt.close(fig)

    # 2. Pricing error vs moneyness
    fig, ax = plt.subplots(figsize=(7, 5))
    for opt_type, color in [("CE", "#16a34a"), ("PE", "#dc2626")]:
        sub = df[df["option_type"] == opt_type]
        ax.scatter(sub["moneyness"], sub["pct_error"], s=8, alpha=0.35, color=color, label=opt_type)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Moneyness  ln(K/S)")
    ax.set_ylabel("% Pricing Error  (BS − Market) / Market")
    ax.set_title("Black-Scholes Pricing Error Across the Volatility Smile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_vs_moneyness.png"), dpi=150)
    plt.close(fig)

    # 3. Error distribution histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["pct_error"].clip(-100, 100), bins=60, color="#7c3aed", alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% Pricing Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Black-Scholes % Pricing Errors")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_distribution.png"), dpi=150)
    plt.close(fig)

    # 4. RMSE by maturity bucket
    df2 = df.copy()
    df2["T_days"] = (df2["T"] * 365).round()
    df2["maturity_bucket"] = pd.cut(
        df2["T_days"], bins=[0, 3, 7, 14, np.inf], labels=["0-3d", "4-7d", "8-14d", "15d+"])
    grp = df2.groupby("maturity_bucket", observed=True)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    grp.plot(kind="bar", ax=ax, color="#0ea5e9")
    ax.set_ylabel("Mean Absolute Error (₹)")
    ax.set_title("BS Pricing Error by Time-to-Expiry")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mae_by_maturity.png"), dpi=150)
    plt.close(fig)

    print(f"Saved 4 plots to {outdir}/")


def main():
    parser = argparse.ArgumentParser(description="Backtest Black-Scholes accuracy on Nifty50 options")
    parser.add_argument("--data", default="data/nifty50_options_sample.csv")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data)
    print(f"{len(df)} option quotes loaded across {df['date'].nunique()} trading days.")

    print("Running backtest (estimating daily ATM vol, pricing full chain with BS)...")
    results = run_backtest(df)

    overall = error_metrics(results)
    print("\n=== Overall Accuracy ===")
    for k, v in overall.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Accuracy by Moneyness x Maturity Bucket ===")
    buckets = bucketed_metrics(results)
    print(buckets.to_string(index=False))

    results.to_csv(os.path.join(args.outdir, "backtest_results.csv"), index=False)
    buckets.to_csv(os.path.join(args.outdir, "bucketed_metrics.csv"), index=False)
    pd.Series(overall).to_csv(os.path.join(args.outdir, "overall_metrics.csv"))

    make_plots(results, args.outdir)
    print(f"\nFull results saved to {args.outdir}/backtest_results.csv")


if __name__ == "__main__":
    main()
