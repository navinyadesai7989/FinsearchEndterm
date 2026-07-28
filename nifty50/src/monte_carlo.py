# Prices European options by simulating GBM paths directly, mainly to
# cross-check the closed-form black_scholes.py against something built
# independently. Same lognormal assumptions as BS, so they should
# converge as n_paths grows -- any gap left is just sampling noise
# (reported below as a std error / 95% CI), not a model disagreement.
#
# Not to be confused with the accuracy_analysis.py backtest -- that one
# compares BS to real market prices (model risk), this compares BS to MC
# (pure numerical/simulation error, same underlying model).

import numpy as np
import matplotlib.pyplot as plt

from black_scholes import BSInputs, bs_price


def mc_european_price(option_type: str, inp: BSInputs, n_paths: int = 200_000,
                       antithetic: bool = True, seed: int = None):
    """
    Single-step Monte Carlo price of a European call/put under GBM.
    Returns (price, std_error).
    """
    rng = np.random.default_rng(seed)
    S, K, T, r, q, sigma = inp.S, inp.K, inp.T, inp.r, inp.q, inp.sigma

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

    if option_type.upper() in ("CE", "CALL", "C"):
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return price, std_error


def convergence_study(option_type: str, inp: BSInputs, path_counts=None, seed=7):
    if path_counts is None:
        path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    bs_ref = bs_price(option_type, inp)
    rows = []
    for n in path_counts:
        price, se = mc_european_price(option_type, inp, n_paths=n, seed=seed)
        rows.append({"n_paths": n, "mc_price": price, "std_error": se,
                      "bs_price": bs_ref, "abs_diff": abs(price - bs_ref)})
    return rows, bs_ref


def plot_convergence(rows, bs_ref, outpath="outputs/mc_convergence.png"):
    ns = [r["n_paths"] for r in rows]
    prices = [r["mc_price"] for r in rows]
    ses = [r["std_error"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ns, prices, yerr=[1.96 * s for s in ses], fmt="o-",
                color="#2563eb", ecolor="#93c5fd", capsize=3, label="MC price ± 95% CI")
    ax.axhline(bs_ref, color="red", linestyle="--", label=f"Black-Scholes = {bs_ref:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Option price (₹)")
    ax.set_title("Monte Carlo Convergence to the Black-Scholes Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved convergence plot -> {outpath}")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    # Example: an ATM Nifty50 weekly call, ~7 days to expiry -- using the
    # actual spot from the last date in the real dataset (27 Jul 2026)
    inp = BSInputs(S=23996, K=24000, T=7 / 365, r=0.065, q=0.013, sigma=0.14)

    print("Example: ATM Nifty50 Call, S=23996, K=24000, T=7d, r=6.5%, q=1.3%, sigma=14%")
    bs_val = bs_price("CE", inp)
    mc_val, mc_se = mc_european_price("CE", inp, n_paths=500_000, seed=7)
    print(f"  Black-Scholes closed-form price : {bs_val:.4f}")
    print(f"  Monte Carlo price (500k paths)   : {mc_val:.4f}  (std err {mc_se:.4f})")
    print(f"  95% CI                           : [{mc_val - 1.96*mc_se:.4f}, {mc_val + 1.96*mc_se:.4f}]")
    print(f"  Absolute difference              : {abs(bs_val - mc_val):.4f}")

    rows, bs_ref = convergence_study("CE", inp)
    print("\nConvergence study:")
    for r in rows:
        print(f"  n={r['n_paths']:>7,}  MC={r['mc_price']:.4f}  SE={r['std_error']:.4f}  |diff|={r['abs_diff']:.4f}")

    plot_convergence(rows, bs_ref)
