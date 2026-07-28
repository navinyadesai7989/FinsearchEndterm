# quick sanity checks on the pricing engine
# run: python -m pytest src/test_black_scholes.py -v
import numpy as np
from black_scholes import BSInputs, bs_call_price, bs_put_price, implied_volatility
from monte_carlo import mc_european_price


def test_put_call_parity():
    inp = BSInputs(S=24800, K=24800, T=0.05, r=0.068, q=0.012, sigma=0.15)
    c = bs_call_price(inp)
    p = bs_put_price(inp)
    lhs = c - p
    rhs = inp.S * np.exp(-inp.q * inp.T) - inp.K * np.exp(-inp.r * inp.T)
    assert abs(lhs - rhs) < 1e-8, "Put-call parity violated"


def test_call_price_positive_and_bounded():
    inp = BSInputs(S=24800, K=25000, T=0.02, r=0.068, q=0.012, sigma=0.13)
    c = bs_call_price(inp)
    assert c > 0
    assert c < inp.S  # call can never be worth more than the underlying


def test_implied_vol_roundtrip():
    inp = BSInputs(S=24800, K=25000, T=0.05, r=0.068, q=0.012, sigma=0.18)
    price = bs_call_price(inp)
    iv = implied_volatility("CE", price, inp)
    assert abs(iv - 0.18) < 1e-4


def test_monte_carlo_matches_black_scholes():
    inp = BSInputs(S=24800, K=24800, T=7 / 365, r=0.068, q=0.012, sigma=0.13)
    bs = bs_call_price(inp)
    mc, se = mc_european_price("CE", inp, n_paths=300_000, seed=1)
    assert abs(bs - mc) < 6 * se  # within a comfortable multiple of MC standard error


if __name__ == "__main__":
    test_put_call_parity()
    test_call_price_positive_and_bounded()
    test_implied_vol_roundtrip()
    test_monte_carlo_matches_black_scholes()
    print("All tests passed.")
