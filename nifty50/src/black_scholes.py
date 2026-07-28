# Black-Scholes-Merton pricer, with a dividend/carry yield q built in
# (needed for an index like Nifty50, unlike the plain no-dividend version
# usually taught first).
#
#   d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
#   d2 = d1 - sigma*sqrt(T)
#   Call = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
#   Put  = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class BSInputs:
    S: float       # spot
    K: float       # strike
    T: float       # time to expiry (years)
    r: float       # risk-free rate
    q: float       # dividend/carry yield
    sigma: float   # volatility


def _d1_d2(inp: BSInputs):
    S, K, T, r, q, sigma = inp.S, inp.K, inp.T, inp.r, inp.q, inp.sigma
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(inp: BSInputs) -> float:
    S, K, T, r, q = inp.S, inp.K, inp.T, inp.r, inp.q
    d1, d2 = _d1_d2(inp)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(inp: BSInputs) -> float:
    S, K, T, r, q = inp.S, inp.K, inp.T, inp.r, inp.q
    d1, d2 = _d1_d2(inp)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_price(option_type: str, inp: BSInputs) -> float:
    option_type = option_type.upper()
    if option_type in ("CE", "CALL", "C"):
        return bs_call_price(inp)
    elif option_type in ("PE", "PUT", "P"):
        return bs_put_price(inp)
    raise ValueError(f"Unknown option_type: {option_type}")


def greeks(option_type: str, inp: BSInputs) -> dict:
    """Return delta, gamma, vega, theta, rho for the given option."""
    S, K, T, r, q, sigma = inp.S, inp.K, inp.T, inp.r, inp.q, inp.sigma
    d1, d2 = _d1_d2(inp)
    pdf_d1 = norm.pdf(d1)
    is_call = option_type.upper() in ("CE", "CALL", "C")

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100  # per 1% vol change

    if is_call:
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_volatility(option_type: str, market_price: float, inp: BSInputs,
                        lo: float = 1e-4, hi: float = 5.0) -> float:
    """
    Back out the Black-Scholes implied volatility that reprices `market_price`,
    using Brent's method root-finding on sigma.
    Returns np.nan if no solution is bracketed (e.g. price violates no-arbitrage bounds).
    """
    def objective(sigma):
        trial = BSInputs(inp.S, inp.K, inp.T, inp.r, inp.q, sigma)
        return bs_price(option_type, trial) - market_price

    try:
        f_lo, f_hi = objective(lo), objective(hi)
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(objective, lo, hi, xtol=1e-6, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan
