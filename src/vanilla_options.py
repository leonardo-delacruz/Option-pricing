
import numpy as np
from scipy.stats import norm

def bs_call(S0, strike, r, sigma, T):
    """
  European Call Option
    """
    d1 = (np.log(S0 / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S0 * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    return call

def bs_put(S0, strike, r, sigma, T):
    """European put option"""
    d1 = (np.log(S0 / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put  = strike * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put




