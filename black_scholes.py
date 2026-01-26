import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


# ===========================
# Configuration / Parameters
# ===========================
@dataclass
class OptionParams:
    S: float        # Spot price
    K: float        # Strike price
    r: float        # Risk-free rate (annual, cont. comp.)
    sigma: float    # Volatility (annual)
    T: float        # Time to maturity (years)
    is_call: bool   # Call or put


# ===========================
# Analytic Black–Scholes
# ===========================
def black_scholes_price(params: OptionParams) -> float:
    """
    Analytic Black–Scholes price for a European option (no dividends).
    """

    S, K, r, sigma, T = params.S, params.K, params.r, params.sigma, params.T

    if T <= 0:
        return max(S - K, 0.0) if params.is_call else max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)

    discount_factor = np.exp(-r * T)

    if params.is_call:
        return S * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)
    else:
        return K * discount_factor * norm.cdf(-d2) - S * norm.cdf(-d1)


# ===========================
# Monte Carlo Black–Scholes
# ===========================
def black_scholes_mc(
    params: OptionParams,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> float:
    """
    Monte Carlo price for a European option under Black–Scholes.
    """

    if seed is not None:
        np.random.seed(seed)

    S, K, r, sigma, T = params.S, params.K, params.r, params.sigma, params.T

    Z = np.random.standard_normal(n_sims)

    ST = S * np.exp(
        (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z
    )

    if params.is_call:
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

    return np.exp(-r * T) * np.mean(payoffs)


# ===========================
# Main execution
# ===========================
def main() -> None:
    """
    Example experiment comparing analytic and Monte Carlo prices.
    """

    call_params = OptionParams(
        S=100.0,
        K=100.0,
        r=0.03,
        sigma=0.2,
        T=1.0,
        is_call=True,
    )

    put_params = OptionParams(
        S=100.0,
        K=100.0,
        r=0.03,
        sigma=0.2,
        T=1.0,
        is_call=False,
    )

    call_bs = black_scholes_price(call_params)
    put_bs = black_scholes_price(put_params)

    call_mc = black_scholes_mc(call_params, seed=42)
    put_mc = black_scholes_mc(put_params, seed=42)

    print(f"Call (BS): {call_bs:.4f}")
    print(f"Call (MC): {call_mc:.4f}")
    print(f"Put  (BS): {put_bs:.4f}")
    print(f"Put  (MC): {put_mc:.4f}")


if __name__ == "__main__":
    main()

