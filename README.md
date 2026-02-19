# Black–Scholes Option Pricing (Analytic & Monte Carlo)

This project implements pricing for European options under the Black–Scholes framework using:

- A **closed-form analytic solution**
- A **Monte Carlo simulation** for numerical approximation and validation

The aim is to demonstrate both theoretical understanding of option pricing and practical numerical implementation in Python.


Models Implemented
Analytic Black–Scholes Model -European call and put options
assumptions:
- Lognormally distributed asset prices
- Constant volatility
- Constant risk-free interest rate
- No dividends
- No arbitrage

This analytic price is used as a benchmark.


Monte Carlo Simulation
The Monte Carlo pricer simulates the terminal asset price ( Assest price at time t) 

\[
S_T = S_0 \exp\left((r - \frac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z\right)
\]

where \( Z \sim \mathcal{N}(0,1) \).

Steps:
1. Draw standard normal random variables
2. Simulate terminal prices \( S_T \)
3. Compute option payoffs at maturity
4. Discount the expected payoff back to today




How to Use

Run the script from the project directory:

```bash
python black_scholes.py

