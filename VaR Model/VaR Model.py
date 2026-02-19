import pandas as pd
import numpy as np  
## Need to pull the data from Yahoo Finance using the yfinance library
import yfinance as yf
## Symbol for the stock we want to analyze (SPY in this case)
symbol = 'SPY'
## Download historical data for the stock (last 5 years)        
data = yf.download(symbol, period='5y')
print(data.head(3))
## Calculate daily returns from the adjusted closing prices
data['Returns'] = data['Close'].pct_change()



def calculate_var(returns, confidence_level=0.95):
    # Calculate the VaR at the specified confidence level
    var = np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    return var

var_95 = calculate_var(data['Returns'], confidence_level=0.95)
print(f"Value at Risk (VaR) at 95% confidence level: {var_95:.4f}")

def calculate_parametric_var(returns, confidence_level=0.95):
    mean = returns.mean()
    std = returns.std()
    Pvar = mean - 1.645 * std  # 1.645 is the z-score for 95%
    return Pvar
parametric_var_95 = calculate_parametric_var(data['Returns'], confidence_level=0.95)
print(f"Parametric VaR at 95% confidence level: {parametric_var_95:.4f}")

from scipy import stats

def calculate_montecarlo_var(returns, confidence_level=0.95, simulations=10000):
    mean = returns.mean()
    std = returns.std()
    
    # Simulate 10,000 possible daily returns
    simulated_returns = np.random.normal(mean, std, simulations)
    
    # Find the VaR from the simulated returns
    MCvar = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return MCvar
montecarlo_var_95 = calculate_montecarlo_var(data['Returns'], confidence_level=0.95, simulations=10000)
print(f"Monte Carlo VaR at 95% confidence level: {montecarlo_var_95:.4f}")


def calculate_montecarlo_var_with_t_dist(returns, confidence_level=0.95, simulations=10000, df=5):
    mean = returns.mean()
    std = returns.std()
   

    # Simulate returns using a t-distribution
    simulated_returns = mean + std * np.random.standard_t(df, size=simulations)
    
    # Find the VaR from the simulated returns
    MCvar_t = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return MCvar_t
montecarlo_var_t_95 = calculate_montecarlo_var_with_t_dist(data['Returns'], confidence_level=0.95, simulations=10000, df=5)
print(f"Monte Carlo VaR with t-distribution at 95% confidence level: {montecarlo_var_t_95:.4f}")