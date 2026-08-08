# apple-financial-dashboard
🎯 Problem Statement
Evaluating mega-cap tech performance requires going beyond headline revenue numbers. Investors and corporate strategists face three core challenges:

Capital Efficiency vs. Growth: Understanding how Apple maintains industry-leading Return on Equity (ROE) despite slowing hardware volume growth.

Asset Allocation & Risk Minimization: Constructing an optimal multi-asset tech portfolio that maximizes risk-adjusted returns (Sharpe Ratio).

Subscription Churn Impact: Quantifying the revenue risk associated with subscriber churn in Apple's high-margin Services ecosystem ($109B+ annual segment).

📊 Dataset & Tools
Data Sources: Real-time and historical financial data pulled via yfinance (Yahoo Finance API) and verified against SEC EDGAR 10-K filings.

Historical Range: 5-Year Financial Statements (Income Statement, Balance Sheet, Cash Flow) and 3-Year Daily Adjusted Closing Prices for tech tickers (AAPL, MSFT, GOOGL, NVDA).

Tech Stack:

Core Language: Python 3.11+

Data Manipulation: pandas, numpy

Quantitative & Financial Modeling: scipy.optimize (Sequential Least Squares Programming), statsmodels (ARIMA Time Series)

Interactive UI & Visuals: Streamlit, Plotly Express, Plotly Graph Objects
⚙️ Methodology & Technical Architecture
[SEC EDGAR / Yahoo Finance API]
             │
             ▼
[Data Extraction & Processing (Pandas / NumPy)]
             │
 ┌───────────┼───────────────────┬──────────────────────┐
 ▼           ▼                   ▼                      ▼
[Financial  [Portfolio Opt.]    [Time Series]          [SaaS Churn]
 Ratio Engine] (SciPy SLSQP)    (ARIMA 1,1,1)          (ARR & LTV Model)
 └───────────┼───────────────────┴──────────────────────┘
             │
             ▼
[Interactive Dashboard UI (Streamlit & Plotly)]

1.Financial Ratio Computation: Calculated Gross Margin, Net Margin, Current Ratio, Debt-to-Equity, and Return on Equity across multi-year historical periods.
2.Portfolio Risk-Return Optimization (Modern Portfolio Theory):
  1.Computed annualized mean returns and variance-covariance matrix.
  2.Formulated an objective function using scipy.optimize.minimize to find weights that maximize the Sharpe Ratio ($\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}$) subject to $\sum w_i = 1$ and $w_i \ge 0$.
  3.Ran a 2,000-run Monte Carlo Simulation to plot the Efficient Frontier.
3.Time Series Forecasting: Fitted an ARIMA(1,1,1) model on historical daily closing prices to generate future price paths along with a 95% confidence interval.
4.SaaS Churn Sensitivity Analysis: Modeled monthly subscription churn across Services (iCloud, Apple Music, TV+) to quantify ARR loss, Customer Lifetime Value (LTV), and LTV/CAC ratios.

Key Insights & Strategic Recommendations
1. High-Margin Services Transition Drives Margin Expansion:
While hardware product margins hover around ~36%, Services gross margins exceed 75%. Services revenue surpassed $109B in FY2025, enabling total company Gross Margin to expand near 47%.
2. Aggressive Capital Allocation via Buybacks:
Operating cash flow ($110B+) is heavily directed toward share repurchases ($85B+ annually). Reducing shares outstanding artificially boosts Earnings Per Share (EPS) and maintains an exceptional Return on Equity (ROE).
3. Negative Working Capital Efficiency:
Apple maintains a Current Ratio below 1.0 (0.85–0.95). Rather than signaling liquidity distress, this reflects a Negative Cash Conversion Cycle where Apple collects cash from customers upfront while leveraging 60–90 day payment terms with hardware suppliers.
4. Portfolio Risk Optimization:
Under a 4.5% risk-free rate assumption, allocating across AAPL, MSFT, GOOGL, and NVDA via SciPy optimization achieves a significantly higher Sharpe Ratio compared to an equal-weighted benchmark.

# Apple Inc. (AAPL) Financial Health & Valuation Dashboard

Interactive dashboard for corporate finance analysis, portfolio optimization, and financial forecasting.

## 📊 Overview
This project provides a comprehensive analysis of Apple Inc. (AAPL), integrating real-time financial statements, stock market optimization, and time-series forecasting.

## 💡 Key Code Snippet (Portfolio Optimization)
This project uses **SciPy's SLSQP** algorithm to optimize stock weights for the maximum Sharpe Ratio:

```python
from scipy.optimize import minimize
import numpy as np

# Portfolio performance calculation
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, std

# Objective: Maximize Sharpe Ratio (Minimize negative)
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.045):
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std
