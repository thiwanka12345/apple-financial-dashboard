import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Apple Inc. (AAPL) Advanced Financial & Investment Analytics",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Apple Inc. (AAPL) - Advanced Corporate Finance & Analytics Dashboard")
st.markdown("""
**Corporate Finance | Portfolio Optimization | Financial Forecasting | Churn Analysis**  
*Built for Portfolio Showcase & Executive Decision Making*
""")
st.markdown("---")

# ---------------------------------------------------------
# Data Caching Functions
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_aapl_financials():
    ticker = yf.Ticker("AAPL")
    income = ticker.financials
    bs = ticker.balance_sheet
    cf = ticker.cashflow
    return income, bs, cf

@st.cache_data(ttl=3600)
def get_stock_data(tickers, years=3):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    data = yf.download(tickers, start=start_date, end=end_date)
    if 'Adj Close' in data:
        df= data['Adj Close']
    elif 'Close' in data:
        df = data['Close']
    else:
        df = data

    return df

# Sidebar Navigation
st.sidebar.header("🧭 Dashboard Navigation")
menu = st.sidebar.radio(
    "Select Module:",
    [
        "1. Financial Ratios & Strategy",
        "2. Portfolio Risk & Return Optimization",
        "3. Financial Forecasting (Time Series)",
        "4. Customer Churn & Revenue Impact"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tech Stack:** Python, Streamlit, yfinance, Plotly, SciPy Optimization, Statsmodels")

# =========================================================
# MODULE 1: FINANCIAL RATIOS & STRATEGY
# =========================================================
if menu == "1. Financial Ratios & Strategy":
    st.header("📊 Corporate Finance & Ratio Analysis")
    
    with st.spinner("Fetching Apple's SEC Financial Statements via Yahoo Finance..."):
        try:
            income, bs, cf = get_aapl_financials()
            
            # Historical Dates (Years)
            dates = income.columns.strftime('%Y').tolist()[::-1]
            
            revenue = (income.loc['Total Revenue'][::-1] / 1e9).round(2)
            gross_profit = (income.loc['Gross Profit'][::-1] / 1e9).round(2)
            net_income = (income.loc['Net Income'][::-1] / 1e9).round(2)
            
            current_assets = (bs.loc['Current Assets'][::-1] / 1e9).round(2)
            current_liab = (bs.loc['Current Liabilities'][::-1] / 1e9).round(2)
            total_debt = (bs.loc['Total Debt'][::-1] / 1e9).round(2) if 'Total Debt' in bs.index else (bs.loc['Long Term Debt'][::-1] / 1e9).round(2)
            equity = (bs.loc['Stockholders Equity'][::-1] / 1e9).round(2)
            
            gross_margin = ((gross_profit / revenue) * 100).round(2)
            net_margin = ((net_income / revenue) * 100).round(2)
            current_ratio = (current_assets / current_liab).round(2)
            debt_to_equity = (total_debt / equity).round(2)
            roe = ((net_income / equity) * 100).round(2)

            # Top KPI Summary Cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latest Revenue", f"${revenue.iloc[-1]} B", f"{((revenue.iloc[-1]/revenue.iloc[-2])-1)*100:.1f}% YoY")
            col2.metric("Gross Margin", f"{gross_margin.iloc[-1]}%", f"{gross_margin.iloc[-1]-gross_margin.iloc[-2]:.1f}% YoY")
            col3.metric("Net Margin", f"{net_margin.iloc[-1]}%", f"{net_margin.iloc[-1]-net_margin.iloc[-2]:.1f}% YoY")
            col4.metric("Current Ratio", f"{current_ratio.iloc[-1]}", "Negative Working Capital")

            st.markdown("### Historical Financial Performance")
            
            tab1, tab2 = st.tabs(["📈 Profitability & Ratios", "💼 Strategic Insights"])
            
            with tab1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=dates, y=revenue, name="Revenue ($B)", marker_color='#007AFF'), secondary_y=False)
                fig.add_trace(go.Scatter(x=dates, y=gross_margin, name="Gross Margin %", mode='lines+markers', line=dict(color='#FF9500', width=3)), secondary_y=True)
                fig.add_trace(go.Scatter(x=dates, y=net_margin, name="Net Margin %", mode='lines+markers', line=dict(color='#34C759', width=3)), secondary_y=True)
                fig.update_layout(title="Revenue vs Profit Margins (Multi-Year Trend)", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                # Ratio Breakdown Table
                ratio_df = pd.DataFrame({
                    "Fiscal Year": dates,
                    "Revenue ($B)": revenue.values,
                    "Gross Margin (%)": gross_margin.values,
                    "Net Margin (%)": net_margin.values,
                    "Current Ratio": current_ratio.values,
                    "Debt-to-Equity": debt_to_equity.values,
                    "Return on Equity (%)": roe.values
                }).set_index("Fiscal Year")
                st.dataframe(ratio_df, use_container_width=True)

            with tab2:
                st.markdown("""
                ### 🎯 Strategic Corporate Finance Recommendations
                
                #### 1. Capital Allocation & Aggressive Stock Buybacks
                * **Observation:** Apple generates over $100B in annual Operating Cash Flow. It prioritizes **Share Repurchases** over large acquisitions.
                * **Impact:** Buying back billions in shares reduces the float, driving **EPS (Earnings Per Share)** and **ROE (Return on Equity)** significantly higher even when overall top-line growth is modest.
                
                #### 2. High-Margin Services Transition
                * **Observation:** Hardware products (iPhone, Mac) carry a ~36% gross margin, whereas **Services (iCloud, Apple Music, App Store)** carry a **70%+ gross margin**.
                * **Impact:** As the Services user base grows, Apple's overall Gross Margin expands YoY, shielding total profitability during hardware supply chain disruptions.
                
                #### 3. Negative Working Capital Efficiency
                * **Observation:** Apple's Current Ratio sits below 1.0 (approx 0.85 - 0.95). In traditional finance, this signals liquidity issues, but for Apple, it represents high operational efficiency.
                * **Impact:** Apple collects cash from customers almost immediately while negotiating 60-90 day payment terms with suppliers (Accounts Payable). It funds daily operations using supplier credit (Negative Cash Conversion Cycle).
                """)
        except Exception as e:
            st.error(f"Failed to pull live SEC data: {e}")

# =========================================================
# MODULE 2: PORTFOLIO OPTIMIZATION (RISK VS RETURN)
# =========================================================
elif menu == "2. Portfolio Risk & Return Optimization":
    st.header("🎯 Stock Market / Investment Portfolio Optimization")
    st.markdown("A Modern Portfolio Theory (MPT) model optimizing a 4-stock Tech Portfolio including **Apple (AAPL)** to maximize the **Sharpe Ratio**.")
    
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    selected_tickers = st.multiselect("Select Portfolio Assets:", tickers, default=tickers)
    years = st.slider("Historical Data Period (Years):", 1, 5, 3)
    
    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 assets to optimize portfolio diversification.")
    else:
        with st.spinner("Downloading stock price data & running Modern Portfolio Optimization..."):
            df_prices = get_stock_data(selected_tickers, years=years)
            returns = df_prices.pct_change().dropna()
            
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            risk_free_rate = 0.045  # 4.5% US Treasury Rate Assumption
            
            # Portfolio Performance Calculator Function
            def portfolio_performance(weights, mean_returns, cov_matrix):
                ret = np.sum(mean_returns * weights)
                std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return ret, std

            # Negative Sharpe Ratio Function (For SciPy Minimizer)
            def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
                p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
                return -(p_ret - risk_free_rate) / p_std

            # Optimization Constraints
            num_assets = len(selected_tickers)
            args = (mean_returns, cov_matrix, risk_free_rate)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            init_guess = num_assets * [1. / num_assets,]

            # Optimal Portfolio Computation
            opt_results = minimize(neg_sharpe_ratio, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            opt_weights = opt_results.x
            opt_ret, opt_std = portfolio_performance(opt_weights, mean_returns, cov_matrix)
            opt_sharpe = (opt_ret - risk_free_rate) / opt_std

            # Monte Carlo Simulation for Efficient Frontier Visual
            num_portfolios = 2000
            mc_returns, mc_volatility, mc_sharpe = [], [], []
            
            for _ in range(num_portfolios):
                w = np.random.random(num_assets)
                w /= np.sum(w)
                r, s = portfolio_performance(w, mean_returns, cov_matrix)
                mc_returns.append(r)
                mc_volatility.append(s)
                mc_sharpe.append((r - risk_free_rate) / s)

            # Results Display
            col1, col2, col3 = st.columns(3)
            col1.metric("Optimal Portfolio Return (Annualized)", f"{opt_ret*100:.2f}%")
            col2.metric("Portfolio Risk (Volatility)", f"{opt_std*100:.2f}%")
            col3.metric("Maximized Sharpe Ratio", f"{opt_sharpe:.2f}")

            st.markdown("### Optimized Asset Allocation Weights")
            col_chart, col_weights = st.columns([2, 1])
            
            with col_weights:
                weights_df = pd.DataFrame({
                    'Ticker': selected_tickers,
                    'Optimal Weight (%)': (opt_weights * 100).round(2)
                }).sort_values(by='Optimal Weight (%)', ascending=False)
                st.dataframe(weights_df, use_container_width=True)

            with col_chart:
                fig_weights = px.pie(weights_df, values='Optimal Weight (%)', names='Ticker', title="Maximum Sharpe Ratio Allocation", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_weights, use_container_width=True)

            # Efficient Frontier Plot
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=mc_volatility, y=mc_returns, mode='markers',
                marker=dict(color=mc_sharpe, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe Ratio")),
                name="Simulated Portfolios"
            ))
            fig_ef.add_trace(go.Scatter(
                x=[opt_std], y=[opt_ret], mode='markers+text',
                marker=dict(color='red', size=15, symbol='star'),
                text=["Max Sharpe Portfolio"], textposition="top center",
                name="Optimal Target"
            ))
            fig_ef.update_layout(title="Efficient Frontier (Monte Carlo Simulation)", xaxis_title="Annualized Volatility (Risk)", yaxis_title="Annualized Return", hovermode="closest")
            st.plotly_chart(fig_ef, use_container_width=True)

# =========================================================
# MODULE 3: FINANCIAL FORECASTING & TIME SERIES
# =========================================================
elif menu == "3. Financial Forecasting (Time Series)":
    st.header("📈 Financial Forecasting & Time Series Analysis")
    st.markdown("Predicting **Apple's Stock Price / Sales Trend** for future periods using Time Series Modeling (ARIMA).")

    forecast_days = st.slider("Select Forecast Horizon (Trading Days):", 30, 252, 90)
    
    with st.spinner("Fetching Apple's historical daily closing prices..."):
        df_aapl = get_stock_data(['AAPL'], years=3)
        if isinstance(df_aapl, pd.DataFrame):
            prices = df_aapl['AAPL'].dropna()
        else:
            prices = df_aapl.dropna()

        # Fit ARIMA Model (Order 1,1,1 for Stock Price Random Walk)
        model = ARIMA(prices, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast
        forecast_res = model_fit.get_forecast(steps=forecast_days)
        forecast_index = [prices.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
        forecast_values = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        latest_price = prices.iloc[-1]
        projected_price = forecast_values.iloc[-1]
        pct_change = ((projected_price - latest_price) / latest_price) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Current AAPL Stock Price", f"${latest_price:.2f}")
        col2.metric(f"Forecasted Price ({forecast_days} Days)", f"${projected_price:.2f}", f"{pct_change:+.2f}% Growth")
        col3.metric("Model Used", "ARIMA (1, 1, 1)", "Statistical Time Series")

        # Plot Historical + Forecast
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=prices.index[-250:], y=prices.values[-250:], name="Historical Daily Close", line=dict(color='#007AFF', width=2)))
        fig_fc.add_trace(go.Scatter(x=forecast_index, y=forecast_values, name="ARIMA Forecast Mean", line=dict(color='#FF9500', width=2, dash='dash')))
        
        # Confidence Interval Band
        fig_fc.add_trace(go.Scatter(
            x=forecast_index + forecast_index[::-1],
            y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0])[::-1],
            fill='toself',
            fillcolor='rgba(255, 149, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval"
        ))

        fig_fc.update_layout(title=f"Apple Inc. (AAPL) Price Projection ({forecast_days} Days Horizon)", xaxis_title="Date", yaxis_title="Stock Price ($)", hovermode="x unified")
        st.plotly_chart(fig_fc, use_container_width=True)

        st.info("💡 **Methodology Note:** Stock prices follow geometric Brownian motion and high variance. The ARIMA model provides a statistical baseline with upper/lower bounds calculated at a 95% confidence interval.")

# =========================================================
# MODULE 4: CUSTOMER CHURN & REVENUE IMPACT (SERVICES)
# =========================================================
elif menu == "4. Customer Churn & Revenue Impact":
    st.header("🔄 Customer Churn & Revenue Impact Analysis (Apple Services Ecosystem)")
    st.markdown("Analyzing how subscription churn across **iCloud, Apple Music, TV+, and Apple One** impacts recurring ARR (Annual Recurring Revenue) and overall valuation.")

    col_input, col_kpi = st.columns([1, 2])
    
    with col_input:
        st.subheader("Subscribers & Churn Parameters")
        active_subscribers = st.number_input("Paid Services Subscribers (Millions):", min_value=100, max_value=2000, value=1000, step=50)
        arpu_monthly = st.slider("Average Revenue Per User (ARPU / Month $):", 5.0, 30.0, 11.5, 0.5)
        current_churn = st.slider("Monthly Churn Rate (%):", 0.5, 5.0, 1.2, 0.1)
        cac = st.number_input("Customer Acquisition Cost (CAC $):", value=45.0, step=5.0)

    with col_kpi:
        st.subheader("Financial Impact Metrics")
        
        # Financial Calculations
        mrr = active_subscribers * arpu_monthly  # in Millions
        arr = mrr * 12  # in Millions
        monthly_lost_subs = active_subscribers * (current_churn / 100)
        annual_lost_subs = monthly_lost_subs * 12
        annual_revenue_loss = annual_lost_subs * arpu_monthly * 12
        ltv = (arpu_monthly * (1 / (current_churn / 100))) - cac
        ltv_cac_ratio = ltv / cac if cac > 0 else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Annual Recurring Revenue (ARR)", f"${arr:,.0f} Million")
        m2.metric("Annual Revenue Lost to Churn", f"${annual_revenue_loss:,.0f} Million", delta=f"-{annual_revenue_loss/arr*100:.1f}% ARR", delta_color="inverse")
        m3.metric("Customer Lifetime Value (LTV)", f"${ltv:.2f}", f"LTV/CAC Ratio: {ltv_cac_ratio:.1f}x")

    st.markdown("---")
    st.markdown("### 📉 Churn Sensitivity & Retention Strategy Simulation")

    # Sensitivity Table (Varying Churn Rates)
    churn_rates = np.arange(0.5, 3.5, 0.25)
    sensitivity_data = []

    for rate in churn_rates:
        lost_rev = (active_subscribers * (rate / 100) * 12) * arpu_monthly * 12
        retained_arr = arr - lost_rev
        c_ltv = (arpu_monthly * (1 / (rate / 100))) - cac
        sensitivity_data.append({
            "Monthly Churn Rate (%)": rate,
            "Annual Revenue Loss ($M)": round(lost_rev, 1),
            "Retained ARR ($M)": round(retained_arr, 1),
            "Customer LTV ($)": round(c_ltv, 2)
        })

    df_sens = pd.DataFrame(sensitivity_data)

    col_chart, col_strat = st.columns([3, 2])

    with col_chart:
        fig_sens = px.line(df_sens, x="Monthly Churn Rate (%)", y="Annual Revenue Loss ($M)", title="Impact of Monthly Churn Rate on Annual Revenue Loss ($ Millions)", markers=True, color_discrete_sequence=['#FF3B30'])
        fig_sens.add_vline(x=current_churn, line_dash="dash", line_color="green", annotation_text="Current Selected Churn")
        st.plotly_chart(fig_sens, use_container_width=True)

    with col_strat:
        st.markdown("""
        ### 🛡️ Churn Mitigation Strategies for Apple Services
        
        1. **Apple One Bundling Strategy:**  
           * Combining Apple Music, TV+, Arcade, and iCloud into single tier plans reduces churn by over **30-40%** compared to standalone app subscriptions.
           
        2. **Hardware Eco-System Lock-in:**  
           * Deep integration with iOS hardware creates high switching costs for end-users, lowering structural churn.
           
        3. **Predictive AI Churn Intervention:**  
           * Using machine learning to flag users with declining app engagement in Apple TV+ or Fitness+ and offering targeted multi-month promotional extensions before subscription cancellation occurs.
        """)