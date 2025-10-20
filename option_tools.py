# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta, date
from forward_volatility import fetch_and_process_data
from straddle_prices import fetch_straddle_prices
from vrp import create_implied_VRP_chart, create_realized_VRP_chart, create_iv_rv_scatter_plot
from corr import plot_btc_altcoin_correlations, plot_btc_financial_correlations
from spot_vol import calculate_spot_vol_correlation
from rv import create_btc_volatility_chart, rv_table, create_rv_range_plot
from session_return import calculate_session_returns_and_volatility
from open_interest import get_deribit_options_data, process_options_data, create_bubble_chart
from hrh import historical_return_histogram
import matplotlib

st.set_page_config(layout="wide",
                   page_icon="🍉")

# Streamlit UI
st.title("watermelon 🍉")



# Sidebar navigation
page = st.sidebar.selectbox("Functions", ["Calculator", "Forward Volatility", "RV","RV Range", "Straddle Prices","VRP","Correlation Heatmap","Spot Vol Correlation","Session Return","Open Interest", "HRH"])

# Forward Volatility Page
if page == "Forward Volatility":
    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH"], key="currency_selector")

    forward_matrix, atm_df, current_time = fetch_and_process_data(currency)
    if forward_matrix is None or atm_df is None or current_time is None:
        st.error("Failed to fetch forward volatility data from Deribit API. Please try again later.")
    else:
        t = current_time.strftime('%Y-%m-%d %H:%M:%S')
        st.write(f"Data fetched at: UTC {t}")

        # Display forward volatility matrix
        st.subheader("Forward Volatility Matrix")
        display_matrix = forward_matrix.copy()
        fig = px.imshow(
            display_matrix,
            labels=dict(x="End Date", y="Start Date"),
            color_continuous_scale='Tealrose',
            color_continuous_midpoint=np.nanmean(display_matrix.values),
            aspect="auto",
            text_auto=".2f"
        )
        fig.update_layout(
            coloraxis_showscale=False,
            coloraxis_cmin=np.nanpercentile(display_matrix.values, 5) if not display_matrix.empty else 0,
            coloraxis_cmax=np.nanpercentile(display_matrix.values, 95) if not display_matrix.empty else 100,
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            ),
            title=f"{currency} Forward Volatility Matrix",
            xaxis_title="End Date",
            yaxis_title="Start Date",
            height=800,
            width=1800
        )
        fig.update_traces(
            hovertemplate="<b>Start:</b> %{y}<br><b>End:</b> %{x}<br><b>Forward Vol:</b> %{z:.2f}%<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Display ATM IV
        st.subheader("ATM IV")
        col1, col2 = st.columns(2)

        with col1:
            atm_df['dte'] = (atm_df['expiry'] - pd.Timestamp(current_time)).dt.days
            atm_df['expiry_str'] = atm_df['expiry'].dt.strftime('%m/%d/%y')
            iv_fig = px.line(
                atm_df,
                x='dte',
                y='atm_iv',
                markers=True,
                labels={'dte': 'Days to Expiry', 'atm_iv': 'ATM IV (%)'},
                title=f"{currency} ATM Implied Volatility Curve"
            )
            iv_fig.update_traces(
                hovertemplate="<b>Days to Expiry:</b> %{x}<br><b>ATM IV:</b> %{y:.2f}%<extra></extra>",
                line=dict(width=2, color='#4adaaf'),
                marker=dict(size=8, color='#4adaaf')
            )
            iv_fig.update_layout(
                hoverlabel=dict(bgcolor="white", font_size=14),
                height=400,
                showlegend=False
            )
            st.plotly_chart(iv_fig, use_container_width=True)

        with col2:
            atm_display = atm_df[['dte', 'expiry_str', 'atm_iv']].rename(columns={
                'dte': 'Days to Expiry (dte)',
                'expiry_str': 'Expiration Date',
                'atm_iv': 'ATM IV (%)'
            }).round({'ATM IV (%)': 2})
            custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'custom_gradient', ['#009392', '#d0587e']
            )
            styled_df = atm_display.style.format({'ATM IV (%)': '{:.2f}'}) \
                .background_gradient(subset=['ATM IV (%)'], cmap=custom_cmap)
            st.table(styled_df)

# Straddle Prices Page
elif page == "Straddle Prices":

    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH"], key="currency_selector")

    # Fetch data
    straddle_df, current_time, fig = fetch_straddle_prices(currency)

    st.write(f"Data fetched at: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Display straddle price chart
    st.subheader(f"{currency} ATM Straddle Prices")
    
    st.plotly_chart(fig, use_container_width=True)

    # Display straddle price table
    straddle_display = straddle_df[['dte', 'expiry', 'straddle_price', 'call_price', 'put_price']].rename(columns={
        'dte': 'Days to Expiry (dte)',
        'expiry': 'Expiration Date',
        'straddle_price': 'Straddle Price (USD)',
        'call_price': 'Call Price (USD)',
        'put_price': 'Put Price (USD)'
    })
    straddle_display['Expiration Date'] = straddle_display['Expiration Date'].dt.strftime('%m/%d/%y')

    st.divider()

    styled_df = straddle_display.style.format({
        'Straddle Price (USD)': '{:.2f}',
        'Call Price (USD)': '{:.2f}',
        'Put Price (USD)': '{:.2f}'
    }).background_gradient(subset=['Straddle Price (USD)'], cmap='YlOrRd')
    st.table(styled_df)

elif page == "VRP":

    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH"], key="currency_selector")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 1).date(),           
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),            
        )
    
    # Convert date_input (datetime.date) to datetime.datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())
    
    fig_implied_vrp = create_implied_VRP_chart(start_date=start_date, end_date=end_date, asset = currency)
    st.plotly_chart(fig_implied_vrp, use_container_width=True)

    st.divider()

    fig_realized_vrp = create_realized_VRP_chart(start_date=start_date, end_date=end_date, asset = currency)
    st.plotly_chart(fig_realized_vrp, use_container_width=True)

    st.divider()

    fig_scatter_vrp = create_iv_rv_scatter_plot(start_date=start_date, end_date=end_date, asset = currency)
    st.plotly_chart(fig_scatter_vrp, use_container_width=True)

elif page == "Correlation Heatmap":

    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH"], key="currency_selector")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 1).date(),           
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),            
        )
    
    with col3:
        max_lookback = st.number_input('Max Lookback Days', value = 100)

    fig1, fig2 = plot_btc_altcoin_correlations(start_date, end_date, max_lookback=100)

    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = plot_btc_financial_correlations(start_date, end_date, lookback=30)

    st.divider()
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Spot Vol Correlation":

    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH"], key="currency_selector")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        lookback_days = st.number_input("Lookback days: ", value = 365)

    fig1, fig2 = calculate_spot_vol_correlation(days = lookback_days)

    st.plotly_chart(fig1, use_container_width=False)

    st.divider()

    st.plotly_chart(fig2, use_container_width=True)

elif page == "RV":

    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH","SOL", "XRP", "HYPE","SUI", "VIRTUAL","TRUMP","ENA"], key="currency_selector")

    symbol = currency+"USDT.P"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 1).date(),           
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),            
        )

    fig, table_fig = create_btc_volatility_chart(start_date=start_date, end_date=end_date, symbol=symbol)

    st.plotly_chart(fig)

    st.divider()

    st.plotly_chart(table_fig)


elif page == "Session Return":

    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH"], key="currency_selector")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2025, 1, 1).date(),           
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),            
        )
    
    symbol = 'BTCUSDT' if currency == "BTC" else "ETHUSDT"

    result = calculate_session_returns_and_volatility(symbol = symbol, start_date = start_date, end_date = end_date)
    fig_session_return = px.line(result, x = 'date', y = ['Asia_Cumulative', 'US_Cumulative'], title = 'Return over Asia / US sessions')
    st.plotly_chart(fig_session_return)

    st.divider()

    fig_session_rv = px.line(result, x = 'date', y = ['Asia_Volatility', 'US_Volatility'], title = '14d High-Freq RV over Asia / US sessions')
    st.plotly_chart(fig_session_rv)

elif page == "Open Interest":

    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH"], key="currency_selector")

    coin = currency
    # Fetch data to get available ranges
    options_data = get_deribit_options_data(coin)
    df, expiry_dates, min_strike, max_strike = process_options_data(options_data)

    col1, col2, col3, col4 = st.columns(4)

    # Time range selection
    expiry_dates_str = [d.strftime("%b %d, %Y") for d in expiry_dates]
    with col1:
        start_date_str = st.selectbox("Select Start Date", expiry_dates_str)
    with col2:
        end_date_str = st.selectbox("Select End Date", expiry_dates_str)

    # Convert string dates back to datetime
    start_date = datetime.strptime(start_date_str, "%b %d, %Y")
    end_date = datetime.strptime(end_date_str, "%b %d, %Y")

    with col3:
        # Strike range selection
        min_strike_input, max_strike_input = st.slider(
            "Select Strike Price Range",
            min_value=float(min_strike),
            max_value=float(max_strike),
            value=(float(min_strike), float(max_strike)),
            step=100.0
        )

    # Generate and display chart
    fig = create_bubble_chart(coin, start_date, end_date, min_strike_input, max_strike_input)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected parameters.")

elif page == "HRH":

    # Currency selection on main page
    col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

    with col1:
        currency = st.selectbox("Select Currency", ["BTC", "ETH","SOL", "XRP", "HYPE","SUI", "VIRTUAL"], key="currency_selector")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2025, 1, 1).date(),           
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),            
        )
    # Convert date_input (datetime.date) to datetime.datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())

    with col3:
        return_interval = st.number_input('Return Interval', 1)

    fig_bar, fig_hist = historical_return_histogram(start_date=start_date, end_date=end_date, return_interval = return_interval, asset = currency)

    st.plotly_chart(fig_bar)

    st.divider()

    st.plotly_chart(fig_hist)

elif page == "Calculator":

    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    from datetime import datetime, date
    from scipy.stats import norm
    import uuid

    # Black-Scholes calculations
    def black_scholes(S, K, T, r, sigma, option_type):
        """Calculate Black-Scholes option price and Greeks."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "Call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Vega per 1% change in IV
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:  # Put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        return price, delta, gamma, vega, theta

    # Function to calculate portfolio payoff and Greeks
    def calculate_portfolio(S_range, options, eval_date, r=0.05):
        """Calculate portfolio payoff (including premium) and Greeks for a range of spot prices."""
        total_payoff = np.zeros_like(S_range)
        total_delta = np.zeros_like(S_range)
        total_gamma = np.zeros_like(S_range)
        total_vega = np.zeros_like(S_range)
        total_theta = np.zeros_like(S_range)
        
        for opt in options:
            K = opt['strike']
            T = max((opt['expiry'] - eval_date).days / 365, 1e-10)  # Avoid division by zero
            sigma = opt['iv']
            units = opt['units']
            
            # Calculate premium at the spot price when the leg was added
            premium = black_scholes(opt['spot'], K, max((opt['expiry'] - date.today()).days / 365, 1e-10), r, sigma, opt['type'])[0]
            
            if T > 0:  # Before expiry
                prices, deltas, gammas, vegas, thetas = black_scholes(S_range, K, T, r, sigma, opt['type'])
            else:  # At expiry
                if opt['type'] == "Call":
                    prices = np.maximum(S_range - K, 0)
                    deltas = np.where(S_range > K, 1, 0)
                    gammas = np.zeros_like(S_range)
                    vegas = np.zeros_like(S_range)
                    thetas = np.zeros_like(S_range)
                else:  # Put
                    prices = np.maximum(K - S_range, 0)
                    deltas = np.where(S_range < K, -1, 0)
                    gammas = np.zeros_like(S_range)
                    vegas = np.zeros_like(S_range)
                    thetas = np.zeros_like(S_range)
            
            # Adjust payoff for premium (cost for long, credit for short)
            total_payoff += (prices - premium) * units
            total_delta += deltas * units
            total_gamma += gammas * units
            total_vega += vegas * units
            total_theta += thetas * units
        
        return total_payoff, total_delta, total_gamma, total_vega, total_theta

    # Function to calculate individual leg details
    def calculate_leg_details(options, eval_date, spot_price, r=0.05):
        """Calculate price and Greeks for each leg at a specific date and spot price."""
        data = []
        for opt in options:
            K = opt['strike']
            T = max((opt['expiry'] - eval_date).days / 365, 1e-10)
            sigma = opt['iv']
            S = spot_price
            
            if T > 0:
                price, delta, gamma, vega, theta = black_scholes(S, K, T, r, sigma, opt['type'])
            else:
                price = np.maximum(S - K, 0) if opt['type'] == "Call" else np.maximum(K - S, 0)
                delta = 1 if opt['type'] == "Call" and S > K else -1 if opt['type'] == "Put" and S < K else 0
                gamma, vega, theta = 0, 0, 0
            
            data.append({
                'Type': opt['type'],
                'Strike': opt['strike'],
                'Expiry': opt['expiry'],
                'Units': opt['units'],
                'IV (%)': opt['iv'] * 100,
                'Price': price * opt['units'],
                'Delta': delta * opt['units'],
                'Gamma': gamma * opt['units'],
                'Vega': vega * opt['units'],
                'Theta': theta * opt['units']
            })
        
        return pd.DataFrame(data)

    # Streamlit app
    # st.set_page_config(page_title="Options Strategy Analyzer", layout="wide")

    # Custom CSS for a polished look
    st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }
        .stSidebar {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            color: #333333;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e6e6e6;
        }
        .stTabs [aria-selected="true"] {
            background-color: #007bff;
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #1a1a1a;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for option inputs
    st.sidebar.header("Option Legs Configuration")

    # Initialize session state for options
    if 'options' not in st.session_state:
        st.session_state.options = []

    # Input for spot price
    input_spot_price = st.sidebar.number_input("Spot Price", min_value=0.01, value=100.0, step=0.1)

    # Form for adding option legs
    with st.sidebar.form(key='option_form'):
        st.subheader("Add Option Leg")
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        strike = st.number_input("Strike Price", min_value=0.01, value=100.0, step=0.1)
        expiry = st.date_input("Expiry Date", min_value=date.today())
        units = st.number_input("Units (negative for short)", value=1, step=1)
        iv = st.number_input("Implied Volatility (%)", min_value=0.01, value=20.0, step=0.1)
        submit = st.form_submit_button("Add Option Leg")
        
        if submit:
            st.session_state.options.append({
                'type': option_type,
                'strike': strike,
                'expiry': expiry,
                'units': units,
                'iv': iv / 100,
                'spot': input_spot_price,
                'id': str(uuid.uuid4())
            })

    # Display and manage existing options
    if st.session_state.options:
        st.sidebar.subheader("Current Option Legs")
        for i, opt in enumerate(st.session_state.options):
            with st.sidebar.expander(f"Leg {i+1}: {opt['type']}"):
                st.write(f"Strike: {opt['strike']}")
                st.write(f"Expiry: {opt['expiry']}")
                st.write(f"Units: {opt['units']}")
                st.write(f"IV: {opt['iv']*100:.2f}%")
                if st.button("Remove", key=f"remove_{opt['id']}"):
                    st.session_state.options.pop(i)
                    st.rerun()

    # Analysis Parameters at the top
    st.header("Options Calculator")
    with st.container():
        st.subheader("Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            eval_date = st.date_input("Pricing Date", value=date.today(), key="eval_date")
        with col2:
            # Set default IV to first leg's IV if available, otherwise 20%
            default_iv = st.session_state.options[0]['iv'] * 100 if st.session_state.options else 20.0
            iv_adjust = st.slider("Adjust Implied Volatility (%)", 0.01, 100.0, default_iv, step=1.0)
        with col3:
            spot_price = st.slider("Adjust Spot Price", max(0.01, input_spot_price * 0.5), input_spot_price * 1.5, input_spot_price, step=0.1)

    # Adjust IV for all options
    adjusted_options = [
        {**opt, 'iv': iv_adjust / 100}
        for opt in st.session_state.options
    ]

    # Generate spot price range
    spot_range = np.linspace(max(0.01, spot_price * 0.5), spot_price * 1.5, 100)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Payoff Diagram", "Greeks Analysis", "Leg Details"])

    # Tab 1: Payoff Diagram
    with tab1:
        if not st.session_state.options:
            st.warning("Please add at least one option leg to view the payoff diagram.")
        else:
            # Calculate payoffs
            payoff_expiry, _, _, _, _ = calculate_portfolio(spot_range, adjusted_options, max(opt['expiry'] for opt in adjusted_options))
            payoff_current, _, _, _, _ = calculate_portfolio(spot_range, adjusted_options, eval_date)
            
            # Create Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=payoff_expiry,
                mode='lines',
                name='Payoff at Expiry',
                line=dict(dash='dash', color='#007bff')
            ))
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=payoff_current,
                mode='lines',
                name=f'Payoff at {eval_date}',
                line=dict(color='#ff6b6b')
            ))
            fig.add_vline(x=spot_price, line=dict(color='gray', dash='dot'), annotation_text="Current Spot")
            
            # Customize layout
            fig.update_layout(
                title="Options Strategy Payoff Diagram (Net of Premium)",
                xaxis_title="Spot Price",
                yaxis_title="Payoff ($)",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
                margin=dict(l=50, r=50, t=80, b=50),
                font=dict(family="Arial", size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Greeks Analysis
    with tab2:
        if not st.session_state.options:
            st.warning("Please add at least one option leg to view the Greeks analysis.")
        else:
            # Calculate Greeks
            _, delta, gamma, vega, theta = calculate_portfolio(spot_range, adjusted_options, eval_date)
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Delta", "Gamma", "Vega", "Theta"),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Add traces
            fig.add_trace(go.Scatter(x=spot_range, y=delta, mode='lines', name='Delta', line=dict(color='#007bff')), row=1, col=1)
            fig.add_trace(go.Scatter(x=spot_range, y=gamma, mode='lines', name='Gamma', line=dict(color='#ff6b6b')), row=1, col=2)
            fig.add_trace(go.Scatter(x=spot_range, y=vega, mode='lines', name='Vega', line=dict(color='#28a745')), row=2, col=1)
            fig.add_trace(go.Scatter(x=spot_range, y=theta, mode='lines', name='Theta', line=dict(color='#ffc107')), row=2, col=2)
            
            # Add vertical line for current spot
            for row in [1, 2]:
                for col in [1, 2]:
                    fig.add_vline(x=spot_price, line=dict(color='gray', dash='dot'), row=row, col=col)
            
            # Update layout
            fig.update_layout(
                title="Options Strategy Greeks",
                template="plotly_white",
                showlegend=False,
                height=600,
                margin=dict(l=50, r=50, t=80, b=50),
                font=dict(family="Arial", size=12)
            )
            fig.update_xaxes(title_text="Spot Price", row=2, col=1)
            fig.update_xaxes(title_text="Spot Price", row=2, col=2)
            fig.update_yaxes(title_text="Delta", row=1, col=1)
            fig.update_yaxes(title_text="Gamma", row=1, col=2)
            fig.update_yaxes(title_text="Vega", row=2, col=1)
            fig.update_yaxes(title_text="Theta", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Leg Details
    with tab3:
        if not st.session_state.options:
            st.warning("Please add at least one option leg to view the leg details.")
        else:
            # Calculate leg details
            leg_df = calculate_leg_details(adjusted_options, eval_date, spot_price)
            
            # Format the dataframe for display
            leg_df['Expiry'] = leg_df['Expiry'].apply(lambda x: x.strftime('%Y-%m-%d'))
            leg_df['IV (%)'] = leg_df['IV (%)'].round(2)
            leg_df['Price'] = leg_df['Price'].round(2)
            leg_df['Delta'] = leg_df['Delta'].round(4)
            leg_df['Gamma'] = leg_df['Gamma'].round(4)
            leg_df['Vega'] = leg_df['Vega'].round(4)
            leg_df['Theta'] = leg_df['Theta'].round(4)
            
            st.subheader(f"Option Legs Details at {eval_date}")
            st.dataframe(
                leg_df,
                use_container_width=True,
                column_config={
                    "Type": st.column_config.TextColumn("Option Type"),
                    "Strike": st.column_config.NumberColumn("Strike Price", format="$%.2f"),
                    "Expiry": st.column_config.TextColumn("Expiry Date"),
                    "Units": st.column_config.NumberColumn("Units"),
                    "IV (%)": st.column_config.NumberColumn("IV (%)", format="%.2f"),
                    "Price": st.column_config.NumberColumn("Price ($)", format="$%.2f"),
                    "Delta": st.column_config.NumberColumn("Delta", format="%.4f"),
                    "Gamma": st.column_config.NumberColumn("Gamma", format="%.4f"),
                    "Vega": st.column_config.NumberColumn("Vega", format="%.4f"),
                    "Theta": st.column_config.NumberColumn("Theta", format="%.4f")
                }
            )

elif page == "RV Range":
    st.subheader("7-Day RV Historical Range")
    st.markdown("This chart displays the current 7-day realized volatility (RV) compared to its historical high-low ranges over the past 20 and 60 days.")

    all_symbols = ["BTC", "ETH", "SOL", "BNB", "HYPE", "XRP", "DOGE", "SUI", "VIRTUAL", "TRUMP", "ENA"]
    selected_symbols = st.multiselect(
        "Select currencies to plot", 
        options=all_symbols, 
        default=["BTC", "ETH", "SOL", "BNB"]
    )
    
    if st.button("Generate Plot"):
        if not selected_symbols:
            st.warning("Please select at least one currency.")
        else:
            with st.spinner("Fetching data and generating plot... This may take a moment."):
                fig_rv_range = create_rv_range_plot(symbols=selected_symbols)
                st.plotly_chart(fig_rv_range, use_container_width=True)
    
    # ADD THE FOLLOWING CODE TO DISPLAY THE STYLED TABLE
    st.divider()
    st.subheader("Realized Volatility Summary")

    with st.spinner("Loading RV summary table..."):
        df_rv = rv_table()
        if not df_rv.empty:
            vol_cols = [col for col in df_rv.columns if 'Vol' in col]

            # Apply numerical formatting directly to the DataFrame
            df_display = df_rv.copy()
            df_display['Spot Price'] = df_display['Spot Price'].apply(lambda x: f'${x:,.2f}')
            for col in vol_cols:
                df_display[col] = df_display[col].apply(lambda x: f'{x:.2%}')
            
            # Display the dataframe without any custom styling
            st.dataframe(df_display, use_container_width=True, height=320)
        else:
            st.warning("Could not fetch data for the RV summary table.")

