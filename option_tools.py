# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta, date
from forward_volatility import fetch_and_process_data
from straddle_prices import fetch_straddle_prices
from vrp import create_implied_VRP_chart, create_realized_VRP_chart
from corr import plot_btc_altcoin_correlations


st.set_page_config(layout="wide",
                   page_icon="🍉")

# Streamlit UI
st.title("Options Analysis")

# Currency selection on main page
col1, col2, col3, col4 = st.columns(4) # make the select box smaller and tidy

with col1:
    currency = st.selectbox("Select Currency", ["BTC", "ETH"], key="currency_selector")

# Sidebar navigation
page = st.sidebar.selectbox("Functions", ["Forward Volatility", "Straddle Prices","VRP","Correlation Heatmap"])

# Forward Volatility Page
if page == "Forward Volatility":
    # Fetch data
    forward_matrix, atm_df, current_time = fetch_and_process_data(currency)

    t = current_time.strftime('%Y-%m-%d %H:%M:%S')

    st.write(f"Data fetched at: UTC ", t)

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

    # Display ATM IVs
    st.subheader("ATM IV")
    
    col1, col2 = st.columns(2)

    with col1:

        # Plot IV across expiries
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
            line=dict(width=2),
            marker=dict(size=8)
        )
        iv_fig.update_layout(
            hoverlabel=dict(bgcolor="white", font_size=14),
            height=400,
            showlegend=False
        )
        st.plotly_chart(iv_fig, use_container_width=True)

    with col2:

        # Display ATM IV table
        atm_display = atm_df[['dte', 'expiry_str', 'atm_iv']].rename(columns={
            'dte': 'Days to Expiry (dte)',
            'expiry_str': 'Expiration Date',
            'atm_iv': 'ATM IV (%)'
        }).round({'ATM IV (%)': 2})
        styled_df = atm_display.style.format({'ATM IV (%)': '{:.2f}'}) \
            .background_gradient(subset=['ATM IV (%)'], cmap='RdYlGn_r')
        st.table(styled_df)

# Straddle Prices Page
elif page == "Straddle Prices":
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

elif page == "Correlation Heatmap":

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
