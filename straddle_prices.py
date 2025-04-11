# straddle_prices.py
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Deribit API base URL
API_URL = "https://www.deribit.com/api/v2/public/"

def call_api(method, params):
    """Helper function to call Deribit API."""
    url = f"{API_URL}{method}"
    response = requests.get(url, params=params)
    return response.json()

def fetch_straddle_prices(currency):
    """
    Fetch ATM straddle prices for given currency.
    
    Args:
        currency (str): Either 'BTC' or 'ETH'
    
    Returns:
        tuple: (straddle_df, current_time, fig)
    """
    # Fetch server time and index price
    server_time_resp = call_api("get_time", {})
    server_time = server_time_resp['result']
    current_time = datetime.fromtimestamp(server_time / 1000)

    index_price_resp = call_api("get_index_price", {"index_name": f"{currency.lower()}_usd"})
    underlying_price = index_price_resp['result']['index_price']

    # Fetch options data
    instruments_resp = call_api("get_instruments", {"currency": currency, "kind": "option"})
    instruments = instruments_resp['result']
    instrument_dict = {inst['instrument_name']: inst for inst in instruments}

    book_summary_resp = call_api("get_book_summary_by_currency", {"currency": currency, "kind": "option"})
    book_summary = book_summary_resp['result']

    # Collect straddle data
    straddle_data = []
    for summary in book_summary:
        inst_name = summary['instrument_name']
        mark_price = summary.get('mark_price')
        if inst_name in instrument_dict and mark_price is not None:
            inst = instrument_dict[inst_name]
            expiry = datetime.fromtimestamp(inst['expiration_timestamp'] / 1000)
            time_to_expiry = (expiry - current_time).total_seconds() / (365.25 * 24 * 3600)
            if time_to_expiry > 0:
                straddle_data.append({
                    'expiry': expiry,
                    'strike': inst['strike'],
                    'option_type': inst['option_type'],
                    'mark_price': mark_price * underlying_price,  # Convert to USD
                    'instrument_name': inst_name
                })

    # Convert to DataFrame and find ATM straddles
    df = pd.DataFrame(straddle_data)
    grouped = df.groupby('expiry')
    straddle_prices = []
    for expiry in sorted(df['expiry'].unique()):
        group = grouped.get_group(expiry)
        group['strike_diff'] = abs(group['strike'] - underlying_price)
        min_diff = group['strike_diff'].min()
        atm_options = group[group['strike_diff'] == min_diff]
        call_atm = atm_options[atm_options['option_type'] == 'call']
        put_atm = atm_options[atm_options['option_type'] == 'put']
        if not call_atm.empty and not put_atm.empty:
            call_price = call_atm['mark_price'].mean()
            put_price = put_atm['mark_price'].mean()
            straddle_price = call_price + put_price
            straddle_prices.append({
                'expiry': expiry,
                'dte': (pd.Timestamp(expiry) - pd.Timestamp(current_time)).days,
                'straddle_price': straddle_price,
                'call_price': call_price,
                'put_price': put_price
            })

    # Create DataFrame
    straddle_df = pd.DataFrame(straddle_prices).sort_values('expiry')
    
    # Convert dte to string for categorical x-axis
    straddle_df['dte_str'] = straddle_df['dte'].astype(str)

    # Define colors from the screenshot
    colors = [
        '#A3D4A8',  # Light green (A)
        '#4AB8C1',  # Teal (B)
        '#A77BCA',  # Purple (C)
        '#AEC7E8',  # Light blue (D)
        '#FF9999'   # Peach (E)
    ]

    # Assign colors cyclically based on the number of bars
    bar_colors = [colors[i % len(colors)] for i in range(len(straddle_df))]

    # Create visualization (bar chart)
    fig = go.Figure(data=[
        go.Bar(
            x=straddle_df['dte_str'],
            y=straddle_df['straddle_price'],
            marker_color=bar_colors,
            width=0.8,  # Wider bars to match the screenshot
            text=straddle_df['straddle_price'].round(2),  # Add values on top
            textposition='outside',  # Position text above bars
            textfont=dict(size=12, color="#000000")  # Black text, size 12
        )
    ])
    fig.update_layout(
        xaxis_title="Days to Expiry",
        yaxis_title="Straddle Price (USD)",
        height=500,
        showlegend=False,
        # plot_bgcolor='rgba(240, 240, 240, 1)',  # Light gray background
        # paper_bgcolor='rgba(240, 240, 240, 1)',  # Light gray paper background
        font=dict(family="Arial", size=12, color="#000000"),
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            type='category',  # Even spacing
            showgrid=False,  # No x-axis grid
            zeroline=False
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=False,  # No y-axis grid
            zeroline=False,
            range=[0, max(straddle_df['straddle_price']) * 1.2]  # Extend y-axis for text
        ),
        margin=dict(l=50, r=50, t=20, b=50)  # Minimal top margin (no title)
    )
    fig.update_traces(
        hovertemplate="<b>Days to Expiry:</b> %{x}<br><b>Straddle Price:</b> $%{y:.2f}<extra></extra>",
        texttemplate='$%{text:.2f}'  # Format text as USD with two decimals
    )

    return straddle_df, current_time, fig