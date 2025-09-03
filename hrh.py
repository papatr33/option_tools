import pandas as pd
import plotly.graph_objects as go
from binance.client import Client
from datetime import datetime
import numpy as np

def historical_return_histogram(start_date, end_date, return_interval = 1, asset = 'BTC'):
    # Initialize Binance client (no API key needed for public data)
    # Note: If rate limits are hit, register for an API key at binance.com
    client = Client()
   

    # Function to fetch historical daily data
    def get_historical_data(symbol, interval, start_date, end_date):
        # Convert datetime objects to string format 'YYYY-MM-DD'
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Construct trading pair (e.g., BTCUSDT, ETHUSDT)
        trading_pair = f"{symbol}USDT"
        
        klines = client.get_historical_klines(trading_pair, interval, start_date_str, end_date_str)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        return df[['timestamp', 'close']]

    # Function to calculate n-day returns
    def calculate_interval_returns(df, interval):
        # Keep every nth row for the specified interval
        df_interval = df.iloc[::interval].copy()
        # Calculate returns based on close prices
        df_interval['returns'] = df_interval['close'].pct_change() * 100  # Returns in percentage
        return df_interval

    # Fetch Bitcoin daily data (BTCUSDT)
    df = get_historical_data(asset, Client.KLINE_INTERVAL_1DAY, start_date, end_date)

    # Calculate returns for the specified interval
    df_returns = calculate_interval_returns(df, return_interval)

    # Calculate standard deviations
    mean_return = df_returns['returns'].mean()
    std_return = df_returns['returns'].std()
    sd1_upper = mean_return + std_return
    sd1_lower = mean_return - std_return
    sd2_upper = mean_return + 2 * std_return
    sd2_lower = mean_return - 2 * std_return

    # Create Plotly figure
    fig = go.Figure()

    # Add bar chart for interval returns
    fig.add_trace(go.Bar(
        x=df_returns['timestamp'],
        y=df_returns['returns'],
        name=f'{return_interval}-Day Returns (%)',
        marker_color=['rgb(0, 147, 146)' if x >= 0 else 'rgb(208, 88, 126)' for x in df_returns['returns']]
    ))

    # Add standard deviation lines
    fig.add_trace(go.Scatter(
        x=df_returns['timestamp'], y=[sd1_upper] * len(df_returns),
        name='+1 SD', line=dict(color='rgb(0, 223, 221)', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df_returns['timestamp'], y=[sd1_lower] * len(df_returns),
        name='-1 SD', line=dict(color='rgb(0, 223, 221)', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df_returns['timestamp'], y=[sd2_upper] * len(df_returns),
        name='+2 SD', line=dict(color='rgb(255, 107, 154)', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df_returns['timestamp'], y=[sd2_lower] * len(df_returns),
        name='-2 SD', line=dict(color='rgb(255, 107, 154)', dash='dot')
    ))

    # Update layout
    fig.update_layout(
        title=f'Bitcoin {return_interval}-Day Returns',
        xaxis_title='Date',
        yaxis_title=f'{return_interval}-Day Return (%)',
        template='plotly_white',
        showlegend=True,
        width = 1400,
        height = 600
    )

    # Create Plotly figure for histogram
    fig_2 = go.Figure()

    # Add histogram of returns
    fig_2.add_trace(go.Histogram(
        x=df_returns['returns'].dropna(),  # Drop NaN values from returns
        name=f'{return_interval}-Day Returns',
        nbinsx=30,  # Number of bins for histogram
        opacity=0.7,
        marker_color='skyblue'
    ))

    # Add vertical lines for standard deviations
    fig_2.add_vline(x=sd1_upper, line_dash="dash", line_color="rgb(0, 223, 221)", annotation_text="+1SD", annotation_position="top")
    fig_2.add_vline(x=sd1_lower, line_dash="dash", line_color="rgb(0, 223, 221)", annotation_text="-1SD", annotation_position="top")
    fig_2.add_vline(x=sd2_upper, line_dash="dot", line_color="rgb(255, 107, 154)", annotation_text="+2SD", annotation_position="top")
    fig_2.add_vline(x=sd2_lower, line_dash="dot", line_color="rgb(255, 107, 154)", annotation_text="-2SD", annotation_position="top")

    # Update layout
    fig_2.update_layout(
        title=f'Histogram of Bitcoin {return_interval}-Day Returns ({start_date} to {end_date})',
        xaxis_title=f'{return_interval}-Day Return (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        showlegend=True,
        bargap=0.1,
        width = 1400,
        height = 600
    )

    return fig, fig_2
