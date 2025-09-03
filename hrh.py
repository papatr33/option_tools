import pandas as pd
import plotly.graph_objects as go
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
import numpy as np
import time

def historical_return_histogram(start_date, end_date, return_interval=1, asset='BTC'):
    # Validate inputs
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("start_date and end_date must be datetime objects")
    if not isinstance(return_interval, int) or return_interval < 1:
        raise ValueError("return_interval must be a positive integer")
    if asset not in ['BTC', 'ETH', 'BNB']:  # Add more valid assets as needed
        raise ValueError(f"Unsupported asset: {asset}")

    # Initialize Binance client with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = Client()
            client.ping()  # Test connectivity
            break
        except BinanceAPIException as e:
            if e.status_code == 429:
                print(f"Rate limit exceeded. Waiting {60 * (attempt + 1)} seconds...")
                time.sleep(60 * (attempt + 1))
            else:
                print(f"Binance API error: {e.status_code} - {e.message}")
                return None, None
        except Exception as e:
            print(f"Unexpected error during client initialization: {e}")
            return None, None
    else:
        print("Max retries reached. Unable to initialize client.")
        return None, None

    # Function to fetch historical daily data
    def get_historical_data(symbol, interval, start_date, end_date):
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        trading_pair = f"{symbol}USDT"
        try:
            klines = client.get_historical_klines(trading_pair, interval, start_date_str, end_date_str)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            return df[['timestamp', 'close']]
        except BinanceAPIException as e:
            print(f"Error fetching klines: {e.status_code} - {e.message}")
            return None

    # Function to calculate n-day returns
    def calculate_interval_returns(df, interval):
        df_interval = df.iloc[::interval].copy()
        df_interval['returns'] = df_interval['close'].pct_change() * 100
        return df_interval

    # Fetch data
    df = get_historical_data(asset, Client.KLINE_INTERVAL_1DAY, start_date, end_date)
    if df is None:
        return None, None

    # Calculate returns
    df_returns = calculate_interval_returns(df, return_interval)
    if df_returns.empty:
        print("No returns data available.")
        return None, None

    # Calculate standard deviations
    mean_return = df_returns['returns'].mean()
    std_return = df_returns['returns'].std()
    sd1_upper = mean_return + std_return
    sd1_lower = mean_return - std_return
    sd2_upper = mean_return + 2 * std_return
    sd2_lower = mean_return - 2 * std_return

    # Create Plotly figure for bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_returns['timestamp'],
        y=df_returns['returns'],
        name=f'{return_interval}-Day Returns (%)',
        marker_color=['rgb(0, 147, 146)' if x >= 0 else 'rgb(208, 88, 126)' for x in df_returns['returns']]
    ))
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
    fig.update_layout(
        title=f'{asset} {return_interval}-Day Returns',
        xaxis_title='Date',
        yaxis_title=f'{return_interval}-Day Return (%)',
        template='plotly_white',
        showlegend=True,
        width=1400,
        height=600
    )

    # Create Plotly figure for histogram
    fig_2 = go.Figure()
    fig_2.add_trace(go.Histogram(
        x=df_returns['returns'].dropna(),
        name=f'{return_interval}-Day Returns',
        nbinsx=30,
        opacity=0.7,
        marker_color='skyblue'
    ))
    fig_2.add_vline(x=sd1_upper, line_dash="dash", line_color="rgb(0, 223, 221)", annotation_text="+1SD", annotation_position="top")
    fig_2.add_vline(x=sd1_lower, line_dash="dash", line_color="rgb(0, 223, 221)", annotation_text="-1SD", annotation_position="top")
    fig_2.add_vline(x=sd2_upper, line_dash="dot", line_color="rgb(255, 107, 154)", annotation_text="+2SD", annotation_position="top")
    fig_2.add_vline(x=sd2_lower, line_dash="dot", line_color="rgb(255, 107, 154)", annotation_text="-2SD", annotation_position="top")
    fig_2.update_layout(
        title=f'Histogram of {asset} {return_interval}-Day Returns ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})',
        xaxis_title=f'{return_interval}-Day Return (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        showlegend=True,
        bargap=0.1,
        width=1400,
        height=600
    )

    return fig, fig_2
