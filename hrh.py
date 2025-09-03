import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime
import numpy as np
import time

# Function to clean OHLC data by dropping rows with outliers
def clean_ohlc_data(df):
    """
    Cleans OHLC data by dropping rows where any OHLC value is over 2x or under 0.5x
    the median of adjacent points (previous and next day).
    """
    if df.empty:
        return df
    df_clean = df.copy()
    columns_to_check = ["open", "high", "low", "close"]
    
    # Initialize a mask for rows to keep (True means keep, False means drop)
    keep_rows = pd.Series(True, index=df_clean.index)
    
    for col in columns_to_check:
        # Calculate median of previous and next day for each point
        prev_values = df_clean[col].shift(1)
        next_values = df_clean[col].shift(-1)
        median_adjacent = pd.concat([prev_values, next_values], axis=1).median(axis=1)
        
        # Identify outliers: >2x or <0.5x the median of adjacent points
        ratio = df_clean[col] / median_adjacent
        outliers = (ratio > 2) | (ratio < 0.5)
        
        # Update mask: mark rows with outliers to be dropped
        keep_rows = keep_rows & ~outliers
    
    # Drop rows where any column has an outlier
    df_clean = df_clean[keep_rows]
    print(f"Cleaned data: {len(df)} rows reduced to {len(df_clean)} rows")
    return df_clean

# Function to fetch Binance OHLC data using public API
def get_historical_data(symbol, interval, start_date, end_date):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_date
    chunk_size = pd.Timedelta(days=90)  # Fetch 90 days at a time
    
    while current_start < end_date:
        chunk_end = min(current_start + chunk_size, end_date)
        start_ts = int(current_start.timestamp() * 1000)
        end_ts = int(chunk_end.timestamp() * 1000)
        params = {
            "symbol": f"{symbol}USDT",
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    print(f"No data returned for {symbol}USDT from {current_start} to {chunk_end}")
                else:
                    all_data.extend(data)
                current_start = chunk_end + pd.Timedelta(days=1)
                time.sleep(1)  # Avoid rate limits
            else:
                print(f"Binance API error: {response.status_code} for {symbol}USDT")
                return pd.DataFrame()
        except requests.RequestException as e:
            print(f"Error fetching data for {symbol}USDT: {e}")
            return pd.DataFrame()
    
    if not all_data:
        print(f"No data returned for {symbol}USDT from {start_date} to {end_date}")
        return pd.DataFrame()
    
    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    
    # Clean the OHLC data (optional: comment out to disable)
    df = clean_ohlc_data(df)
    
    return df[["timestamp", "close"]]

def historical_return_histogram(start_date, end_date, return_interval=1, asset='BTC'):
    # Validate inputs
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("start_date and end_date must be datetime objects")
    if not isinstance(return_interval, int) or return_interval < 1:
        raise ValueError("return_interval must be a positive integer")
    if asset not in ['BTC', 'ETH', 'BNB']:  # Add more valid assets as needed
        raise ValueError(f"Unsupported asset: {asset}")

    # Fetch Bitcoin daily data
    df = get_historical_data(asset, "1d", start_date, end_date)
    
    # Check if data is empty
    if df.empty:
        print("No valid data available after fetching and cleaning.")
        return None, None

    # Function to calculate n-day returns
    def calculate_interval_returns(df, interval):
        if df.empty or len(df) < 2:
            print("Insufficient data points to calculate returns (need at least 2).")
            return pd.DataFrame()
        df_interval = df.iloc[::interval].copy()
        df_interval['returns'] = df_interval['close'].pct_change() * 100  # Returns in percentage
        if df_interval['returns'].dropna().empty:
            print("All returns are NaN after calculation.")
            return pd.DataFrame()
        return df_interval

    # Calculate returns for the specified interval
    df_returns = calculate_interval_returns(df, return_interval)
    
    # Check if returns data is valid
    if df_returns.empty:
        print("No valid returns data available.")
        return None, None

    # Calculate standard deviations
    mean_return = df_returns['returns'].mean()
    std_return = df_returns['returns'].std()
    if np.isnan(mean_return) or np.isnan(std_return):
        print("Invalid statistical data (NaN in mean or std).")
        return None, None
    sd1_upper = mean_return + std_return
    sd1_lower = mean_return - std_return
    sd2_upper = mean_return + 2 * std_return
    sd2_lower = mean_return - 2 * std_return

    # Create Plotly figure for bar chart
    try:
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
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        return None, None

    # Create Plotly figure for histogram
    try:
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
    except Exception as e:
        print(f"Error creating histogram: {e}")
        return None, None

    return fig, fig_2
