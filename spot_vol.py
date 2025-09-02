import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
import plotly.express as px

def clean_ohlc_data(df):
    """
    Cleans OHLC data by dropping rows where any OHLC value is over 2x or under 0.5x
    the median of adjacent points (previous and next hour).
    """
    df_clean = df.copy()
    columns_to_check = ["open", "high", "low", "close"]
    
    # Initialize a mask for rows to keep (True means keep, False means drop)
    keep_rows = pd.Series(True, index=df_clean.index)
    
    for col in columns_to_check:
        # Calculate median of previous and next hour for each point
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
    
    return df_clean

def fetch_binance_btc_spot_price(start_date, end_date):
    """
    Fetch hourly BTC spot price (OHLC) from Binance API for a given date range, handling pagination.
    Returns a DataFrame with daily closing prices at 00:00 UTC.
    """
    url = "https://api.binance.us/api/v3/klines"
    chunk_days = 30  # Each chunk covers 30 days (~720 hours, within limit=1000)
    all_data = []
    
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        start_ts = int(current_start.timestamp() * 1000)
        end_ts = int(current_end.timestamp() * 1000)
        
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"No Binance data for {current_start.date()} to {current_end.date()}")
                current_start = current_end
                continue
            
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ]
            )
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
            
            # Clean the OHLC data
            df = clean_ohlc_data(df)
            
            all_data.append(df)
            
        except requests.RequestException as e:
            print(f"Error fetching Binance BTC price data for {current_start.date()} to {current_end.date()}: {e}")
        
        current_start = current_end
    
    if not all_data:
        print("No Binance data retrieved")
        return pd.DataFrame()
    
    # Consolidate data
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp")
    
    # Filter for 00:00 UTC each day
    df = df[df["timestamp"].dt.hour == 0]
    
    # Convert timestamp to date for merging
    df["date"] = df["timestamp"].dt.date
    
    return df[["date", "close"]].rename(columns={"close": "price"})

def fetch_deribit_dvol_index(start_date, end_date):
    """
    Fetch hourly DVOL index data from Deribit API for a given date range, handling pagination.
    Returns a DataFrame with daily DVOL values at 00:00 UTC.
    """
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    chunk_days = 30  # Each chunk covers 30 days (~720 hours)
    all_data = []
    
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        start_timestamp = int(current_start.timestamp() * 1000)
        end_timestamp = int(current_end.timestamp() * 1000)
        
        params = {
            "currency": "BTC",
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "resolution": "3600"  # 3600 seconds = 1 hour
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'result' not in data or 'data' not in data['result']:
                print(f"No DVOL data for {current_start.date()} to {current_end.date()}")
                current_start = current_end
                continue
            
            chunk_data = []
            for entry in data['result']['data']:
                timestamp_ms = entry[0]
                dvol_value = entry[1]  # DVOL is in percentage (e.g., 55.0 for 55%)
                dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
                # Only include 00:00 UTC entries
                if dt.hour == 0:
                    chunk_data.append({
                        'timestamp': timestamp_ms,
                        'date': dt.date(),
                        'dvol': dvol_value
                    })
            
            all_data.append(pd.DataFrame(chunk_data))
            
        except requests.RequestException as e:
            print(f"Error fetching DVOL data for {current_start.date()} to {current_end.date()}: {e}")
        
        current_start = current_end
    
    if not all_data:
        print("No DVOL data retrieved")
        return pd.DataFrame()
    
    # Consolidate data
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp")
    
    return df[["date", "dvol"]]

def calculate_spot_vol_correlation(days=365):
    """
    Calculate BTC spot-vol correlation and generate Plotly figures using 00:00 UTC data.
    Args:
        days: Number of days of historical data to fetch (default: 365 for one year)
    Returns:
        Tuple of two Plotly figures: (scatter_fig, beta_fig)
    """
    # Calculate date range
    end_dt = datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)
    
    # Fetch data
    btc_df = fetch_binance_btc_spot_price(start_dt, end_dt)
    dvol_df = fetch_deribit_dvol_index(start_dt, end_dt)
    
    if btc_df.empty or dvol_df.empty:
        print("Failed to fetch data. Cannot generate plots.")
        return None, None
    
    # Merge data on date
    merged_df = pd.merge(btc_df[['date', 'price']], dvol_df[['date', 'dvol']], on='date', how='inner')
    if merged_df.empty:
        print("No overlapping data between BTC price and DVOL. Cannot generate plots.")
        return None, None
    
    # Calculate daily returns and changes
    merged_df['btc_return'] = merged_df['price'].pct_change() * 100  # In percentage
    merged_df['dvol_change'] = merged_df['dvol'].diff()  # DVOL is already in percentage
    merged_df = merged_df.dropna()
    
    if len(merged_df) < 2:
        print("Insufficient data after processing. Cannot generate plots.")
        return None, None
    
    # Scatter Plot with Regression Line
    scatter_fig = px.scatter(
        merged_df,
        x='btc_return',
        y='dvol_change',
        title='BTC Spot Vol Correlation',
        labels={'btc_return': 'BTC Daily Return (%)', 'dvol_change': 'DVOL Daily Change (%)'},
        hover_data=['date'],
        color_discrete_sequence=['rgb(128, 198, 128)']  # Light green dots
    )
    scatter_fig.update_traces(marker=dict(size=8, opacity=0.6))
    
    # Calculate linear regression
    x = merged_df['btc_return']
    y = merged_df['dvol_change']
    valid = ~(x.isna() | y.isna())
    x_valid = x[valid]
    y_valid = y[valid]
    
    if len(x_valid) > 1:  # Ensure enough data for regression
        coeffs = np.polyfit(x_valid, y_valid, 1)
        x_trend = np.array([x_valid.min(), x_valid.max()])
        y_trend = np.polyval(coeffs, x_trend)
        
        # Add regression line
        scatter_fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name='Linear Trend',
                line=dict(color='rgb(255, 138, 138)', dash='dash', width=2),  # Coral pink dashed line
                hoverinfo='skip'
            )
        )
    
    scatter_fig.update_layout(
        showlegend=True,
        plot_bgcolor='white',
        width=800,
        height=800
    )
    scatter_fig.update_xaxes(gridcolor='lightgrey')
    scatter_fig.update_yaxes(gridcolor='lightgrey')
    
    # Calculate 30-day rolling beta
    window = 30
    rolling_cov = merged_df['btc_return'].rolling(window=window).cov(merged_df['dvol_change'])
    rolling_var = merged_df['btc_return'].rolling(window=window).var()
    rolling_beta = rolling_cov / rolling_var
    
    # Rolling Beta Plot
    beta_df = pd.DataFrame({
        'date': merged_df['date'],
        'beta': rolling_beta
    }).dropna()
    
    beta_fig = go.Figure()
    beta_fig.add_trace(
        go.Scatter(
            x=beta_df['date'],
            y=beta_df['beta'],
            mode='lines',
            name='30-Day Rolling Beta',
            line=dict(color='rgb(0, 213, 255)')  # Cyan line
        )
    )
    beta_fig.update_layout(
        title='30-Day Rolling Beta of IV on BTC',
        xaxis_title='Date',
        yaxis_title='Beta',
        plot_bgcolor='white',
        showlegend=True
    )
    beta_fig.update_xaxes(gridcolor='lightgrey', tickangle=45)
    beta_fig.update_yaxes(gridcolor='lightgrey')
    
    return scatter_fig, beta_fig
