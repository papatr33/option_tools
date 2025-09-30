import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import math

# Function to clean OHLC data by dropping rows with outliers
def clean_ohlc_data(df):
    """
    Cleans OHLC data by dropping rows where any OHLC value is over 2x or under 0.5x
    the median of adjacent points (previous and next day).
    """
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
    
    return df_clean

# Function to fetch Binance OHLC data using public API
def get_binance_data(symbol, start_date, end_date):
    # Convert date objects to datetime if necessary
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())
    
    url = "https://api.binance.us/api/v3/klines"
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": 1000
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        
        # Clean the OHLC data
        df = clean_ohlc_data(df)
        
        return df[["timestamp", "open", "high", "low", "close"]]
    else:
        raise Exception(f"Binance API error: {response.status_code}")

# Function to calculate Garman-Klass volatility
def garman_klass_volatility(price_data, window=30, trading_periods=365, clean=True):
    log_hl = (price_data["high"] / price_data["low"]).apply(np.log)
    log_co = (price_data["close"] / price_data["open"]).apply(np.log)
    rs = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2
    def f(v):
        return (trading_periods * v.mean()) ** 0.5
    result = rs.rolling(window=window, center=False).apply(func=f)
    if clean:
        return result.dropna()
    return result

# Function to calculate percentile rank
def calculate_percentile_rank(series, value):
    """
    Calculate the percentile rank of a value within a series.
    Returns a value between 0 and 100.
    """
    return (sum(series <= value) / len(series)) * 100

# Function to create volatility chart and percentile table
def create_btc_volatility_chart(start_date, end_date, symbol="BTCUSDT"):
    # Convert date objects to datetime if necessary
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())
    
    # Fetch Binance data
    df = get_binance_data(symbol, start_date, end_date)
    df.set_index("timestamp", inplace=True)
    
    # Calculate Garman-Klass volatility for different windows
    windows = [
        {"days": 7, "name": "7-Day RV"},
        {"days": 14, "name": "14-Day RV"},
        {"days": 30, "name": "30-Day RV"},
        {"days": 60, "name": "60-Day RV"}
    ]
    
    for window in windows:
        df[f"gk_vol_{window['days']}"] = garman_klass_volatility(
            df, 
            window=window["days"], 
            trading_periods=365
        )
    
    # Calculate current RV values and their percentiles
    current_rvs = []
    for window in windows:
        col = f"gk_vol_{window['days']}"
        current_value = df[col].iloc[-1] if not df[col].empty else np.nan
        if not np.isnan(current_value):
            percentile = calculate_percentile_rank(df[col].dropna(), current_value)
        else:
            percentile = np.nan
        current_rvs.append({
            "Window": window["name"],
            "RV": current_value,
            "Percentile": percentile
        })
    
    # Create subplot with 2 rows: Price and Volatility
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Spot Price",
            "Garman-Klass Realized Volatility"
        ),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            name="Price",
            line=dict(color="rgb(31, 119, 180)")
        ),
        row=1,
        col=1
    )
    
    # Add volatility traces
    colors = [
        "rgb(255, 127, 14)",  # Orange for 7-day
        "rgb(44, 160, 44)",   # Green for 14-day
        "rgb(214, 39, 40)",   # Red for 30-day
        "rgb(148, 103, 189)"  # Purple for 60-day
    ]
    
    for i, window in enumerate(windows):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"gk_vol_{window['days']}"],
                name=window["name"],
                line=dict(color=colors[i])
            ),
            row=2,
            col=1
        )
    
    # Create table for current RV and percentiles
    table = go.Table(
        header=dict(
            values=["Window", "Current RV", "Percentile"],
            align="center",
            fill_color="rgb(200, 200, 200)",
            font=dict(color="black", size=12)
        ),
        cells=dict(
            values=[
                [rv["Window"] for rv in current_rvs],
                [f"{rv['RV']*100:.2f}%" if not np.isnan(rv['RV']) else "N/A" for rv in current_rvs],
                [f"{rv['Percentile']:.2f}%" if not np.isnan(rv['Percentile']) else "N/A" for rv in current_rvs]
            ],
            align="center",
            fill_color="rgb(235, 235, 235)",
            font=dict(color="black", size=11)
        )
    )
    
    # Create a separate figure for the table
    table_fig = go.Figure(data=[table])
    table_fig.update_layout(
        title_text="Current Realized Volatility Percentiles",
        height=200,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Update layout for main chart
    fig.update_layout(
        height=800,
        title_text="Price and RV",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig, table_fig    

def rv_table():
    assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'HYPE']
    periods = [7, 14, 30, 60]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    data = []
    for asset in assets:
        symbol = f"{asset}USDT"

        df = get_binance_hourly_data(symbol, start_date, end_date)
            
        # Get spot price (latest close)
        spot = df['close'].iloc[-1]
        
        row = {'Asset': asset, 'Spot Price': spot}
        for p in periods:
            # Current volatility
            vol = garman_klass_volatility(df, window=p, trading_periods=365, clean=True)
            latest_vol = vol.iloc[-1] * 100 if not vol.empty else None  # Convert to percentage
            
            # Volatility from one week ago
            one_week_ago = df.index[-1] - timedelta(days=7)
            past_df = df[df.index <= one_week_ago]
            if len(past_df) >= p:
                past_vol = garman_klass_volatility(past_df, window=p, trading_periods=365, clean=True)
                past_vol_value = past_vol.iloc[-1] * 100 if not past_vol.empty else None
            else:
                past_vol_value = None
                
            row[f'{p}d Vol'] = latest_vol
            row[f'{p}d Vol 1w Ago'] = past_vol_value
        
        data.append(row)

    if data:
        df_display = pd.DataFrame(data)
        df_display = df_display.round(2)

    return df_display

    

