import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pytz import timezone
import math

# Function to clean OHLC data by dropping rows with outliers
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

# Function to calculate Garman-Klass volatility for hourly data
def garman_klass_volatility(price_data, window_hours=14*24, trading_periods=12*365, clean=True):
    """
    Calculates Garman-Klass volatility for hourly data.
    window_hours: Number of hours in the rolling window (e.g., 14*24 for 14 days).
    trading_periods: Number of trading hours per year for annualization (12*365 for one session).
    """
    log_hl = (price_data["high"] / price_data["low"]).apply(np.log)
    log_co = (price_data["close"] / price_data["open"]).apply(np.log)
    rs = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2
    def f(v):
        return (trading_periods * v.mean()) ** 0.5
    result = rs.rolling(window=window_hours, center=False).apply(func=f)
    if clean:
        return result.dropna()
    return result

# Function to fetch Binance hourly OHLC data
def get_binance_hourly_data(symbol, start_date, end_date):
    """
    Fetches hourly OHLC data from Binance API for given symbol and date range.
    Handles pagination for large date ranges and includes detailed error handling.
    """
    # Convert date objects to datetime if necessary
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())
    
    # Ensure timezone-naive datetime
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)
    
    url = "https://api.binance.com/api/v3/klines"  # Switched to global endpoint
    all_data = []
    current_start = start_date
    
    while current_start < end_date:
        start_ts = int(current_start.timestamp() * 1000)
        end_ts = min(int((current_start + timedelta(days=41)).timestamp() * 1000), 
                     int(end_date.timestamp() * 1000))
        params = {
            "symbol": symbol.upper(),  # Ensure uppercase symbol
            "interval": "1h",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    break
                all_data.extend(data)
                # Update current_start to the last timestamp + 1 hour
                last_timestamp = int(data[-1][0])
                current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(hours=1)
            else:
                error_msg = response.json().get('msg', 'No error message provided')
                raise Exception(f"Binance API error {response.status_code}: {error_msg}")
        except requests.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(
        all_data,
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

# Function to calculate session-based cumulative returns and volatility
def calculate_session_returns_and_volatility(symbol, start_date, end_date):
    """
    Calculates cumulative returns and 14-day Garman-Klass volatility for Asia (8AM-8PM)
    and US (8PM-8AM) sessions in UTC+8. Returns a DataFrame with returns and volatility.
    """
    try:
        # Fetch hourly data
        df = get_binance_hourly_data(symbol, start_date, end_date)
        if df.empty:
            return pd.DataFrame(columns=['date', 'Asia_Cumulative', 'US_Cumulative', 
                                       'Asia_Volatility', 'US_Volatility'])
        
        # Convert timestamps to UTC+8 (Hong Kong timezone)
        hk_tz = timezone('Asia/Hong_Kong')
        df['timestamp_hk'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(hk_tz)
        
        # Assign sessions based on hour in UTC+8
        df['hour'] = df['timestamp_hk'].dt.hour
        df['session'] = np.where(
            (df['hour'] >= 8) & (df['hour'] < 20),
            'Asia',
            'US'
        )
        
        # Calculate hourly returns
        df['return'] = df['close'].pct_change()
        
        # Split data by session
        asia_df = df[df['session'] == 'Asia'].copy()
        us_df = df[df['session'] == 'US'].copy()
        
        # Calculate Garman-Klass volatility for each session (14-day window, 12 hours/day)
        if not asia_df.empty:
            asia_df['volatility'] = garman_klass_volatility(
                asia_df, 
                window_hours=14*12,  # 14 days * 12 hours
                trading_periods=12*365,  # 12 hours per day * 365 days
                clean=True
            )
        else:
            asia_df['volatility'] = np.nan
        
        if not us_df.empty:
            us_df['volatility'] = garman_klass_volatility(
                us_df, 
                window_hours=14*12,  # 14 days * 12 hours
                trading_periods=12*365,  # 12 hours per day * 365 days
                clean=True
            )
        else:
            us_df['volatility'] = np.nan
        
        # Group by session and date for returns
        df['date'] = df['timestamp_hk'].dt.date
        session_returns = df.groupby(['date', 'session'])['return'].apply(
            lambda x: (1 + x).prod() - 1
        ).unstack().fillna(0)
        
        # Calculate cumulative returns for each session
        session_returns['Asia_Cumulative'] = (1 + session_returns.get('Asia', 0)).cumprod() - 1
        session_returns['US_Cumulative'] = (1 + session_returns.get('US', 0)).cumprod() - 1
        
        # Group by date for volatility
        asia_vol = asia_df.groupby(asia_df['timestamp_hk'].dt.date)['volatility'].last()
        us_vol = us_df.groupby(us_df['timestamp_hk'].dt.date)['volatility'].last()
        
        # Combine results
        result = session_returns.reset_index()
        result = result.merge(asia_vol.rename('Asia_Volatility'), 
                           left_on='date', right_index=True, how='left')
        result = result.merge(us_vol.rename('US_Volatility'), 
                           left_on='date', right_index=True, how='left')
        
        return result[['date', 'Asia_Cumulative', 'US_Cumulative', 'Asia_Volatility', 'US_Volatility']]
    except Exception as e:
        raise Exception(f"Error calculating session returns and volatility: {str(e)}")

