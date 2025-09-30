import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import math

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

    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_date

    while current_start < end_date:
        start_ts = int(current_start.timestamp() * 1000)
        # Binance API limit is 1000 candles per request
        end_ts = min(int((current_start + timedelta(days=41)).timestamp() * 1000), 
                     int(end_date.timestamp() * 1000))
        params = {
            "symbol": symbol.upper(),
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
                last_timestamp = int(data[-1][0])
                current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(hours=1)
            else:
                error_msg = response.json().get('msg', 'No error message provided')
                raise Exception(f"Binance API error {response.status_code} for {symbol}: {error_msg}")
        except requests.RequestException as e:
            raise Exception(f"Network error for {symbol}: {str(e)}")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    
    # Clean the OHLC data
    df = clean_ohlc_data(df)
    
    return df[["timestamp", "open", "high", "low", "close"]]

def garman_klass_volatility(price_data, window_hours=14*24, trading_periods=24*365, clean=True):
    """
    Calculates Garman-Klass volatility for hourly data.
    window_hours: Number of hours in the rolling window (e.g., 14*24 for 14 days).
    trading_periods: Number of trading hours per year for annualization (24*365 for one session).
    """
    log_hl = (price_data["high"] / price_data["low"]).apply(np.log)
    log_co = (price_data["close"] / price_data["open"]).apply(np.log)
    rs = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2

    def f(v):
        return (trading_periods * v.mean()) ** 0.5

    result = rs.rolling(window=window_hours, center=False).apply(func=f, raw=False)
    
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
    df = get_binance_hourly_data(symbol, start_date, end_date)
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
            window_hours=window["days"]*24, 
            trading_periods=365*24
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
    # Define the list of crypto symbols
    symbols = ["BTC", "ETH", "SOL", "BNB", "HYPE", "XRP", "DOGE"]
    
    # Define volatility windows in days
    vol_windows = [7, 14, 30, 60]

    # Set the date range for data fetching
    end_date = datetime.now()
    start_date = end_date - timedelta(days=68) 
    
    all_results = []
    
    print("Fetching data for RV table...")

    for symbol in symbols:
        try:
            pair_symbol = f"{symbol}USDT"
            price_df = get_binance_hourly_data(pair_symbol, start_date, end_date)

            if price_df.empty:
                print(f"Could not retrieve data for {symbol}.")
                continue
                
            price_df = price_df.set_index("timestamp")
            spot_price = price_df["close"].iloc[-1]
            result_row = {"Symbol": symbol, "Spot Price": spot_price}
            
            for days in vol_windows:
                window_hours = days * 24
                volatility_series = garman_klass_volatility(price_df, window_hours=window_hours)
                
                if not volatility_series.empty:
                    current_vol = volatility_series.iloc[-1]
                    try:
                        vol_1w_ago = volatility_series.iloc[-(24*7)]
                    except IndexError:
                        vol_1w_ago = np.nan
                        
                    result_row[f"{days}d Vol"] = current_vol
                    result_row[f"{days}d Vol (1w ago)"] = vol_1w_ago
                else:
                    result_row[f"{days}d Vol"] = np.nan
                    result_row[f"{days}d Vol (1w ago)"] = np.nan
            
            all_results.append(result_row)
            print(f"Successfully processed {symbol} for table.")

        except Exception as e:
            print(f"An error occurred for {symbol}: {e}")

    # Return the DataFrame with raw numerical data for styling in Streamlit
    results_df = pd.DataFrame(all_results)
    return results_df


def create_rv_range_plot(symbols=["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]):
    """
    Generates a plot showing the 7-day RV's current value and its
    high/low range over the past 20 and 60 days for a list of cryptocurrencies.
    
    Args:
        symbols (list): A list of cryptocurrency tickers to plot (e.g., ["BTC", "ETH"]).
        
    Returns:
        plotly.graph_objects.Figure: The generated plot.
    """
    plot_data = []
    
    # We need 60 days of history + 7 days for the initial RV calculation window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=67)

    for symbol in symbols:
        try:
            pair_symbol = f"{symbol.upper()}USDT"
            
            # Fetch hourly data
            price_df = get_binance_hourly_data(pair_symbol, start_date, end_date)
            if price_df.empty:
                print(f"No data for {symbol}, skipping.")
                continue
            price_df = price_df.set_index("timestamp")

            # Calculate 7-day rolling RV
            rv_7d = garman_klass_volatility(price_df, window_hours=7*24)
            if rv_7d.empty:
                print(f"Could not calculate RV for {symbol}, skipping.")
                continue

            if len(rv_7d) < 60 * 24:
                print(f"Not enough data for 60-day lookback for {symbol}, skipping.")
                continue

            current_rv = rv_7d.iloc[-1]
            rv_last_20d = rv_7d.tail(20 * 24)
            high_20d, low_20d = rv_last_20d.max(), rv_last_20d.min()
            rv_last_60d = rv_7d.tail(60 * 24)
            high_60d, low_60d = rv_last_60d.max(), rv_last_60d.min()

            plot_data.append({
                "symbol": symbol, "current": current_rv * 100,
                "high_20d": high_20d * 100, "low_20d": low_20d * 100,
                "high_60d": high_60d * 100, "low_60d": low_60d * 100,
            })

        except Exception as e:
            print(f"An error occurred while processing {symbol}: {e}")

    if not plot_data:
        return go.Figure().update_layout(title_text="Not enough data to generate the plot for the selected symbols.")

    df = pd.DataFrame(plot_data)
    fig = go.Figure()

    # New Color Scheme
    color_60d = '#d6eaf8' # Light Blue
    color_20d = '#85c1e9' # Medium Blue
    color_current = '#CB1B45' # Vibrant Orange

    # 60-Day Range
    fig.add_trace(go.Bar(
        name='60-Day H/L', x=df['symbol'], y=df['high_60d'] - df['low_60d'],
        base=df['low_60d'], marker_color=color_60d, width=0.6, hoverinfo='none'
    ))

    # 20-Day Range
    fig.add_trace(go.Bar(
        name='20-Day H/L', x=df['symbol'], y=df['high_20d'] - df['low_20d'],
        base=df['low_20d'], marker_color=color_20d, width=0.3, hoverinfo='none'
    ))

    # Current Value
    fig.add_trace(go.Scatter(
        name='Current', x=df['symbol'], y=df['current'], mode='markers',
        marker=dict(symbol='x-thin', color=color_current, size=14, line=dict(width=3)),
        hovertemplate='<b>%{x}</b><br>Current RV: %{y:.1f}%<extra></extra>'
    ))

    # Add annotations with bigger fonts
    for _, row in df.iterrows():
        fig.add_annotation(x=row['symbol'], y=row['high_60d'], text=f"{row['high_60d']:.1f}",
                           showarrow=False, yshift=10, font=dict(size=13, color='#566573'))
        fig.add_annotation(x=row['symbol'], y=row['low_60d'], text=f"{row['low_60d']:.1f}",
                           showarrow=False, yshift=-10, font=dict(size=13, color='#566573'))
        fig.add_annotation(x=row['symbol'], y=row['current'], text=f"<b>{row['current']:.1f}</b>",
                           showarrow=False, xshift=30, font=dict(size=14, color=color_current))

    # Update layout with bigger fonts
    fig.update_layout(
        title=dict(text='<b>7-Day Realized Volatility: Current vs. Historical Ranges</b>', font=dict(size=22)),
        xaxis_title='Asset', yaxis_title='7-Day Annualized RV (%)',
        barmode='overlay', template='plotly_white', height=650,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
