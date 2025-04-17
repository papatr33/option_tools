import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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

# Function to fetch DVOL data from Deribit API
def get_dvol_data(currency, start_timestamp, end_timestamp, resolution):
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    params = {
        "currency": currency,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "resolution": resolution
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("result"):
            df = pd.DataFrame(
                data["result"]["data"],
                columns=["timestamp", "open", "high", "low", "close"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["dvol"] = df["close"] / 100  # Convert percentage to decimal
            return df[["timestamp", "dvol"]]
        else:
            raise Exception(f"No data returned from Deribit API for {currency}")
    else:
        raise Exception(f"Deribit API error: {response.status_code}")

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

# Main function for non-shifted volatility chart
def create_implied_VRP_chart(start_date, end_date, asset="BTC"):
    if asset not in ["BTC", "ETH"]:
        raise ValueError("Asset must be 'BTC' or 'ETH'")
    
    # Map asset to currency and symbol
    asset_info = {
        "BTC": {"currency": "BTC", "symbol": "BTCUSDT", "name": "Bitcoin"},
        "ETH": {"currency": "ETH", "symbol": "ETHUSDT", "name": "Ethereum"}
    }[asset]

    # Convert dates to timestamps (milliseconds)
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)

    # Fetch DVOL data
    dvol_df = get_dvol_data(asset_info["currency"], start_timestamp, end_timestamp, "86400")
    dvol_df.set_index("timestamp", inplace=True)

    # Fetch Binance data
    binance_df = get_binance_data(asset_info["symbol"], start_date, end_date)
    binance_df.set_index("timestamp", inplace=True)

    # Calculate Garman-Klass volatility
    binance_df["gk_vol"] = garman_klass_volatility(binance_df, window=30, trading_periods=365)

    # Merge dataframes on common dates
    merged_df = pd.merge(
        dvol_df,
        binance_df[["gk_vol"]],
        left_index=True,
        right_index=True,
        how="inner"
    )

    # Calculate IV-RV spread
    merged_df["spread"] = merged_df["dvol"] - merged_df["gk_vol"]

    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{asset_info['name']} 30-Day Implied vs. Realized Volatility",
            f"{asset_info['name']} IV-RV Spread"
        ),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Add DVOL trace (IV)
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df["dvol"],
            name="30D IV (DVOL)",
            line=dict(color="rgb(213, 213, 213)")
        ),
        row=1,
        col=1
    )

    # Add Garman-Klass trace (RV)
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df["gk_vol"],
            name="30D RV (GK)",
            line=dict(color="rgb(142, 141, 140)")
        ),
        row=1,
        col=1
    )

    # Add filled area between IV and RV
    fig.add_trace(
        go.Scatter(
            x=merged_df.index.tolist() + merged_df.index[::-1].tolist(),
            y=merged_df["dvol"].tolist() + merged_df["gk_vol"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(192, 192, 192, 0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1,
        col=1
    )

    # Add spread trace
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df["spread"],
            name="IV-RV Spread",
            line=dict(color="rgb(71, 217, 174)")
        ),
        row=2,
        col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{asset_info['name']} 30D IV v.s. RV",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True
    )

    # Update axes
    fig.update_yaxes(title_text="Volatility", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Spread (IV-RV)", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig

# Main function for shifted RV chart
def create_realized_VRP_chart(start_date, end_date, asset="BTC"):
    if asset not in ["BTC", "ETH"]:
        raise ValueError("Asset must be 'BTC' or 'ETH'")
    
    # Map asset to currency and symbol
    asset_info = {
        "BTC": {"currency": "BTC", "symbol": "BTCUSDT", "name": "Bitcoin"},
        "ETH": {"currency": "ETH", "symbol": "ETHUSDT", "name": "Ethereum"}
    }[asset]

    # Convert dates to timestamps (milliseconds)
    start_timestamp = int(start_date.timestamp() * 1000)
    extended_end_date = end_date + timedelta(days=30)
    end_timestamp = int(extended_end_date.timestamp() * 1000)

    # Fetch DVOL data
    dvol_df = get_dvol_data(asset_info["currency"], start_timestamp, end_timestamp, "86400")
    dvol_df.set_index("timestamp", inplace=True)

    # Fetch Binance data (extended period)
    binance_df = get_binance_data(asset_info["symbol"], start_date, extended_end_date)
    binance_df.set_index("timestamp", inplace=True)

    # Calculate Garman-Klass volatility
    binance_df["gk_vol"] = garman_klass_volatility(binance_df, window=30, trading_periods=365)

    # Shift RV data by 30 days
    shifted_rv = binance_df[["gk_vol"]].shift(-30, freq="D")

    # Merge dataframes
    merged_df = pd.merge(
        dvol_df,
        shifted_rv,
        left_index=True,
        right_index=True,
        how="inner"
    )

    # Trim to original date range
    merged_df = merged_df.loc[start_date:end_date]

    # Calculate IV-RV spread
    merged_df["spread"] = merged_df["dvol"] - merged_df["gk_vol"]

    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{asset_info['name']} 30-Day Implied vs. Future Realized Volatility",
            f"{asset_info['name']} IV-Future RV Spread"
        ),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Add DVOL trace (IV)
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df["dvol"],
            name="30D IV (DVOL)",
            line=dict(color="rgb(213, 213, 213)")
        ),
        row=1,
        col=1
    )

    # Add Garman-Klass trace (shifted RV)
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df["gk_vol"],
            name="30D Future RV (GK)",
            line=dict(color="rgb(142, 141, 140)")
        ),
        row=1,
        col=1
    )

    # Add filled area between IV and RV
    fig.add_trace(
        go.Scatter(
            x=merged_df.index.tolist() + merged_df.index[::-1].tolist(),
            y=merged_df["dvol"].tolist() + merged_df["gk_vol"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(192, 192, 192, 0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1,
        col=1
    )

    # Add spread trace
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df["spread"],
            name="IV-Future RV Spread",
            line=dict(color="rgb(0, 213, 255)")
        ),
        row=2,
        col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{asset_info['name']} 30D IV v.s. forward RV",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True
    )

    # Update axes
    fig.update_yaxes(title_text="Volatility", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Spread (IV-Future RV)", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig

# Function to calculate R² score
def calculate_r2(y_true, y_pred):
    """Calculate R² score for regression fit."""
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - ss_residual / ss_total if ss_total != 0 else 0

def create_iv_rv_scatter_plot(start_date, end_date, asset="BTC" ):
    # Define date range
    # extended_end_date = end_date + timedelta(days=30)  # For forward RV
    extended_end_date = end_date
    if asset not in ["BTC", "ETH"]:
        raise ValueError("Asset must be 'BTC' or 'ETH'")
    
    # Map asset to currency and symbol
    asset_info = {
        "BTC": {"currency": "BTC", "symbol": "BTCUSDT", "name": "Bitcoin"},
        "ETH": {"currency": "ETH", "symbol": "ETHUSDT", "name": "Ethereum"}
    }[asset]

    # Fetch DVOL data (IV)
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(extended_end_date.timestamp() * 1000)

    # Fetch DVOL data
    dvol_df = get_dvol_data(asset_info["currency"], start_timestamp, end_timestamp, "86400")
    dvol_df.set_index("timestamp", inplace=True)

    # Fetch Binance data
    binance_df = get_binance_data(asset_info["symbol"], start_date, end_date)
    binance_df.set_index("timestamp", inplace=True)

    # Calculate Garman-Klass volatility
    binance_df["rv"] = garman_klass_volatility(binance_df, window=30, trading_periods=365)

    shifted_rv = binance_df[["rv"]].shift(-30, freq="D")
    
    # Merge dataframes
    merged_df = pd.merge(
        dvol_df,
        shifted_rv,
        left_index=True,
        right_index=True,
        how="inner"
    )
    
    # Trim to original date range
    merged_df = merged_df.loc[start_date:end_date]
    
    # Calculate IV-RV spread
    merged_df["spread"] = merged_df["dvol"] * 100 - merged_df["rv"] * 100  # IV to %, RV already in %
    
    # Extract year for coloring
    merged_df["year"] = merged_df.index.year
    
    # Create scatter plot
    fig = go.Figure()
    
    # Define colors for each year
    colors = {
        2022: "rgb(31, 119, 180)",    # Blue
        2023: "rgb(255, 127, 14)",    # Orange
        2024: "rgb(44, 160, 44)",     # Green
        2025: "rgb(214, 39, 40)"      # Red
    }
    
    # Add scatter traces for each year
    for year in sorted(merged_df["year"].unique()):
        year_data = merged_df[merged_df["year"] == year]
        fig.add_trace(
            go.Scatter(
                x=year_data["dvol"] * 100,  # Convert IV to percentage
                y=year_data["spread"],
                mode="markers",
                name=str(year),
                marker=dict(
                    size=8,
                    color=colors.get(year, "rgb(128, 128, 128)"),
                    opacity=0.6
                ),
                text=year_data.index.strftime("%Y-%m-%d"),
                hovertemplate="Date: %{text}<br>IV: %{x:.1f}%<br>IV-RV Spread: %{y:.1f}%<extra></extra>"
            )
        )
    
    #-------------------------------------------------------------------#

    # Calculate trend line using polynomial regression (degree 2)
    x = merged_df["dvol"] * 100  # IV in percentage
    y = merged_df["spread"]
    # Remove NaN values for regression
    valid = ~(x.isna() | y.isna())
    x_valid = x[valid]
    y_valid = y[valid]
    if len(x_valid) > 2:  # Ensure enough data points for regression
        # Fit linear regression (degree 1)
        linear_coeffs = np.polyfit(x_valid, y_valid, 1)
        linear_pred = np.polyval(linear_coeffs, x_valid)
        linear_r2 = calculate_r2(y_valid, linear_pred)
        
        # Fit quadratic regression (degree 2)
        quad_coeffs = np.polyfit(x_valid, y_valid, 2)
        # Generate smooth trend line points
        x_trend = np.linspace(x_valid.min(), x_valid.max(), 100)
        y_trend = np.polyval(quad_coeffs, x_trend)
        quad_pred = np.polyval(quad_coeffs, x_valid)
        quad_r2 = calculate_r2(y_valid, quad_pred)
        
        # Print R² scores for comparison
        print(f"{asset_info['name']} VRP Trend Line Fit:")
        print(f"Linear R²: {linear_r2:.4f}")
        print(f"Quadratic R²: {quad_r2:.4f}")
        
        # Add quadratic trend line trace
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name="Quadratic Trend",
                line=dict(color="rgb(71, 217, 174)", dash="dash", width = 3),
                hoverinfo="skip"
            )
        )
    

    #-------------------------------------------------------------------#

    # Update layout
    fig.update_layout(
        title="BTC 30D Realized VRP",
        xaxis_title="30-Day IV (DVOL, %)",
        yaxis_title="Realized VRP (%)",
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
        height=800,
        width=800
    )
    
    # Update axes
    fig.update_xaxes(tickformat=".0f", title_standoff=10)
    fig.update_yaxes(tickformat=".0f", title_standoff=10)
    
    
    return fig
