import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
from datetime import datetime, date, timedelta
import time
import yfinance as yf

def plot_btc_altcoin_correlations(start_date, end_date, max_lookback=100):
    """
    Generate two heatmaps showing BTC correlations with altcoins for a Streamlit app.
    
    Parameters:
    - start_date (str or datetime.date): Start date in 'YYYY-MM-DD' format or date object.
    - end_date (str or datetime.date): End date in 'YYYY-MM-DD' format or date object.
    - max_lookback (int): Maximum lookback window in days (default: 100).
    
    Returns:
    - tuple: (fig1, fig2), Plotly figures for the two heatmaps.
    """
    # Convert date objects to strings if necessary
    if isinstance(start_date, date):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, date):
        end_date = end_date.strftime('%Y-%m-%d')

    # Initialize Binance.US client
    client = Client(tld='us')  # Use 'us' TLD for Binance.US

    # Define symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 
               'NEARUSDT', 'BNBUSDT', 'DOGEUSDT', 'CRVUSDT', 'BCHUSDT', 'LTCUSDT']

    # Calculate fetch start date (200 days prior to start_date)
    fetch_start_date = (pd.to_datetime(start_date) - timedelta(days=200)).strftime('%Y-%m-%d')

    # Effective start date for display
    effective_start = start_date

    # Define lookback periods for second heatmap (10 to max_lookback, step 10)
    lookback_periods = list(range(10, max_lookback + 1, 10))

    # Function to fetch daily close prices
    def get_daily_closes(symbols, start_date, end_date):
        data = {}
        for symbol in symbols:
            try:
                klines = client.get_historical_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1DAY,
                    start_str=start_date,
                    end_str=end_date
                )
                if not klines:
                    print(f"No data returned for {symbol}")
                    continue
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore'
                ])
                df['date'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('date', inplace=True)
                df['close'] = df['close'].astype(float)
                data[symbol.replace('USDT', '')] = df['close']
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
            time.sleep(0.5)  # Avoid rate limits
        return pd.DataFrame(data)

    # Fetch data
    data = get_daily_closes(symbols, fetch_start_date, end_date)

    if data.empty:
        raise ValueError("No valid data downloaded for any ticker. Check ticker availability on Binance.US.")

    # Verify BTC is in the data
    if 'BTC' not in data.columns:
        raise ValueError("BTCUSDT data not available")

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    # --- First Heatmap: BTC Correlation with Each Altcoin ---
    correlation_matrix = pd.DataFrame(index=daily_returns.index, 
                                     columns=[t for t in data.columns if t != 'BTC'])

    for altcoin in [t for t in data.columns if t != 'BTC']:
        correlation_matrix[altcoin] = daily_returns['BTC'].rolling(window=max_lookback).corr(daily_returns[altcoin])

    # Filter for dates after effective_start, keeping non-NaN values
    correlation_matrix = correlation_matrix.loc[effective_start:].dropna()
    plot_data = data.loc[effective_start:]

    # Prepare data for first heatmap
    dates = correlation_matrix.index
    altcoin_tickers = [t for t in data.columns if t != 'BTC']
    z_data = correlation_matrix[altcoin_tickers].T.values

    # Create first subplot
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    # Add heatmap
    fig1.add_trace(
        go.Heatmap(
            x=dates,
            y=altcoin_tickers,
            z=z_data,
            colorscale='RdBu_r',
            zmin=0,
            zmax=1,
            zsmooth='best',
            colorbar=dict(
                title='Correlation',
                len=0.75,
                thickness=25,
                y=0.5,
                x=1.02,
            ),
        ),
        secondary_y=False,
    )

    # Add BTC price line
    fig1.add_trace(
        go.Scatter(
            x=plot_data.index,
            y=plot_data['BTC'],
            name='BTC Price',
            line=dict(color='black', width=3),
            hovertemplate='Date: %{x}<br>BTC Price: $%{y:.2f}<extra></extra>',
        ),
        secondary_y=True,
    )

    # Update layout
    fig1.update_layout(
        title=f'BTC Correlation with Altcoins ({effective_start} to {end_date})',
        xaxis_title='Date',
        yaxis_title='Altcoins',
        height=800,
        width=1200,
        showlegend=True,
    )

    # Update axes
    fig1.update_yaxes(title_text='Altcoins', secondary_y=False)
    fig1.update_yaxes(title_text='BTC Price ($)', secondary_y=True)

    # --- Second Heatmap: Average BTC Correlation Across Lookback Periods ---
    avg_correlation_matrix = pd.DataFrame(index=daily_returns.index, columns=lookback_periods)

    for lookback in lookback_periods:
        correlations = pd.DataFrame(index=daily_returns.index)
        for altcoin in [t for t in data.columns if t != 'BTC']:
            correlations[altcoin] = daily_returns['BTC'].rolling(window=lookback).corr(daily_returns[altcoin])
        avg_correlation_matrix[lookback] = correlations.mean(axis=1)

    # Filter for dates after effective_start, keeping non-NaN values
    avg_correlation_matrix = avg_correlation_matrix.loc[effective_start:].dropna()
    plot_data = data.loc[effective_start:]

    # Prepare data for second heatmap
    dates = avg_correlation_matrix.index
    lookback_labels = [str(lb) for lb in lookback_periods]
    z_data = avg_correlation_matrix[lookback_periods].T.values

    # Create second subplot
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Add heatmap
    fig2.add_trace(
        go.Heatmap(
            x=dates,
            y=lookback_labels,
            z=z_data,
            colorscale='RdBu_r',
            zmin=0,
            zmax=1,
            zsmooth='best',
            colorbar=dict(
                title='Avg Correlation',
                len=0.75,
                thickness=25,
                y=0.5,
                x=1.02,
            ),
        ),
        secondary_y=False,
    )

    # Add BTC price line
    fig2.add_trace(
        go.Scatter(
            x=plot_data.index,
            y=plot_data['BTC'],
            name='BTC Price',
            line=dict(color='black', width=3),
            hovertemplate='Date: %{x}<br>BTC Price: $%{y:.2f}<extra></extra>',
        ),
        secondary_y=True,
    )

    # Update layout
    fig2.update_layout(
        title=f'BTC Average Correlation with Altcoins ({effective_start} to {end_date})',
        xaxis_title='Date',
        yaxis_title='Lookback Period (Days)',
        height=800,
        width=1200,
        showlegend=True,
    )

    # Update axes
    fig2.update_yaxes(title_text='Lookback Period (Days)', secondary_y=False)
    fig2.update_yaxes(title_text='BTC Price ($)', secondary_y=True)

    return fig1, fig2


def plot_btc_financial_correlations(start_date, end_date, lookback=30):
    """
    Generate a heatmap showing BTC correlations with SPY, QQQ, GLD, TLT, and VIX for a Streamlit app.
    
    Parameters:
    - start_date (str or datetime.date): Start date in 'YYYY-MM-DD' format or date object.
    - end_date (str or datetime.date): End date in 'YYYY-MM-DD' format or date object.
    - lookback (int): Lookback window in days for correlation calculation (default: 30).
    
    Returns:
    - fig: Plotly figure for the heatmap.
    """
    # Convert date objects to strings if necessary
    if isinstance(start_date, date):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, date):
        end_date = end_date.strftime('%Y-%m-%d')

    # Initialize Binance.US client
    client = Client(tld='us')

    # Define symbols
    btc_symbol = 'BTCUSDT'
    financial_symbols = ['SPY', 'QQQ', 'GLD', 'TLT', '^VIX']

    # Calculate fetch start date (lookback + buffer prior to start_date)
    fetch_start_date = (pd.to_datetime(start_date) - timedelta(days=lookback + 10)).strftime('%Y-%m-%d')

    # Effective start date for display
    effective_start = start_date

    # Fetch BTC data from Binance
    def get_btc_daily_closes(symbol, start_date, end_date):
        try:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1DAY,
                start_str=start_date,
                end_str=end_date
            )
            if not klines:
                raise ValueError(f"No data returned for {symbol}")
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore'
            ])
            df['date'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('date', inplace=True)
            df['close'] = df['close'].astype(float)
            return df['close']
        except Exception as e:
            raise ValueError(f"Error fetching data for {symbol}: {e}")

    # Fetch financial data from yfinance
    def get_financial_daily_closes(symbols, start_date, end_date):
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                if df.empty:
                    print(f"No data returned for {symbol}")
                    continue
                # Remove timezone info to make index timezone-naive
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data[symbol] = df['Close']
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame(data)

    # Fetch data
    btc_data = get_btc_daily_closes(btc_symbol, fetch_start_date, end_date)
    financial_data = get_financial_daily_closes(financial_symbols, fetch_start_date, end_date)

    # Ensure BTC data is timezone-naive
    btc_data.index = pd.to_datetime(btc_data.index).tz_localize(None)

    # Combine BTC and financial data
    data = pd.DataFrame({'BTC': btc_data}).join(financial_data, how='inner')

    if data.empty:
        raise ValueError("No valid data after joining. Check data availability for the specified date range.")

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    if daily_returns.empty:
        raise ValueError("No valid daily returns data after processing.")

    # Calculate correlations
    correlation_matrix = pd.DataFrame(index=daily_returns.index, 
                                     columns=[t for t in data.columns if t != 'BTC'])

    for asset in [t for t in data.columns if t != 'BTC']:
        correlation_matrix[asset] = daily_returns['BTC'].rolling(window=lookback).corr(daily_returns[asset])

    # Filter for dates after effective_start, keeping non-NaN values
    correlation_matrix = correlation_matrix.loc[effective_start:].dropna()
    plot_data = data.loc[effective_start:]

    if correlation_matrix.empty:
        raise ValueError("No valid correlation data after filtering.")

    # Prepare data for heatmap
    dates = correlation_matrix.index
    asset_tickers = [t for t in data.columns if t != 'BTC']
    z_data = correlation_matrix[asset_tickers].T.values

    # Create subplot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            x=dates,
            y=asset_tickers,
            z=z_data,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            zsmooth='best',
            colorbar=dict(
                title='Correlation',
                len=0.75,
                thickness=25,
                y=0.5,
                x=1.02,
            ),
        ),
        secondary_y=False,
    )

    # Add BTC price line
    fig.add_trace(
        go.Scatter(
            x=plot_data.index,
            y=plot_data['BTC'],
            name='BTC Price',
            line=dict(color='black', width=3),
            hovertemplate='Date: %{x}<br>BTC Price: $%{y:.2f}<extra></extra>',
        ),
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title=f'BTC Correlation with TradFi ({effective_start} to {end_date}, {lookback}-Day Lookback)',
        xaxis_title='Date',
        yaxis_title='Assets',
        height=600,
        width=1200,
        showlegend=True,
    )

    # Update axes
    fig.update_yaxes(title_text='Assets', secondary_y=False)
    fig.update_yaxes(title_text='BTC Price ($)', secondary_y=True)

    return fig

