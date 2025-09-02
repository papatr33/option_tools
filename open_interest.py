import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

def get_deribit_options_data(currency):
    """Fetch options data from Deribit API for the specified currency."""
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    params = {"currency": currency, "kind": "option"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])
    except requests.RequestException as e:
        return []

def process_options_data(options_data):
    """Process options data into a DataFrame and extract ranges."""
    records = []
    for item in options_data:
        instrument_name = item.get("instrument_name", "")
        parts = instrument_name.split("-")
        if len(parts) != 4:
            continue
        coin, expiry, strike, option_type = parts
        try:
            expiry_date = datetime.strptime(expiry, "%d%b%y")
            strike_price = float(strike)
        except ValueError:
            continue
        records.append({
            "strike_price": strike_price,
            "expiry_date": expiry_date,
            "option_type": option_type,
            "open_interest": item.get("open_interest", 0)
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df, [], 0, 0
    expiry_dates = sorted(df["expiry_date"].unique())
    min_strike, max_strike = df["strike_price"].min(), df["strike_price"].max()
    return df, expiry_dates, min_strike, max_strike

def create_bubble_chart(coin, start_date, end_date, min_strike, max_strike):
    """
    Create a Plotly bubble chart for options open interest.
    
    Args:
        coin (str): 'BTC' or 'ETH'
        start_date (datetime): Start of expiration date range
        end_date (datetime): End of expiration date range
        min_strike (float): Minimum strike price
        max_strike (float): Maximum strike price
    
    Returns:
        plotly.graph_objs.Figure: Plotly figure object, or None if no data
    """
    # Fetch and process data
    options_data = get_deribit_options_data(coin)
    if not options_data:
        return None

    df, expiry_dates, min_strike_available, max_strike_available = process_options_data(options_data)
    if df.empty:
        return None

    # Validate inputs
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    min_strike = max(min_strike, min_strike_available)
    max_strike = min(max_strike, max_strike_available)
    if min_strike > max_strike:
        min_strike, max_strike = max_strike, min_strike

    # Filter data
    df_filtered = df[
        (df["expiry_date"] >= start_date) &
        (df["expiry_date"] <= end_date) &
        (df["strike_price"] >= min_strike) &
        (df["strike_price"] <= max_strike)
    ]

    if df_filtered.empty:
        return None

    # Create bubble chart
    fig = px.scatter(
        df_filtered,
        x="strike_price",
        y="expiry_date",
        size="open_interest",
        color="option_type",
        hover_data=["open_interest", "option_type"],
        title=f"{coin} Options Open Interest on Deribit",
        labels={
            "strike_price": "Strike Price",
            "expiry_date": "Expiration Date",
        },
        color_discrete_map={"C": "#009392", "P": "#d77d8f"}
    )

    # Update traces
    max_oi = max(df_filtered["open_interest"], default=1)
    fig.update_traces(
        marker=dict(
            sizemode="area",
            sizeref=2. * max_oi / (80.**2),
            line=dict(width=0.5, color="DarkSlateGrey"),
            opacity=0.7
        ),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>" +
            "Strike: $%{x:,.0f}<br>" +
            "Expiry: %{y|%b %d, %Y}<br>" +
            "Open Interest: %{customdata[0]:,.0f}<extra></extra>"
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial, sans-serif")
        ),
        xaxis=dict(
            title="Strike Price",
            tickformat=",.0f",
            gridcolor="LightGrey",
            zeroline=False
        ),
        yaxis=dict(
            title="Expiration Date",
            tickformat="%b %d, %Y",
            gridcolor="LightGrey",
            zeroline=False
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            # title="Option Type",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=700,
        margin=dict(l=50, r=50, t=100, b=50),
        font=dict(family="Arial, sans-serif", size=12)
    )


    return fig