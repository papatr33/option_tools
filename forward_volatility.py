import requests
import pandas as pd
from datetime import datetime
from math import sqrt
import numpy as np
from math import isfinite

# Deribit API base URL
API_URL = "https://www.deribit.com/api/v2/public/"

def call_api(method, params):
    """Helper function to call Deribit API."""
    url = f"{API_URL}{method}"
    response = requests.get(url, params=params)
    return response.json()

def fetch_and_process_data(currency):
    """
    Fetch options data and calculate forward volatility matrix and ATM IVs.
    
    Args:
        currency (str): Either 'BTC' or 'ETH'
    
    Returns:
        tuple: (forward_matrix, atm_df, current_time)
    """
    # Validate currency
    if currency not in ['BTC', 'ETH']:
        raise ValueError("Currency must be 'BTC' or 'ETH'")

    # Fetch server time and index price
    server_time_resp = call_api("get_time", {})
    if 'result' not in server_time_resp:
        raise Exception("Failed to fetch server time")
    server_time = server_time_resp['result']
    current_time = datetime.fromtimestamp(server_time / 1000)

    index_price_resp = call_api("get_index_price", {"index_name": f"{currency.lower()}_usd"})
    if 'result' not in index_price_resp:
        raise Exception(f"Failed to fetch index price for {currency}")
    underlying_price = index_price_resp['result']['index_price']

    # Fetch options data
    instruments_resp = call_api("get_instruments", {"currency": currency, "kind": "option"})
    if 'result' not in instruments_resp:
        raise Exception(f"Failed to fetch instruments for {currency}")
    instruments = instruments_resp['result']
    instrument_dict = {inst['instrument_name']: inst for inst in instruments}

    book_summary_resp = call_api("get_book_summary_by_currency", {"currency": currency, "kind": "option"})
    if 'result' not in book_summary_resp:
        raise Exception(f"Failed to fetch book summary for {currency}")
    book_summary = book_summary_resp['result']

    # Collect options data
    options_data = []
    for summary in book_summary:
        inst_name = summary['instrument_name']
        mark_iv = summary.get('mark_iv')
        if inst_name in instrument_dict and mark_iv is not None:
            inst = instrument_dict[inst_name]
            expiry = datetime.fromtimestamp(inst['expiration_timestamp'] / 1000)
            time_to_expiry = (expiry - current_time).total_seconds() / (365.25 * 24 * 3600)
            if time_to_expiry > 0:
                options_data.append({
                    'expiry': expiry,
                    'time_to_expiry': time_to_expiry,
                    'strike': inst['strike'],
                    'option_type': inst['option_type'],
                    'mark_iv': mark_iv,
                    'instrument_name': inst_name
                })

    if not options_data:
        raise Exception(f"No valid options data found for {currency}")

    # Convert to DataFrame and find ATM IVs
    df = pd.DataFrame(options_data)
    grouped = df.groupby('expiry')
    atm_ivs = []
    for expiry in sorted(df['expiry'].unique()):
        group = grouped.get_group(expiry).copy()  # Create a copy to avoid SettingWithCopyWarning
        # Use .loc to assign new column safely
        group.loc[:, 'strike_diff'] = abs(group['strike'] - underlying_price)
        min_diff = group['strike_diff'].min()
        atm_options = group[group['strike_diff'] == min_diff]
        call_atm = atm_options[atm_options['option_type'] == 'call']
        if not call_atm.empty:
            atm_iv = call_atm['mark_iv'].mean()
            atm_ivs.append({
                'expiry': expiry,
                'time_to_expiry': group['time_to_expiry'].iloc[0],
                'atm_iv': atm_iv
            })

    if not atm_ivs:
        raise Exception(f"No ATM IVs calculated for {currency}")

    # Create ATM IVs DataFrame and calculate forward vols
    atm_df = pd.DataFrame(atm_ivs).sort_values('expiry')
    expiries = sorted(atm_df['expiry'].unique())
    forward_vols = []

    for i, start in enumerate(expiries):
        for j, end in enumerate(expiries):
            if end > start:
                T1 = atm_df[atm_df['expiry'] == start]['time_to_expiry'].values[0]
                sigma1 = atm_df[atm_df['expiry'] == start]['atm_iv'].values[0] / 100
                T2 = atm_df[atm_df['expiry'] == end]['time_to_expiry'].values[0]
                sigma2 = atm_df[atm_df['expiry'] == end]['atm_iv'].values[0] / 100
                if T2 > T1 and sigma2 > 0 and sigma1 > 0:
                    try:
                        forward_var = (sigma2**2 * T2 - sigma1**2 * T1) / (T2 - T1)
                        if forward_var > 0 and isfinite(forward_var):
                            forward_vol = sqrt(forward_var) * 100
                            forward_vols.append({
                                'start_expiry': start,
                                'end_expiry': end,
                                'forward_vol': forward_vol
                            })
                    except Exception as e:
                        continue

    # Create forward volatility matrix
    forward_matrix = pd.DataFrame(index=expiries, columns=expiries)
    for fv in forward_vols:
        forward_matrix.loc[fv['start_expiry'], fv['end_expiry']] = fv['forward_vol']
    forward_matrix = forward_matrix.astype(float)

    # Set lower triangle (including diagonal) to NaN
    for i in range(len(expiries)):
        for j in range(i + 1):
            forward_matrix.iloc[i, j] = np.nan

    # Create expiry labels
    expiry_labels = [
        f"{int((pd.Timestamp(exp).to_pydatetime() - current_time).total_seconds() / (24 * 3600))}d ({pd.Timestamp(exp).strftime('%m/%d/%y')})"
        for exp in expiries
    ]
    forward_matrix.index = expiry_labels
    forward_matrix.columns = expiry_labels

    return forward_matrix, atm_df, current_time
