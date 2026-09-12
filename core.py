"""
heatseeker/core.py

Headless port of the HeatSeeker notebook's data + greeks engine.
No plotting, no !pip installs, no display/print side effects (only logging).
Math is unchanged from the original notebook.
"""

from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from py_vollib.black_scholes.greeks.analytical import gamma, delta

warnings.simplefilter(action="ignore", category=FutureWarning)


# --- GREEKS MATH (unchanged from notebook) ---

def calculate_vanna(flag, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    n_prime_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1 ** 2)
    vanna = -n_prime_d1 * d2 / sigma
    return vanna


def calculate_charm(flag, S, K, T, r, sigma):
    if T <= 1 / 365.0:
        return 0
    delta_today = delta(flag, S, K, T, r, sigma)
    T_tomorrow = max(T - (1 / 365.0), 0.00001)
    delta_tomorrow = delta(flag, S, K, T_tomorrow, r, sigma)
    return delta_tomorrow - delta_today


def get_greeks(row, spot_price, risk_free_rate=0.045):
    try:
        T = row["days_to_expiration"] / 365.0
        if T <= 0:
            return pd.Series([0, 0, 0, 0])
        sigma = max(row["impliedVolatility"], 0.01)
        flag = "c" if row["type"] == "call" else "p"
        K = row["strike"]
        oi = row["openInterest"] if row["openInterest"] > 0 else 0
        multiplier = 100

        g = gamma(flag, spot_price, K, T, risk_free_rate, sigma)
        v = calculate_vanna(flag, spot_price, K, T, risk_free_rate, sigma)
        d = delta(flag, spot_price, K, T, risk_free_rate, sigma)
        c = calculate_charm(flag, spot_price, K, T, risk_free_rate, sigma)

        dollar_gamma = g * spot_price * oi * multiplier
        dollar_vanna = v * spot_price * oi * multiplier * sigma
        dollar_delta = d * spot_price * oi * multiplier
        dollar_charm = c * spot_price * oi * multiplier

        if flag == "c":
            return pd.Series([dollar_gamma, dollar_vanna, dollar_delta, dollar_charm])
        else:
            return pd.Series([-dollar_gamma, -dollar_vanna, dollar_delta, -dollar_charm])
    except Exception:
        return pd.Series([0, 0, 0, 0])


# --- DATA ENGINE (unchanged logic, print -> logging optional) ---

def fetch_heatseeker_data(ticker, mode="daily", verbose=False):
    tk = yf.Ticker(ticker)
    try:
        current_price = tk.fast_info["last_price"]
    except Exception:
        try:
            current_price = tk.history(period="1d")["Close"].iloc[-1]
        except Exception:
            return None, None

    if verbose:
        print(f"Fetching {ticker} (price: ${current_price:.2f})...")

    try:
        expirations = tk.options
    except Exception:
        return None, None
    if not expirations:
        return None, None

    target_expirations = []
    if mode == "daily":
        target_expirations = expirations[:6]
    elif mode == "weekly":
        for date in expirations:
            if datetime.strptime(date, "%Y-%m-%d").weekday() == 4:
                target_expirations.append(date)
            if len(target_expirations) >= 6:
                break
        if not target_expirations:
            target_expirations = expirations[:6]
    elif mode == "all":
        target_expirations = list(expirations)
    elif mode == "today":
        target_expirations = expirations[:1]

    all_options = []
    for date in target_expirations:
        try:
            opt = tk.option_chain(date)
            if opt.calls.empty and opt.puts.empty:
                continue
            calls = opt.calls.copy()
            calls["type"] = "call"
            puts = opt.puts.copy()
            puts["type"] = "put"
            chain = pd.concat([calls, puts]).fillna(0)

            mask_zero = chain["impliedVolatility"] < 0.01
            if mask_zero.any():
                chain.loc[mask_zero, "impliedVolatility"] = 0.5

            chain["expirationDate"] = date
            days = (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days
            chain["days_to_expiration"] = max(days, 0.5)
            chain["T_precise"] = chain["days_to_expiration"] / 365.0
            all_options.append(chain)
        except Exception:
            continue

    if not all_options:
        return None, None
    df = pd.concat(all_options)
    if verbose:
        print("Calculating GEX, VEX, DEX, CHARM...")
    greeks = df.apply(get_greeks, axis=1, spot_price=current_price)
    df[["GEX", "VEX", "DEX", "CHARM"]] = greeks
    return df, current_price


# --- NODE EXTRACTION (new: this replaces the plotting cells) ---

def extract_nodes(df, current_price, bin_size=5, top_n=None):
    """
    Collapse the raw option chain into per-strike-bin nodes with summed
    GEX / VEX / DEX / CHARM. This is the thing we snapshot over time.

    Returns a DataFrame indexed by strike_bin with columns:
    GEX, VEX, DEX, CHARM, distance_from_spot
    """
    df_scan = df.copy()
    df_scan["strike_bin"] = (df_scan["strike"] / bin_size).round() * bin_size
    nodes = df_scan.groupby("strike_bin")[["GEX", "VEX", "DEX", "CHARM"]].sum()
    nodes["distance_from_spot"] = (nodes.index.to_series() - current_price).abs()
    nodes = nodes.sort_index()
    if top_n:
        nodes = nodes.reindex(
            nodes["GEX"].abs().sort_values(ascending=False).index[:top_n]
        ).sort_index()
    return nodes


def extract_matrix(df, current_price, mode="GEX", bin_size=5, zoom_range=None):
    """
    Strike x Expiration matrix (rows=strike_bin, cols=expirationDate), summed
    over the chosen greek. This is what feeds the heatmap view, matching the
    original notebook's plot_heatseeker().

    zoom_range: optional (low, high) strike bounds; defaults to a window
    around current_price sized off bin_size, same as the notebook.
    """
    df_plot = df.copy()
    df_plot["strike_bin"] = (df_plot["strike"] / bin_size).round() * bin_size
    matrix = df_plot.groupby(["strike_bin", "expirationDate"])[mode].sum().unstack()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    if zoom_range is None:
        r = max(8 * bin_size, 4.0)
        zoom_range = (current_price - r, current_price + r)
    zoomed = matrix.loc[(matrix.index >= zoom_range[0]) & (matrix.index <= zoom_range[1])]
    if zoomed.empty:
        zoomed = matrix
    return zoomed.sort_index(ascending=False)  # descending strikes, spot near top
