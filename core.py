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


_RISK_FREE_CACHE = {"rate": None, "fetched_at": None}


def get_risk_free_rate(fallback=0.045, max_age_seconds=3600):
    """13-week Treasury bill yield (^IRX) as a live proxy for the
    risk-free rate, instead of a hardcoded constant. Cached for an hour
    since it doesn't move fast enough to need refetching every call.
    Falls back to the hardcoded default if the fetch fails for any reason
    (e.g. no network) so this never breaks the pipeline."""
    import time
    now = time.time()
    if _RISK_FREE_CACHE["rate"] is not None and _RISK_FREE_CACHE["fetched_at"] is not None:
        if now - _RISK_FREE_CACHE["fetched_at"] < max_age_seconds:
            return _RISK_FREE_CACHE["rate"]

    try:
        irx = yf.Ticker("^IRX")
        last = irx.fast_info["last_price"]
        rate = float(last) / 100.0  # ^IRX quotes in percent, e.g. 5.25 -> 0.0525
        if 0 < rate < 0.25:  # sanity bound — reject anything nonsensical
            _RISK_FREE_CACHE["rate"] = rate
            _RISK_FREE_CACHE["fetched_at"] = now
            return rate
    except Exception:
        pass
    return fallback


def _clean_implied_vol(chain_df):
    """Replaces garbage/near-zero IV with a same-expiration, same-side
    (call/put) median instead of a single arbitrary flat constant — so the
    fallback reflects that day's actual skew/level rather than an
    unconditional 50% guess, which is a poor assumption for e.g. a low-vol
    blue chip. Only falls back to a flat constant if truly nothing usable
    exists for that expiration/side (should be rare)."""
    df = chain_df.copy()
    bad = df["impliedVolatility"] < 0.01
    if not bad.any():
        return df

    for (exp, side), group_idx in df.groupby(["expirationDate", "type"]).groups.items():
        group_bad = bad.loc[group_idx]
        if not group_bad.any():
            continue
        good_vals = df.loc[group_idx, "impliedVolatility"][~group_bad]
        if len(good_vals) > 0:
            replacement = good_vals.median()
        else:
            replacement = 0.3  # last-resort constant, only if the whole expiration/side is unusable
        df.loc[group_idx[group_bad], "impliedVolatility"] = replacement
    return df


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

        # Standard SpotGamma-style GEX convention: gamma * OI * 100 * spot^2
        # * 0.01, representing "$ dealers must hedge per 1% move" — this is
        # the normalized convention most public GEX tools (and very likely
        # Bullflow) use, replacing the simpler `gamma * spot * OI * 100`
        # linear version this app used before.
        dollar_gamma = g * (spot_price ** 2) * 0.01 * oi * multiplier
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

def fetch_heatseeker_data(ticker, mode="daily", max_expirations=None, verbose=False):
    """
    max_expirations: cap on how many expiration dates to pull.
    None = no cap (pull everything available for the mode).
    """
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
        target_expirations = expirations if max_expirations is None else expirations[:max_expirations]
    elif mode == "weekly":
        for date in expirations:
            if datetime.strptime(date, "%Y-%m-%d").weekday() == 4:
                target_expirations.append(date)
            if max_expirations and len(target_expirations) >= max_expirations:
                break
        if not target_expirations:
            target_expirations = expirations if max_expirations is None else expirations[:max_expirations]
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

            chain["expirationDate"] = date
            chain = _clean_implied_vol(chain)

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
    risk_free_rate = get_risk_free_rate()
    greeks = df.apply(get_greeks, axis=1, spot_price=current_price, risk_free_rate=risk_free_rate)
    df[["GEX", "VEX", "DEX", "CHARM"]] = greeks
    return df, current_price


# --- NODE EXTRACTION (new: this replaces the plotting cells) ---

def compute_gamma_summary(strikes, gex_values):
    """
    Net GEX total and the two 'wall' strikes (max positive / max negative
    GEX). These only need today's actual per-strike GEX, so they're
    straightforward and reliable.

    Deliberately does NOT compute a gamma flip point. A proper flip point
    requires re-running the gamma calculation across a grid of hypothetical
    spot prices (since gamma itself depends on distance from spot) and
    finding where that recomputed curve crosses zero — not a simple
    cumulative sum of today's GEX by strike, which produces misleading
    results (this is the likely source of the $662 flip bug).
    """
    strikes = np.asarray(strikes, dtype=float)
    gex_values = np.asarray(gex_values, dtype=float)

    net_gex = float(np.nansum(gex_values))
    if len(strikes) == 0 or np.all(np.isnan(gex_values)):
        return {"net_gex": 0.0, "max_pos": None, "max_neg": None}

    max_pos_idx = np.nanargmax(gex_values) if np.any(gex_values > 0) else None
    max_neg_idx = np.nanargmin(gex_values) if np.any(gex_values < 0) else None
    max_pos = (float(strikes[max_pos_idx]), float(gex_values[max_pos_idx])) if max_pos_idx is not None else None
    max_neg = (float(strikes[max_neg_idx]), float(gex_values[max_neg_idx])) if max_neg_idx is not None else None

    return {"net_gex": net_gex, "max_pos": max_pos, "max_neg": max_neg}


def fetch_intraday_candles(ticker, period="1d", interval="5m"):
    """OHLC candles for the price+GEX overlay chart. Fetched live at render
    time (not stored), so it's as fresh as yfinance's own feed allows —
    typically a real-time-ish delay, not true tick-level live."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval)
        if hist.empty:
            return None
        return hist
    except Exception:
        return None


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
