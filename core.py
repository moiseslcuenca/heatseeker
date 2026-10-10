"""
heatseeker/core.py

Headless port of the HeatSeeker notebook's data + greeks engine.
No plotting, no !pip installs, no display/print side effects (only logging).
Math is unchanged from the original notebook.
"""

from datetime import datetime
import re
import threading
import time as _time
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
    unconditional 50% guess. Only falls back to a flat constant if truly
    nothing usable exists for that expiration/side (should be rare).

    Uses groupby().transform() rather than label-based .loc[group_idx, ...]
    indexing — the incoming chain (calls + puts concatenated without
    ignore_index) always has DUPLICATE index labels (0,1,2... from calls,
    then 0,1,2... again from puts), which breaks label-based .loc lookups.
    transform() aligns by position within each group instead, so it's safe
    regardless of duplicate labels."""
    df = chain_df.copy()
    bad = df["impliedVolatility"] < 0.01
    if not bad.any():
        return df

    def _replacement(s):
        good = s[s >= 0.01]
        return good.median() if len(good) > 0 else 0.3

    group_replacement = df.groupby(["expirationDate", "type"])["impliedVolatility"].transform(_replacement)
    df.loc[bad, "impliedVolatility"] = group_replacement[bad]
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


# --- DATA ENGINE ---

def _describe_error(e):
    """Turns an exception into a short, human-readable reason. Rate-limit
    style failures (HTTP 429 / 'Too Many Requests') are called out
    explicitly since that's the leading suspect for the morning lookup
    outage; everything else keeps its exception type + message so the real
    cause is visible instead of being swallowed."""
    if isinstance(e, KeyError) and e.args and e.args[0] in ("calls", "puts"):
        return ("Yahoo returned no option data for this expiration "
                "(its session was rejected or rate limited)")
    text = f"{type(e).__name__}: {e}"
    low = text.lower()
    if "429" in low or "too many requests" in low or "rate limit" in low or "ratelimit" in low:
        return f"Rate limited by Yahoo ({text[:200]})"
    return text[:300]


def _get_price(tk):
    """Returns (price, error). fast_info can return None/NaN instead of
    raising, which the old code let through as a 'valid' price, so both
    cases are checked here."""
    problems = []
    try:
        p = tk.fast_info["last_price"]
        if p is not None and np.isfinite(float(p)) and float(p) > 0:
            return float(p), None
        problems.append("fast_info returned no usable price")
    except Exception as e:
        problems.append(f"fast_info failed: {_describe_error(e)}")

    try:
        hist = tk.history(period="1d")
        if hist is None or hist.empty:
            problems.append("history returned no rows")
        else:
            p = float(hist["Close"].iloc[-1])
            if np.isfinite(p) and p > 0:
                return p, None
            problems.append("history close was not a usable price")
    except Exception as e:
        problems.append(f"history failed: {_describe_error(e)}")

    return None, "Couldn't get a price (" + "; ".join(problems) + ")"


def _peek_yahoo_options_error(tk, ticker):
    """yfinance does NOT raise when Yahoo answers the options request with
    an error payload (bad crumb, throttling...): it silently yields an empty
    expiration list. This re-reads the raw response to recover Yahoo's own
    error text. Uses a private yfinance attribute, so it's best-effort and
    returns None if anything about it fails."""
    try:
        r = tk._data.get(url=f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}")
        try:
            body = r.json()
        except Exception:
            snippet = (getattr(r, "text", "") or "")[:120].strip()
            return _describe_error(Exception(f"HTTP {getattr(r, 'status_code', '?')} non-JSON reply: {snippet}"))
        for path in (("optionChain", "error"), ("finance", "error")):
            node = body
            for k in path:
                node = node.get(k) if isinstance(node, dict) else None
            if node:
                if isinstance(node, dict):
                    return f"{node.get('code', '')}: {node.get('description', '')}".strip(": ")
                return str(node)
        res = (body.get("optionChain") or {}).get("result")
        if not res:
            return "Yahoo returned an options response with no result"
    except Exception as e:
        return f"raw check failed: {_describe_error(e)}"
    return None


# --- Yahoo session health -------------------------------------------------
# yfinance keeps ONE cookie+crumb for the whole process (a singleton shared by
# every Ticker and every Streamlit viewer). If Yahoo's crumb endpoint is rate
# limited, yfinance (1.7.0) stores Yahoo's error text as the crumb and keeps
# reusing it, so every options request fails with "Invalid Crumb" until the
# process restarts. Creating a new yf.Ticker does NOT reset this; the
# singleton has to be reset directly. Everything below is best-effort and
# touches private yfinance attributes, so it never raises.

_AUTH = {"cooldown_until": 0.0, "last_error": None}
_AUTH_COOLDOWN_SECONDS = 60


def _yahoo_auth():
    try:
        from yfinance.data import YfData
        return YfData()
    except Exception:
        return None


def _crumb_is_junk(c):
    """A real crumb is a short token. Yahoo's error text (spaces, HTML) is not."""
    return isinstance(c, str) and (len(c) > 40 or re.search(r"[\s<>]", c) is not None)


def _yahoo_session_state():
    """One-line description of yfinance's shared session, for error messages
    (this is what will confirm or refute the 'junk crumb' diagnosis)."""
    yd = _yahoo_auth()
    if yd is None:
        return "unavailable"
    try:
        c = getattr(yd, "_crumb", None)
        if c is None:
            crumb = "missing"
        elif _crumb_is_junk(c):
            crumb = f"JUNK {str(c)[:60]!r}"
        else:
            crumb = f"present (length {len(c)})"
        return f"cookie strategy={getattr(yd, '_cookie_strategy', '?')}, crumb={crumb}"
    except Exception:
        return "unavailable"


def _wipe_persisted_cookies():
    """yfinance also keeps the Yahoo cookie in an on-disk cache and, after an
    in-memory reset, reloads it if it hasn't 'expired'. A stale-but-unexpired
    cookie would therefore come straight back, so it has to be deleted too."""
    try:
        from yfinance import cache as yc
        cc = yc.get_cookie_cache()
        if getattr(cc, "dummy", False):
            return False
        if cc.initialised == -1:
            cc.initialise()
        if cc.initialised != 1:
            return False
        yc._CookieSchema.delete().execute()
        return True
    except Exception:
        return False


def _reset_yahoo_session(toggle_strategy=False):
    """Throw away yfinance's cached crumb, cookie (memory AND disk) so the
    next request fetches fresh ones; optionally switch between its two
    cookie strategies (basic <-> csrf). Returns True if a reset happened.
    Never raises."""
    yd = _yahoo_auth()
    if yd is None:
        return False
    try:
        lock = getattr(yd, "_cookie_lock", None)
        if lock is not None and not lock.acquire(timeout=2):
            return False  # another thread is mid-refresh; leave it alone
        try:
            yd._crumb = None
            yd._cookie = None
            if not getattr(yd, "_logged_in", False):
                yd._session.cookies.clear()
            if toggle_strategy:
                yd._set_cookie_strategy("csrf" if yd._cookie_strategy == "basic" else "basic", have_lock=True)
        finally:
            if lock is not None:
                lock.release()
        _wipe_persisted_cookies()
        return True
    except Exception:
        return False


def _heal_junk_crumb():
    """If the cached crumb is obviously Yahoo's error text, reset now."""
    yd = _yahoo_auth()
    if yd is not None and _crumb_is_junk(getattr(yd, "_crumb", None)):
        return _reset_yahoo_session()
    return False


def _is_auth_error(text):
    return bool(text) and re.search(r"crumb|unauthorized|invalid cookie|\b401\b", text, re.I) is not None


def _auth_cooldown_remaining():
    return max(0.0, _AUTH["cooldown_until"] - _time.time())


def trip_auth_breaker(reason):
    _AUTH["cooldown_until"] = _time.time() + _AUTH_COOLDOWN_SECONDS
    _AUTH["last_error"] = reason


def reset_auth_breaker():
    _AUTH["cooldown_until"] = 0.0
    _AUTH["last_error"] = None


def _list_expirations(ticker, tk, attempts=3, base_delay=2.0):
    """Returns (tk, expirations, error).

    - A junk cached crumb is cleared before the first attempt.
    - If Yahoo rejects the session (Invalid Crumb / Unauthorized), the shared
      yfinance session is fully reset before the next attempt (and the cookie
      strategy is switched before the last one), so a retry really starts fresh.
    - If it is still rejected, a ~60s circuit breaker stops every further
      lookup from hammering Yahoo (which keeps a rate-limited IP blocked) and
      the user is told to wait instead.
    - A ticker with genuinely no options just fails after the retries."""
    wait = _auth_cooldown_remaining()
    if wait > 0:
        return tk, None, (
            f"Yahoo recently rejected this server's session ({_AUTH['last_error']}). "
            f"Pausing live lookups for another {wait:.0f}s so the block can clear; try again then. "
            f"(This is a Yahoo/server-side block, not a problem with the symbol.)"
        )

    _heal_junk_crumb()
    last, auth_problem = None, False
    for i in range(attempts):
        if i > 0:
            if _is_auth_error(last):
                _reset_yahoo_session(toggle_strategy=(i == attempts - 1))
            _time.sleep(base_delay * (2 ** (i - 1)))
            tk = yf.Ticker(ticker)
        try:
            exps = tk.options
            if exps:
                return tk, exps, None
            last = _peek_yahoo_options_error(tk, ticker)
        except Exception as e:
            last = _describe_error(e)
        if _is_auth_error(last):
            auth_problem = True

    if auth_problem:
        trip_auth_breaker(last)
        return tk, None, (
            f"Yahoo rejected this server's session after {attempts} attempts. Yahoo said: {last}. "
            f"Session state: {_yahoo_session_state()}. Live lookups are paused ~{_AUTH_COOLDOWN_SECONDS}s "
            f"to let the block clear. (This is a Yahoo/server-side block, not a problem with the symbol.)"
        )
    detail = f" Yahoo said: {last}." if last else ""
    return tk, None, (
        f"No option expirations for '{ticker}' after {attempts} attempts.{detail} "
        f"(Either the symbol has no listed options, or Yahoo rejected the request.)"
    )


def _fetch_diag_uncached(ticker, mode="daily", max_expirations=None, verbose=False):
    """
    Same job as fetch_heatseeker_data, but returns (df, price, error).
    On success error is None. On failure df and price are None and error is
    a human-readable reason (price failed / no expirations / rate limited /
    every chain failed + first exception text).

    max_expirations: cap on how many expiration dates to pull.
    None = no cap (pull everything available for the mode).
    """
    tk = yf.Ticker(ticker)

    current_price, price_err = _get_price(tk)
    if current_price is None:
        return None, None, price_err

    if verbose:
        print(f"Fetching {ticker} (price: ${current_price:.2f})...")

    tk, expirations, exp_err = _list_expirations(ticker, tk)
    if exp_err:
        return None, None, exp_err

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
    chain_errors = []  # (date, reason) for expirations that raised
    empty_chains = 0
    consecutive_errors = 0
    stopped_early = False
    for date in target_expirations:
        try:
            opt = tk.option_chain(date)
            consecutive_errors = 0
            if opt.calls.empty and opt.puts.empty:
                empty_chains += 1
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
        except Exception as e:
            chain_errors.append((date, _describe_error(e)))
            consecutive_errors += 1
            # Nothing has worked yet and 3 in a row failed: Yahoo is rejecting
            # us, and pulling 30 more expirations would just hammer it.
            if consecutive_errors >= 3 and not all_options:
                stopped_early = True
                break
            continue

    if not all_options:
        n = len(target_expirations)
        if chain_errors:
            return None, None, (
                f"All {n} option chain request(s) failed or came back empty "
                f"({len(chain_errors)} errored, {empty_chains} empty"
                f"{'; stopped early after 3 failures in a row' if stopped_early else ''}). "
                f"First error ({chain_errors[0][0]}): {chain_errors[0][1]}"
            )
        return None, None, f"All {n} option chain(s) came back empty."

    df = pd.concat(all_options)
    if verbose:
        print("Calculating GEX, VEX, DEX, CHARM...")
    risk_free_rate = get_risk_free_rate()
    greeks = df.apply(get_greeks, axis=1, spot_price=current_price, risk_free_rate=risk_free_rate)
    df[["GEX", "VEX", "DEX", "CHARM"]] = greeks
    return df, current_price, None


_CHAIN_CACHE = {}  # (ticker, mode, max_expirations) -> (fetched_at, df, price)


def fetch_heatseeker_data_diag(ticker, mode="daily", max_expirations=None, verbose=False, cache_seconds=0):
    """Returns (df, price, error); on failure df and price are None and error
    is a human-readable reason.

    cache_seconds > 0 serves a successful result for that long, shared by
    every viewer, so repeat lookups of the same ticker don't hit Yahoo again.
    Failures are never cached. Default 0 keeps the scheduled job unchanged."""
    key = (str(ticker).upper(), mode, max_expirations)
    if cache_seconds > 0:
        hit = _CHAIN_CACHE.get(key)
        if hit and _time.time() - hit[0] < cache_seconds:
            return hit[1].copy(), hit[2], None
    df, price, err = _fetch_diag_uncached(ticker, mode, max_expirations, verbose)
    if err is None and cache_seconds > 0:
        if len(_CHAIN_CACHE) >= 20:
            _CHAIN_CACHE.pop(min(_CHAIN_CACHE, key=lambda k: _CHAIN_CACHE[k][0]))
        _CHAIN_CACHE[key] = (_time.time(), df.copy(), price)
    return df, price, err


def fetch_heatseeker_data(ticker, mode="daily", max_expirations=None, verbose=False):
    """Backward-compatible wrapper: same (df, price) return as before, so
    run_snapshot.py and any other caller keep working unchanged. The
    failure reason is printed to the log instead of being discarded."""
    df, price, err = fetch_heatseeker_data_diag(ticker, mode, max_expirations, verbose)
    if err:
        print(f"   fetch failed for {ticker}: {err}")
    return df, price


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


_CANDLE_CACHE = {}  # (ticker, period, interval) -> (fetched_at, DataFrame)
_CANDLE_TTL_SECONDS = 30


def fetch_intraday_candles(ticker, period="1d", interval="5m"):
    """OHLC candles for the price+GEX overlay chart. Fetched live at render
    time (not stored), so it's as fresh as yfinance's own feed allows —
    typically a real-time-ish delay, not true tick-level live.

    Successful results are cached for 30s (shared by every viewer of the
    app) so a chart that auto-refreshes doesn't multiply Yahoo requests.
    Failures are never cached."""
    import time
    key = (ticker, period, interval)
    now = time.time()
    hit = _CANDLE_CACHE.get(key)
    if hit and now - hit[0] < _CANDLE_TTL_SECONDS:
        return hit[1]
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval)
        if hist.empty:
            return None
        _CANDLE_CACHE[key] = (now, hist)
        return hist
    except Exception:
        return None


# --- LIVE QUOTE (for the auto-refreshing spot price in the UI) ---

import threading as _threading

_QUOTE_CACHE = {}  # ticker -> (fetched_at, quote_dict)
_QUOTE_LOCK = _threading.Lock()
_QUOTE_FAIL_TTL_SECONDS = 5  # failures are retried sooner than successes expire


def fetch_live_quote(ticker, max_age_seconds=10):
    """
    Latest price (and previous close, for a day-change figure) for one
    ticker. Returns {"price": float|None, "prev_close": float|None,
    "error": str|None, "fetched_at": epoch seconds}.

    Cached per ticker in module memory, which Streamlit shares across all
    viewer sessions, so N open browsers still cost at most one Yahoo call
    per ticker per max_age_seconds. The lock stops several sessions that
    refresh at the same instant from all hitting Yahoo at once. Never
    raises.
    """
    import time
    with _QUOTE_LOCK:
        now = time.time()
        hit = _QUOTE_CACHE.get(ticker)
        if hit:
            ttl = max_age_seconds if hit[1]["price"] is not None else min(max_age_seconds, _QUOTE_FAIL_TTL_SECONDS)
            if now - hit[0] < ttl:
                return hit[1]

        try:
            tk = yf.Ticker(ticker)
            price, err = _get_price(tk)
            prev_close = None
            if price is not None:
                try:
                    pc = tk.fast_info["previous_close"]
                    if pc is not None and np.isfinite(float(pc)) and float(pc) > 0:
                        prev_close = float(pc)
                except Exception:
                    pass
            quote = {"price": price, "prev_close": prev_close, "error": err, "fetched_at": now}
        except Exception as e:
            quote = {"price": None, "prev_close": None, "error": _describe_error(e), "fetched_at": now}

        _QUOTE_CACHE[ticker] = (now, quote)
        return quote


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
