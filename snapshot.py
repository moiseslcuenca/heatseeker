"""
heatseeker/snapshot.py

Stores each run's node data as JSON and computes:
  - pct_change_15m: vs the immediately preceding intraday snapshot
  - pct_change_1d:  vs the prior trading day's closing snapshot

Storage layout (flat JSON files, no DB needed at this scale):
  data/<TICKER>/<YYYY-MM-DD>.json
      {
        "snapshots": [
          {"time": "14:32:01", "spot": 123.4, "nodes": {strike_bin: {"GEX":.., "VEX":.., ...}, ...}},
          ...
        ],
        "close_snapshot": {...} or null   # set on the last run of the day
      }
"""

import json
import os
from datetime import datetime, timezone

import pytz

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ET = pytz.timezone("America/New_York")


def _key(group, ticker):
    return f"{group}__{ticker}"


def _day_path(group, ticker, date_str):
    d = os.path.join(DATA_DIR, _key(group, ticker))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{date_str}.json")


def _load_day(group, ticker, date_str):
    path = _day_path(group, ticker, date_str)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"snapshots": [], "close_snapshot": None}


def _save_day(group, ticker, date_str, data):
    path = _day_path(group, ticker, date_str)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def find_prev_close(group, ticker, date_str):
    """Scan stored files for this ticker, find latest date < date_str with a close_snapshot."""
    tdir = os.path.join(DATA_DIR, _key(group, ticker))
    if not os.path.isdir(tdir):
        return None
    candidates = sorted(
        f for f in os.listdir(tdir)
        if f.endswith(".json") and f[:-5] < date_str
    )
    for fname in reversed(candidates):
        with open(os.path.join(tdir, fname)) as f:
            day = json.load(f)
        if day.get("close_snapshot"):
            return day["close_snapshot"]
    return None


def save_snapshot(group, ticker, nodes_df, spot_price, is_close=False, now_utc=None):
    """
    nodes_df: DataFrame indexed by strike_bin with GEX/VEX/DEX/CHARM columns
              (output of core.extract_nodes).
    Returns a DataFrame with pct_change_15m and pct_change_1d columns added.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    date_str = now_et.strftime("%Y-%m-%d")
    time_str = now_et.strftime("%H:%M:%S")

    day = _load_day(group, ticker, date_str)

    nodes_dict = {
        str(strike): {c: float(nodes_df.loc[strike, c]) for c in ["GEX", "VEX", "DEX", "CHARM"]}
        for strike in nodes_df.index
    }
    new_snap = {"time": time_str, "spot": float(spot_price), "nodes": nodes_dict}

    prev_snap = day["snapshots"][-1] if day["snapshots"] else None
    prev_close = find_prev_close(group, ticker, date_str)

    result = nodes_df.copy()
    result["pct_change_15m"] = _pct_change_series(nodes_dict, prev_snap["nodes"] if prev_snap else None)
    result["pct_change_1d"] = _pct_change_series(nodes_dict, prev_close["nodes"] if prev_close else None)

    day["snapshots"].append(new_snap)
    if is_close:
        day["close_snapshot"] = new_snap
    _save_day(group, ticker, date_str, day)

    return result


def _pct_change_series(current_nodes, ref_nodes):
    if not ref_nodes:
        return {k: None for k in current_nodes}
    out = {}
    for strike, vals in current_nodes.items():
        ref = ref_nodes.get(strike)
        if ref is None or ref.get("GEX") in (None, 0):
            out[strike] = None
            continue
        out[strike] = (vals["GEX"] - ref["GEX"]) / abs(ref["GEX"]) * 100
    import pandas as pd
    return pd.Series({float(k): v for k, v in out.items()})


# --- Lightweight per-expiration matrix tracking (overwritten each run, no
# growing history — only "now", "15 min ago", and "yesterday's close" are
# ever needed for the % change columns). ---

def _matrix_path(group, ticker, kind):
    d = os.path.join(DATA_DIR, _key(group, ticker))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"matrix_{kind}.json")


def _load_matrix_file(group, ticker, kind):
    path = _matrix_path(group, ticker, kind)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_matrix_file(group, ticker, kind, data):
    path = _matrix_path(group, ticker, kind)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _matrix_pct_change(current, ref):
    """current/ref: {strike: {expiration: value}}. Returns same shape with
    % change, or None per cell where there's no reference."""
    if not ref:
        return {s: {e: None for e in exps} for s, exps in current.items()}
    out = {}
    for strike, exps in current.items():
        out[strike] = {}
        ref_exps = ref.get(strike, {})
        for exp, val in exps.items():
            ref_val = ref_exps.get(exp)
            if val is None or ref_val is None or ref_val == 0:
                out[strike][exp] = None
            else:
                out[strike][exp] = (val - ref_val) / abs(ref_val) * 100
    return out


def save_matrix_snapshot(group, ticker, gex_dict, vex_dict, spot_price, is_close=False, now_utc=None):
    """
    gex_dict/vex_dict: {strike_str: {expiration_str: value}} (already JSON-safe).
    Returns {"GEX": {...}, "VEX": {...}} where each metric maps to
    {strike: {expiration: {"value":.., "pct_15m":.., "pct_1d":..}}}.

    Storage: matrix_prev.json is overwritten every run (source of 15m change).
    matrix_close.json is overwritten only on the day's last run (source of
    1d change), and is read BEFORE being overwritten so it always reflects
    the prior trading day until today's close actually happens.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)

    prev = _load_matrix_file(group, ticker, "prev")
    close = _load_matrix_file(group, ticker, "close")

    prev_date = prev.get("date") if prev else None
    today_str = now_et.strftime("%Y-%m-%d")
    # Only use "prev" as the 15m reference if it's from today (otherwise
    # it's stale from Friday and shouldn't be treated as "15 minutes ago").
    prev_gex = prev["GEX"] if prev and prev_date == today_str else None
    prev_vex = prev["VEX"] if prev and prev_date == today_str else None
    close_gex = close["GEX"] if close else None
    close_vex = close["VEX"] if close else None

    gex_pct15 = _matrix_pct_change(gex_dict, prev_gex)
    gex_pct1d = _matrix_pct_change(gex_dict, close_gex)
    vex_pct15 = _matrix_pct_change(vex_dict, prev_vex)
    vex_pct1d = _matrix_pct_change(vex_dict, close_vex)

    current = {
        "date": today_str,
        "time": now_et.strftime("%H:%M:%S"),
        "spot": float(spot_price),
        "GEX": gex_dict,
        "VEX": vex_dict,
    }
    _save_matrix_file(group, ticker, "prev", current)
    if is_close:
        _save_matrix_file(group, ticker, "close", current)

    def _combine(value_dict, pct15, pct1d):
        out = {}
        for strike, exps in value_dict.items():
            out[strike] = {}
            for exp, val in exps.items():
                out[strike][exp] = {
                    "value": val,
                    "pct_15m": pct15[strike][exp],
                    "pct_1d": pct1d[strike][exp],
                }
        return out

    return {
        "GEX": _combine(gex_dict, gex_pct15, gex_pct1d),
        "VEX": _combine(vex_dict, vex_pct15, vex_pct1d),
    }
