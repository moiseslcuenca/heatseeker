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
