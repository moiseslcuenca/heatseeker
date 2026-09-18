"""
heatseeker/run_snapshot.py

Entrypoint called by GitHub Actions every 15 min during market hours.

Reads tickers.json (two groups: "market" and "watchlist"), fetches +
computes nodes for each entry, saves a snapshot per group+ticker with
pct_change_15m / pct_change_1d, and writes a combined latest.json shaped as:

{
  "market":    { "^SPX": {...}, "SPY": {...}, "QQQ": {...} },
  "watchlist": { "SPY": {...}, "QQQ": {...}, "AAPL": {...}, ... }
}

Storing market/SPY and watchlist/SPY separately (different folders under
data/) lets the same ticker be tracked with different expiration scope in
each group without the histories interfering.
"""

import json
import os
import sys
from datetime import datetime, timezone

import pytz

from core import fetch_heatseeker_data, extract_nodes, extract_matrix
from market_hours import is_market_open, is_last_run_of_day
from snapshot import save_snapshot, save_matrix_snapshot, DATA_DIR

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tickers.json")
ET = pytz.timezone("America/New_York")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _matrix_to_json(matrix):
    """DataFrame (strike x expiration) -> plain nested dict for JSON."""
    return {
        str(strike): {str(col): (None if pd_isna(v) else float(v)) for col, v in row.items()}
        for strike, row in matrix.iterrows()
    }


def pd_isna(v):
    import pandas as pd
    return pd.isna(v)


def run_group(group, entries, close_run, include_matrix=False):
    results = {}
    for t in entries:
        ticker = t["ticker"]
        bin_size = t.get("bin_size", 5)
        mode = t.get("mode", "daily")

        try:
            df, spot = fetch_heatseeker_data(ticker, mode=mode)
            if df is None:
                print(f"⚠️  [{group}] No data for {ticker}, skipping.")
                continue

            df_expiration_count = df["expirationDate"].nunique()
            print(f"   [{group}] {ticker}: raw fetch returned {df_expiration_count} distinct expirations "
                  f"({sorted(df['expirationDate'].unique())[:3]}{'...' if df_expiration_count > 3 else ''})")

            nodes = extract_nodes(df, spot, bin_size=bin_size)
            result = save_snapshot(group, ticker, nodes, spot, is_close=close_run)

            entry = {
                "spot": spot,
                "nodes": {
                    str(strike): {
                        "GEX": row["GEX"],
                        "VEX": row["VEX"],
                        "pct_change_15m": row["pct_change_15m"],
                        "pct_change_1d": row["pct_change_1d"],
                    }
                    for strike, row in result.iterrows()
                },
            }

            if include_matrix:
                matrix_gex = extract_matrix(df, spot, mode="GEX", bin_size=bin_size,
                                             zoom_range=(df["strike"].min(), df["strike"].max()))
                matrix_vex = extract_matrix(df, spot, mode="VEX", bin_size=bin_size,
                                             zoom_range=(df["strike"].min(), df["strike"].max()))
                gex_dict = _matrix_to_json(matrix_gex)
                vex_dict = _matrix_to_json(matrix_vex)

                annotated = save_matrix_snapshot(group, ticker, gex_dict, vex_dict, spot, is_close=close_run)
                entry["matrix"] = {
                    "GEX": annotated["GEX"],
                    "VEX": annotated["VEX"],
                    "expirations": [str(c) for c in matrix_gex.columns],
                }

            results[ticker] = entry
            print(f"✅ [{group}] {ticker}: spot ${spot:.2f}, {len(nodes)} nodes saved.")

        except Exception as e:
            # A single bad ticker (e.g. an index symbol like ^SPX behaving
            # differently from an ETF in yfinance) must never take down the
            # rest of the run — without this, one failure here crashed the
            # whole script before latest.json was ever written, which is
            # what caused an entire market/watchlist group to go silently
            # empty even though other tickers would have worked fine.
            print(f"❌ [{group}] {ticker} failed: {type(e).__name__}: {e}")
            continue

    return results


def main():
    is_manual_trigger = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

    if not is_manual_trigger and not is_market_open():
        print("Market closed. Skipping run.")
        return

    if is_manual_trigger and not is_market_open():
        print("Market closed, but this was a MANUAL trigger — running anyway for testing/diagnostics.")

    close_run = is_last_run_of_day()
    config = load_config()

    now_et = datetime.now(timezone.utc).astimezone(ET)

    latest = {
        "last_updated": now_et.strftime("%Y-%m-%d %H:%M %Z"),
        "market": run_group("market", config.get("market", []), close_run, include_matrix=False),
        "watchlist": run_group("watchlist", config.get("watchlist", []), close_run, include_matrix=True),
    }

    latest_path = os.path.join(DATA_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(latest, f, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
