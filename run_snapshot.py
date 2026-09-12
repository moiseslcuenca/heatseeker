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

from core import fetch_heatseeker_data, extract_nodes
from market_hours import is_market_open, is_last_run_of_day
from snapshot import save_snapshot, DATA_DIR

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tickers.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def run_group(group, entries, close_run):
    results = {}
    for t in entries:
        ticker = t["ticker"]
        bin_size = t.get("bin_size", 5)
        mode = t.get("mode", "daily")

        df, spot = fetch_heatseeker_data(ticker, mode=mode)
        if df is None:
            print(f"⚠️  [{group}] No data for {ticker}, skipping.")
            continue

        nodes = extract_nodes(df, spot, bin_size=bin_size)
        result = save_snapshot(group, ticker, nodes, spot, is_close=close_run)

        results[ticker] = {
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
        print(f"✅ [{group}] {ticker}: spot ${spot:.2f}, {len(nodes)} nodes saved.")
    return results


def main():
    if not is_market_open():
        print("Market closed. Skipping run.")
        return

    close_run = is_last_run_of_day()
    config = load_config()

    latest = {
        "market": run_group("market", config.get("market", []), close_run),
        "watchlist": run_group("watchlist", config.get("watchlist", []), close_run),
    }

    latest_path = os.path.join(DATA_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(latest, f, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
