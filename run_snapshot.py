"""
heatseeker/run_snapshot.py

Entrypoint called by GitHub Actions every 15 min during market hours.
Reads tickers.json, fetches + computes nodes for each, saves a snapshot
with pct_change_15m / pct_change_1d, writes a combined latest.json for
the dashboard to read.
"""

import json
import os
import sys

from core import fetch_heatseeker_data, extract_nodes
from market_hours import is_market_open, is_last_run_of_day
from snapshot import save_snapshot, DATA_DIR

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tickers.json")


def load_tickers():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg["tickers"]  # e.g. [{"ticker": "SPY", "bin_size": 5, "mode": "daily"}, ...]


def main():
    if not is_market_open():
        print("Market closed. Skipping run.")
        return

    close_run = is_last_run_of_day()
    tickers = load_tickers()
    latest = {}

    for t in tickers:
        ticker = t["ticker"]
        bin_size = t.get("bin_size", 5)
        mode = t.get("mode", "daily")

        df, spot = fetch_heatseeker_data(ticker, mode=mode)
        if df is None:
            print(f"⚠️  No data for {ticker}, skipping.")
            continue

        nodes = extract_nodes(df, spot, bin_size=bin_size)
        result = save_snapshot(ticker, nodes, spot, is_close=close_run)

        latest[ticker] = {
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
        print(f"✅ {ticker}: spot ${spot:.2f}, {len(nodes)} nodes saved.")

    latest_path = os.path.join(DATA_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(latest, f, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
