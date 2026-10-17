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

Who is allowed to run when:
  - scheduled runs and dispatches sent by the outside timer (source=timer)
    only run inside the windows below (regular session, plus the optional
    pre-open / post-close windows) and exit immediately otherwise;
  - a hand-clicked "Run workflow" (no source) always runs, for testing.
"""

import json
import os
import sys
from datetime import datetime, timezone

import pytz

from core import fetch_heatseeker_data, extract_nodes, extract_matrix
from market_hours import get_run_phase, is_last_run_of_day
from snapshot import save_snapshot, save_matrix_snapshot, DATA_DIR

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tickers.json")
ET = pytz.timezone("America/New_York")

# Extra windows (besides the regular session) in which automated runs may
# snapshot. Flip these to change behaviour; nothing else needs editing.
ENABLE_PRE_OPEN = True      # ~15 min before the open
ENABLE_POST_CLOSE = False   # ~15 min after the close


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
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    source = os.environ.get("RUN_SOURCE", "").strip().lower()
    # A hand-clicked "Run workflow" is a manual test and always runs. A
    # dispatch that carries source=timer comes from the outside timer and is
    # gated exactly like a scheduled run.
    is_manual_trigger = event == "workflow_dispatch" and source != "timer"

    phase = get_run_phase()
    allowed = (
        phase == "open"
        or (phase == "pre" and ENABLE_PRE_OPEN)
        or (phase == "post" and ENABLE_POST_CLOSE)
    )

    if not is_manual_trigger and not allowed:
        print(f"Outside the run window (phase: {phase}). Skipping run.")
        return

    if is_manual_trigger and not allowed:
        print("Outside the run window, but this was a MANUAL trigger — running anyway for testing/diagnostics.")
    elif phase in ("pre", "post"):
        print(f"Running in the {phase}-market window.")

    close_run = is_last_run_of_day()
    config = load_config()

    now_et = datetime.now(timezone.utc).astimezone(ET)

    latest = {
        "last_updated": now_et.strftime("%Y-%m-%d %H:%M %Z"),
        "market": run_group("market", config.get("market", []), close_run, include_matrix=False),
        "watchlist": run_group("watchlist", config.get("watchlist", []), close_run, include_matrix=True),
    }

    # The extra windows (pre-open / post-close) are more likely to hit empty
    # Yahoo responses than the regular session. If an automated run there
    # comes back with an empty group, keep the previous latest.json instead
    # of blanking the app until the next run.
    if phase in ("pre", "post") and not is_manual_trigger:
        empty = [g for g in ("market", "watchlist") if config.get(g) and not latest[g]]
        if empty:
            print(f"Incomplete data in the {phase}-market window ({', '.join(empty)} came back empty). "
                  f"Leaving latest.json as it is.")
            return

    latest_path = os.path.join(DATA_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(latest, f, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
