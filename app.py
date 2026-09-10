"""
heatseeker/app.py

Streamlit dashboard. Reads data/latest.json (committed by the GitHub Actions
job) and shows, per ticker, each node's GEX/VEX plus % change vs 15 min ago
and vs yesterday's close.

To run locally:  streamlit run app.py
On Streamlit Cloud: point the app at this file; it auto-redeploys whenever
the repo updates, which happens every 15 min thanks to the Actions commit.
"""

import base64
import json
import os

import pandas as pd
import requests
import streamlit as st

from core import fetch_heatseeker_data, extract_nodes

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "latest.json")
GITHUB_API = "https://api.github.com"

st.set_page_config(page_title="HeatSeeker", layout="wide")
st.title("🎯 HeatSeeker Node Tracker")


def load_latest():
    if not os.path.exists(DATA_PATH):
        return None
    with open(DATA_PATH) as f:
        return json.load(f)


def color_pct(val):
    if val is None or pd.isna(val):
        return ""
    color = "#00C853" if val > 0 else "#FF3131" if val < 0 else "white"
    return f"color: {color}; font-weight: bold"


def fmt_money(x):
    if x is None or pd.isna(x):
        return ""
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.0f}K"
    return f"{sign}${a:.0f}"


# --- GitHub write-back: adds a ticker to tickers.json in the repo ---

def _gh_headers():
    token = st.secrets.get("GITHUB_TOKEN")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def add_ticker_to_watchlist(ticker, bin_size, mode):
    headers = _gh_headers()
    repo = st.secrets.get("GITHUB_REPO")
    if not headers or not repo:
        return False, "GitHub secrets not configured (GITHUB_TOKEN / GITHUB_REPO)."

    url = f"{GITHUB_API}/repos/{repo}/contents/tickers.json"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return False, f"Couldn't read tickers.json ({r.status_code}): {r.text[:200]}"

    file_data = r.json()
    sha = file_data["sha"]
    current = json.loads(base64.b64decode(file_data["content"]))

    if any(t["ticker"] == ticker for t in current["tickers"]):
        return False, f"{ticker} is already on the watchlist."

    current["tickers"].append({"ticker": ticker, "bin_size": bin_size, "mode": mode})
    new_content = json.dumps(current, indent=2) + "\n"

    put_body = {
        "message": f"Add {ticker} to watchlist",
        "content": base64.b64encode(new_content.encode()).decode(),
        "sha": sha,
    }
    r2 = requests.put(url, headers=headers, json=put_body)
    if r2.status_code not in (200, 201):
        return False, f"Couldn't save ({r2.status_code}): {r2.text[:200]}"
    return True, f"{ticker} added! It'll start showing up after the next scheduled run."


latest = load_latest()

tab_tracked, tab_lookup = st.tabs(["📊 Tracked Watchlist", "🔍 Look Up Any Ticker"])

with tab_tracked:
    if latest is None:
        st.warning(
            "No data yet. The GitHub Actions job hasn't run during market hours "
            "yet, or hasn't pushed data. Trigger it manually from the Actions "
            "tab (workflow_dispatch) to test."
        )
    else:
        tickers = list(latest.keys())
        selected = st.selectbox("Ticker", tickers)

        info = latest[selected]
        st.metric("Spot Price", f"${info['spot']:.2f}")

        rows = []
        for strike, vals in sorted(info["nodes"].items(), key=lambda kv: float(kv[0])):
            rows.append({
                "Strike": float(strike),
                "GEX": vals["GEX"],
                "VEX": vals["VEX"],
                "Δ 15m %": vals["pct_change_15m"],
                "Δ 1d %": vals["pct_change_1d"],
            })

        df = pd.DataFrame(rows).sort_values("Strike")

        st.dataframe(
            df.style
            .format({"GEX": fmt_money, "VEX": fmt_money, "Δ 15m %": "{:+.1f}%", "Δ 1d %": "{:+.1f}%"})
            .map(color_pct, subset=["Δ 15m %", "Δ 1d %"]),
            use_container_width=True,
            hide_index=True,
            height=600,
        )

        st.caption(
            "Refreshes automatically every 15 min during market hours "
            "(data pushed by GitHub Actions). Only tickers in tickers.json "
            "get this history — add a ticker there (and commit it) to start "
            "tracking it over time."
        )

with tab_lookup:
    st.caption(
        "Live snapshot only — no 15m/1d history, since this ticker isn't "
        "being tracked by the scheduled job. To get history, add it to "
        "tickers.json in the repo."
    )
    ticker_input = st.text_input("Ticker symbol", placeholder="e.g. NVDA").strip().upper()
    bin_size = st.number_input("Strike bin size", min_value=0.5, value=5.0, step=0.5)
    mode = st.radio("Expiration mode", ["daily", "weekly"], horizontal=True)

    if st.button("Fetch", type="primary") and ticker_input:
        with st.spinner(f"Fetching live option chain for {ticker_input}..."):
            try:
                df_raw, spot = fetch_heatseeker_data(ticker_input, mode=mode)
            except Exception as e:
                st.error(f"Error fetching {ticker_input}: {e}")
                df_raw, spot = None, None

        if df_raw is None:
            st.error(f"No data found for '{ticker_input}'. Check the symbol.")
        else:
            st.metric("Spot Price", f"${spot:.2f}")
            nodes = extract_nodes(df_raw, spot, bin_size=bin_size)
            nodes = nodes.reset_index().rename(columns={"index": "Strike", "strike_bin": "Strike"})
            st.dataframe(
                nodes[["Strike", "GEX", "VEX", "DEX", "CHARM"]].style
                .format({"GEX": fmt_money, "VEX": fmt_money, "DEX": fmt_money, "CHARM": fmt_money}),
                use_container_width=True,
                hide_index=True,
                height=600,
            )

            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(
                    f"Want {ticker_input} tracked every 15 min going forward "
                    "(with 15m/1d % change history)?"
                )
            with col2:
                if st.button(f"➕ Add {ticker_input} to Watchlist"):
                    ok, msg = add_ticker_to_watchlist(ticker_input, bin_size, mode)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
