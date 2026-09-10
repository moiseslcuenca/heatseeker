"""
heatseeker/app.py

Streamlit dashboard. Reads data/latest.json (committed by the GitHub Actions
job) and shows, per ticker, each node's GEX/VEX plus % change vs 15 min ago
and vs yesterday's close.

To run locally:  streamlit run app.py
On Streamlit Cloud: point the app at this file; it auto-redeploys whenever
the repo updates, which happens every 15 min thanks to the Actions commit.
"""

import json
import os

import pandas as pd
import streamlit as st

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "latest.json")

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


latest = load_latest()

if latest is None:
    st.warning(
        "No data yet. The GitHub Actions job hasn't run during market hours "
        "yet, or hasn't pushed data. Trigger it manually from the Actions "
        "tab (workflow_dispatch) to test."
    )
    st.stop()

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
    "(data pushed by GitHub Actions)."
)
