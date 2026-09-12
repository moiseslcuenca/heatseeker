"""
heatseeker/app.py

Streamlit dashboard, three tabs:
- "Market": SPX / SPY / QQQ, same-day expiration only, GEX/VEX toggle.
- "Tracked Watchlist": SPY/QQQ (all expirations) + anything added via the
  lookup tab. Shows the 15m/1d % change table for the aggregated node.
- "Look Up Any Ticker": live on-demand fetch for any symbol, full GEX
  heatmap (strike x expiration), with a button to add it to the tracked
  watchlist (writes to tickers.json in the GitHub repo via the API; the
  scheduled job picks it up next run).
"""

import base64
import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from core import fetch_heatseeker_data, extract_nodes, extract_matrix

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


def gradient_style(df, cols):
    """Diverging red/green background, centered at 0, matching the heatmap."""
    styler = df.style
    for col in cols:
        vmax = df[col].abs().max() or 1
        styler = styler.background_gradient(
            subset=[col], cmap="RdYlGn", vmin=-vmax, vmax=vmax
        )
    return styler


def highlight_spot_row(styler, df, spot, strike_col="Strike"):
    """Outlines the row whose strike is closest to the current spot price,
    so it's obvious at a glance where price is standing right now."""
    if df.empty or spot is None:
        return styler
    strikes = df[strike_col].to_numpy()
    closest = strikes[abs(strikes - spot).argmin()]

    def _row_style(row):
        if row[strike_col] == closest:
            return ["border-top: 2px solid #FFD700; border-bottom: 2px solid #FFD700; font-weight: bold"] * len(row)
        return [""] * len(row)

    return styler.apply(_row_style, axis=1)


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


def fmt_money_short(x):
    if x is None or pd.isna(x):
        return ""
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e6:
        return f"{sign}${a/1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.0f}K"
    return f"{sign}${a:.0f}"


def render_heatmap(matrix, spot, title):
    """
    matrix: DataFrame, index=strike (descending), columns=expiration dates,
    values=GEX/VEX. Renders a Plotly heatmap: diverging color scale, king
    node (max positive / max negative) starred.
    """
    if matrix.empty:
        st.info("Not enough data to draw a heatmap for this range.")
        return

    z = matrix.values
    x_labels = [str(c) for c in matrix.columns]
    y_values = [float(s) for s in matrix.index]  # numeric, so add_hline works

    flat_vals = pd.DataFrame(z).values.flatten()
    flat_vals = flat_vals[~pd.isna(flat_vals)]
    max_abs = max(abs(flat_vals.min()), abs(flat_vals.max()), 1) if len(flat_vals) else 1

    text = [[fmt_money_short(v) if pd.notna(v) else "" for v in row] for row in z]

    flat = pd.DataFrame(z, index=matrix.index, columns=matrix.columns).stack()
    long_king = flat.idxmax() if not flat.empty and flat.max() > 0 else None
    short_king = flat.idxmin() if not flat.empty and flat.min() < 0 else None

    for coords, label in [(long_king, "\U0001F451L"), (short_king, "\U0001F451S")]:
        if coords is None:
            continue
        r = list(matrix.index).index(coords[0])
        c = list(matrix.columns).index(coords[1])
        text[r][c] = f"{label} {text[r][c]}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_values,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title="Exposure"),
            hoverongaps=False,
        )
    )

    fig.add_hline(
        y=spot,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Spot: ${spot:.2f}",
        annotation_position="right",
    )

    fig.update_layout(
        title=title,
        yaxis_title="Strike",
        xaxis_title="Expiration",
        yaxis=dict(type="linear"),
        height=max(500, 32 * len(y_values)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("\U0001F451L = long-gamma king node · \U0001F451S = short-gamma king node")


def _gh_headers():
    token = st.secrets.get("GITHUB_TOKEN")
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "heatseeker-streamlit-app",
    }


def add_ticker_to_watchlist(ticker, bin_size, mode="all"):
    """Adds ticker to the 'watchlist' group in tickers.json (SPY/QQQ track
    all expirations by default there; new tickers do too, matching them)."""
    headers = _gh_headers()
    repo = st.secrets.get("GITHUB_REPO")
    if not headers or not repo:
        return False, "GitHub secrets not configured (GITHUB_TOKEN / GITHUB_REPO)."

    url = f"{GITHUB_API}/repos/{repo}/contents/tickers.json"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return False, f"Couldn't read tickers.json ({r.status_code}): {r.text[:300]}"

    file_data = r.json()
    sha = file_data["sha"]
    current = json.loads(base64.b64decode(file_data["content"]))
    current.setdefault("watchlist", [])

    if any(t["ticker"] == ticker for t in current["watchlist"]):
        return False, f"{ticker} is already on the watchlist."

    current["watchlist"].append({"ticker": ticker, "bin_size": bin_size, "mode": mode})
    new_content = json.dumps(current, indent=2) + "\n"

    put_body = {
        "message": f"Add {ticker} to watchlist",
        "content": base64.b64encode(new_content.encode()).decode(),
        "sha": sha,
    }
    r2 = requests.put(url, headers=headers, json=put_body)
    if r2.status_code not in (200, 201):
        return False, f"Couldn't save ({r2.status_code}): {r2.text[:300]}"
    return True, f"{ticker} added! It'll show up after the next scheduled run (within 15 min, during market hours)."


latest = load_latest()
market_data = (latest or {}).get("market", {})
watchlist_data = (latest or {}).get("watchlist", {})

tab_market, tab_tracked, tab_lookup = st.tabs(
    ["🏛️ Market", "📊 Tracked Watchlist", "🔍 Look Up Any Ticker"]
)

# ============================== MARKET TAB ==============================
with tab_market:
    if not market_data:
        st.warning(
            "No market data yet. The GitHub Actions job hasn't run during "
            "market hours yet, or hasn't pushed data."
        )
    else:
        display_names = {"^SPX": "SPX"}
        tickers = list(market_data.keys())
        labels = [display_names.get(t, t) for t in tickers]
        label_to_ticker = dict(zip(labels, tickers))

        col1, col2 = st.columns([2, 1])
        with col1:
            selected_label = st.selectbox("Index / Ticker", labels, key="market_ticker")
        with col2:
            metric = st.radio("Metric", ["GEX", "VEX"], horizontal=True, key="market_metric")

        selected = label_to_ticker[selected_label]
        info = market_data[selected]
        st.metric("Spot Price", f"${info['spot']:.2f}")

        rows = [{"Strike": float(strike), metric: vals[metric]} for strike, vals in info["nodes"].items()]
        df = pd.DataFrame(rows).sort_values("Strike", ascending=False)

        st.dataframe(
            highlight_spot_row(gradient_style(df, [metric]), df, info["spot"]).format({metric: fmt_money}),
            use_container_width=True,
            hide_index=True,
            height=500,
        )
        st.caption("Same-day expiration only. Refreshes every 15 min during market hours.")

# ============================== TRACKED TAB ==============================
with tab_tracked:
    if not watchlist_data:
        st.warning(
            "No data yet. The GitHub Actions job hasn't run during market hours "
            "yet, or hasn't pushed data. Trigger it manually from the Actions "
            "tab (workflow_dispatch) to test."
        )
    else:
        tickers = list(watchlist_data.keys())
        selected = st.selectbox("Ticker", tickers, key="tracked_ticker")

        info = watchlist_data[selected]
        st.metric("Spot Price", f"${info['spot']:.2f}")

        rows = []
        for strike, vals in info["nodes"].items():
            rows.append({
                "Strike": float(strike),
                "GEX": vals["GEX"],
                "VEX": vals["VEX"],
                "Δ 15m %": vals["pct_change_15m"],
                "Δ 1d %": vals["pct_change_1d"],
            })
        df = pd.DataFrame(rows).sort_values("Strike", ascending=False)

        st.dataframe(
            highlight_spot_row(gradient_style(df, ["GEX", "VEX"]), df, info["spot"])
            .format({"GEX": fmt_money, "VEX": fmt_money, "Δ 15m %": "{:+.1f}%", "Δ 1d %": "{:+.1f}%"})
            .map(color_pct, subset=["Δ 15m %", "Δ 1d %"]),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        st.caption(
            "SPY and QQQ here track ALL expirations (unlike the Market tab, "
            "which is same-day only). This table is aggregated across "
            "expirations — that's what carries the 15m/1d history. For a "
            "full per-expiration heatmap of this ticker, use the lookup tab."
        )

# ============================== LOOKUP TAB ==============================
with tab_lookup:
    st.caption(
        "Live snapshot only — no 15m/1d history, since this ticker isn't "
        "being tracked by the scheduled job. Add it to the watchlist below "
        "to start tracking it (all expirations)."
    )
    ticker_input = st.text_input("Ticker symbol", placeholder="e.g. NVDA").strip().upper()
    bin_size = st.number_input("Strike bin size", min_value=0.5, value=5.0, step=0.5)
    mode = st.radio("Expiration mode", ["daily", "weekly", "all"], horizontal=True, index=2)

    if st.button("Fetch", type="primary") and ticker_input:
        with st.spinner(f"Fetching live option chain for {ticker_input}..."):
            try:
                df_raw, spot = fetch_heatseeker_data(ticker_input, mode=mode)
            except Exception as e:
                st.session_state["lookup_result"] = None
                st.error(f"Error fetching {ticker_input}: {e}")
                df_raw, spot = None, None

        if df_raw is None:
            st.session_state["lookup_result"] = None
            st.error(f"No data found for '{ticker_input}'. Check the symbol.")
        else:
            st.session_state["lookup_result"] = {
                "ticker": ticker_input,
                "spot": spot,
                "df_raw": df_raw,
                "bin_size": bin_size,
                "mode": mode,
            }

    result = st.session_state.get("lookup_result")
    if result:
        st.metric("Spot Price", f"${result['spot']:.2f}")

        matrix = extract_matrix(result["df_raw"], result["spot"], mode="GEX", bin_size=result["bin_size"])
        render_heatmap(matrix, result["spot"], f"{result['ticker']} — GEX by strike & expiration")

        nodes = extract_nodes(result["df_raw"], result["spot"], bin_size=result["bin_size"])
        nodes = nodes.reset_index().rename(columns={"strike_bin": "Strike"})
        nodes = nodes.sort_values("Strike", ascending=False)[["Strike", "GEX", "VEX"]]
        st.dataframe(
            highlight_spot_row(gradient_style(nodes, ["GEX", "VEX"]), nodes, result["spot"])
            .format({"GEX": fmt_money, "VEX": fmt_money}),
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(
                f"Want {result['ticker']} tracked every 15 min going forward "
                "(with 15m/1d % change history, all expirations)?"
            )
        with col2:
            if st.button(f"➕ Add {result['ticker']} to Watchlist"):
                ok, msg = add_ticker_to_watchlist(result["ticker"], result["bin_size"], result["mode"])
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
