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
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

from core import fetch_heatseeker_data, extract_nodes, extract_matrix, compute_gamma_summary, fetch_intraday_candles

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "latest.json")
GITHUB_API = "https://api.github.com"
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

st.set_page_config(page_title="HeatSeeker", layout="wide")
st.title("🎯 HeatSeeker Node Tracker")

auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh every 5 min", value=True)
if auto_refresh:
    st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)
st.sidebar.caption(
    "Reloads the whole page every 5 min so new 15-min snapshots show up "
    "without you refreshing manually. Turn off if it's interrupting typing "
    "in the Lookup tab."
)


def load_latest():
    if not os.path.exists(DATA_PATH):
        return None
    with open(DATA_PATH) as f:
        return json.load(f)


def _num(v):
    """None (from JSON) -> real NaN, so pandas/format strings handle it
    gracefully instead of crashing on a bare Python None."""
    return float("nan") if v is None else v


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
    """Highlights only the Strike label cell for the row closest to spot —
    not the whole row — so the value cells keep their own gradient color
    instead of being overwritten white."""
    if df.empty or spot is None:
        return styler
    strikes = df[strike_col].to_numpy()
    closest = strikes[abs(strikes - spot).argmin()]

    def _row_style(row):
        styles = [""] * len(row)
        if row[strike_col] == closest:
            idx = list(row.index).index(strike_col)
            styles[idx] = "background-color: #FFFFFF; color: #000000; font-weight: bold; border: 2px solid #FFD700;"
        return styles

    return styler.apply(_row_style, axis=1)


def _format_cell(value, pct_15m):
    """Value with a compact 15m-change prefix, e.g. '+14% $22,046K'.
    Streamlit's dataframe can't render two differently-colored spans inside
    one cell, so this folds the badge into the same text instead of a true
    two-tone corner badge — same information, single cell, no new column."""
    if value is None or pd.isna(value):
        return ""
    val_str = fmt_money(value)
    if pct_15m is None or pd.isna(pct_15m):
        return val_str
    sign = "+" if pct_15m >= 0 else ""
    return f"{sign}{pct_15m:.0f}% {val_str}"


def is_monthly_opex(date_str):
    """Monthly OPEX = 3rd Friday of the month (the standard monthly options
    expiration). Works for any date string, regardless of daily/weekly mode."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return False
    return d.weekday() == 4 and 15 <= d.day <= 21


def render_matrix_grid_html(value_grid, pct_grid, spot, cmap_name="RdYlGn"):
    """
    Same data as render_matrix_grid but rendered as a real HTML table
    instead of a pandas Styler through st.dataframe. This is necessary for
    two things Streamlit's native dataframe genuinely can't do:
    - a true two-tone % change badge inside each cell (colored pill next
      to the value, not just prefixed text in one uniform color)
    - a king-node fill color that's visually distinct from the normal
      red/green gradient scale, so it doesn't just blend in as "very green"

    Scroll is preserved via a CSS overflow container with sticky header
    row + sticky Strike column.
    """
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    expirations = list(value_grid.columns)
    strikes = list(value_grid.index)
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        cmap = cm.get_cmap(cmap_name)

    col_vmax = {}
    for exp in expirations:
        v = value_grid[exp].abs().max()
        col_vmax[exp] = v if pd.notna(v) and v > 0 else 1

    # Find the single strongest long-gamma (max positive) and short-gamma
    # (max negative) cell across the whole visible grid.
    flat = value_grid.stack()
    long_king = flat.idxmax() if not flat.empty and flat.max() > 0 else None
    short_king = flat.idxmin() if not flat.empty and flat.min() < 0 else None

    strikes_arr = np.array(strikes, dtype=float)
    closest_strike = strikes_arr[np.abs(strikes_arr - spot).argmin()] if spot is not None else None

    def badge(pct):
        if pct is None or pd.isna(pct):
            return ""
        cls = "badge-pos" if pct >= 0 else "badge-neg"
        sign = "+" if pct >= 0 else ""
        return f'<span class="badge {cls}">{sign}{pct:.0f}%</span>'

    header_cells = ['<th class="strike-col">Strike</th>']
    for exp in expirations:
        cls = "opex-col" if is_monthly_opex(exp) else ""
        label = f"{exp} \U0001F3AF" if is_monthly_opex(exp) else exp
        header_cells.append(f'<th class="{cls}">{label}</th>')

    body_rows = []
    for strike in strikes:
        row_cells = []
        strike_cls = "strike-col"
        if closest_strike is not None and strike == closest_strike:
            strike_cls += " spot-cell"
        row_cells.append(f'<td class="{strike_cls}">{strike:g}</td>')

        for exp in expirations:
            v = value_grid.loc[strike, exp]
            p = pct_grid.loc[strike, exp]
            classes = []
            style = ""
            if is_monthly_opex(exp):
                classes.append("opex-col")

            if (strike, exp) == long_king:
                classes.append("king-long")
                prefix = "\U0001F451 "
            elif (strike, exp) == short_king:
                classes.append("king-short")
                prefix = "\U0001F451 "
            else:
                prefix = ""
                if pd.notna(v):
                    norm = mcolors.TwoSlopeNorm(vmin=-col_vmax[exp], vcenter=0, vmax=col_vmax[exp])
                    style = f"background-color: {mcolors.to_hex(cmap(norm(v)))}; color: #000000;"

            content = f"{prefix}{badge(p)} {fmt_money(v)}" if pd.notna(v) else ""
            row_cells.append(f'<td class="{" ".join(classes)}" style="{style}">{content}</td>')
        body_rows.append(f"<tr>{''.join(row_cells)}</tr>")

    html = f"""
    <style>
    .grid-wrap {{ overflow: auto; max-height: 620px; border: 1px solid #333; border-radius: 4px; }}
    table.gexgrid {{ border-collapse: collapse; font-family: sans-serif; font-size: 12px; white-space: nowrap; width: 100%; }}
    .gexgrid th, .gexgrid td {{ padding: 5px 10px; text-align: right; border: 1px solid #222; }}
    .gexgrid .strike-col {{ position: sticky; left: 0; background: #0E1117; z-index: 2; text-align: left; font-weight: bold; }}
    .gexgrid thead th {{ position: sticky; top: 0; background: #1A1D24; z-index: 3; color: #FAFAFA; }}
    .gexgrid thead th.strike-col {{ z-index: 4; }}
    .badge {{ display:inline-block; font-size:9px; padding:1px 5px; border-radius:8px; margin-right:4px; font-weight:bold; }}
    .badge-pos {{ background:#00C853; color:#000; }}
    .badge-neg {{ background:#FF3131; color:#fff; }}
    .king-long {{ background-color:#FFD700 !important; color:#000 !important; font-weight:bold; }}
    .king-short {{ background-color:#E040FB !important; color:#000 !important; font-weight:bold; }}
    .spot-cell {{ background-color:#FFFFFF !important; color:#000000 !important; font-weight:bold; border: 2px solid #FFD700 !important; }}
    .opex-col {{ border-left: 3px solid #FFD700 !important; border-right: 3px solid #FFD700 !important; }}
    </style>
    <div class="grid-wrap">
    <table class="gexgrid">
    <thead><tr>{''.join(header_cells)}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
    </table>
    </div>
    """
    return html


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


def render_gamma_header(nodes_dict, metric="GEX"):
    """Net total + wall strikes, shown as a metric strip like Bullflow's
    header (minus a flip point — see compute_gamma_summary's docstring for
    why that's deliberately not included yet)."""
    strikes = [float(s) for s in nodes_dict.keys()]
    values = [v[metric] for v in nodes_dict.values()]
    summary = compute_gamma_summary(strikes, values)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total Net {metric}", fmt_money(summary["net_gex"]))
    if summary["max_pos"]:
        c2.metric(f"Max +{metric} Wall", fmt_money(summary["max_pos"][1]), f"@ ${summary['max_pos'][0]:.0f}")
    else:
        c2.metric(f"Max +{metric} Wall", "—")
    if summary["max_neg"]:
        c3.metric(f"Max -{metric} Wall", fmt_money(summary["max_neg"][1]), f"@ ${summary['max_neg'][0]:.0f}")
    else:
        c3.metric(f"Max -{metric} Wall", "—")


def render_price_gex_overlay(ticker, nodes_dict, spot, metric="GEX", top_n=10, band_pct=0.05):
    """Intraday candlestick with the biggest GEX/VEX walls drawn as
    horizontal reference lines at their strike levels, so you can see price
    action relative to the walls on one chart. Candles are fetched live at
    render time (page load), not from the 15-min stored snapshots — so
    price is closer to real-time than the GEX levels are.

    Walls are restricted to a band around spot (band_pct) and the y-axis is
    pinned to the actual candle range — otherwise one huge-magnitude wall
    far from spot stretches the axis and squishes the real price action
    into a sliver, which is what was happening before this fix."""
    candles = fetch_intraday_candles(ticker)
    if candles is None:
        st.info(f"Couldn't fetch intraday price history for {ticker} right now.")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=candles.index, open=candles["Open"], high=candles["High"],
        low=candles["Low"], close=candles["Close"], name=ticker,
    )])

    # Pin the y-axis to the actual price action (candles + spot), with
    # some padding — set BEFORE adding wall lines so they can't stretch it.
    price_low = min(candles["Low"].min(), spot)
    price_high = max(candles["High"].max(), spot)
    pad = (price_high - price_low) * 0.15 or price_high * 0.01
    y_range = [price_low - pad, price_high + pad]

    # Only draw walls that are actually near spot (within band_pct), so a
    # massive far-dated/far-strike wall doesn't dominate the ranking or
    # clutter a chart where it's not even visible.
    lo, hi = spot * (1 - band_pct), spot * (1 + band_pct)
    strikes = [float(s) for s in nodes_dict.keys()]
    values = [v[metric] for v in nodes_dict.values()]
    nearby = [(s, v) for s, v in zip(strikes, values) if lo <= s <= hi]
    ranked = sorted(nearby, key=lambda sv: abs(sv[1]), reverse=True)[:top_n]

    # Line thickness scales with the wall's actual size, so the biggest
    # walls visually dominate instead of every line looking identical.
    # Only the top few get a text label — with 10 lines all annotated,
    # the labels overlap and Plotly silently truncates them (this is what
    # was showing as just "$1" before: cut-off text, not a real bug in the
    # numbers themselves).
    max_abs = max((abs(v) for _, v in ranked), default=1) or 1
    LABEL_COUNT = 4

    for i, (strike, val) in enumerate(ranked):
        color = "#00C853" if val > 0 else "#FF3131"
        label = "resistance" if val > 0 else "support"
        width = 1 + 7 * (abs(val) / max_abs)  # scales roughly 1px (smallest) to 8px (largest)

        show_label = i < LABEL_COUNT
        if show_label:
            fig.add_hline(
                y=strike, line_dash="solid", line_color=color, opacity=0.7,
                line_width=width,
                annotation_text=f"${strike:.0f} {label}: {fmt_money(val)}",
                annotation_position="right", annotation_font_size=10,
            )
        else:
            # No annotation kwargs at all here — passing annotation_text=None
            # with annotation_position still set doesn't suppress the label
            # the way you'd expect; Plotly falls back to a literal "new
            # text" placeholder instead of showing nothing. Omitting the
            # annotation arguments entirely is what actually shows no label.
            fig.add_hline(
                y=strike, line_dash="solid", line_color=color, opacity=0.7,
                line_width=width,
            )

    fig.add_hline(y=spot, line_dash="dash", line_color="white",
                   annotation_text=f"Spot ${spot:.2f}", annotation_position="left")

    fig.update_layout(
        title=f"{ticker} — Intraday Price with {metric} Walls",
        yaxis_title="Price", yaxis_range=y_range, xaxis_rangeslider_visible=False,
        height=550, margin=dict(l=10, r=160, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Top {top_n} {metric} walls by size overlaid as dashed lines "
        "(green = positive, red = negative). Price candles are fetched "
        "fresh on page load; wall levels are from the last 15-min snapshot."
    )


def build_gamma_summary_prompt(ticker, spot, nodes_dict, metric="GEX", top_n=6):
    """Builds a compact, fact-only text summary of the current snapshot for
    Gemini to narrate. Every number in this prompt comes straight from the
    stored data — the model is explicitly told not to invent or estimate
    anything beyond what's given, so it's narrating real numbers rather
    than guessing plausible-sounding ones."""
    strikes = [float(s) for s in nodes_dict.keys()]
    values = [v[metric] for v in nodes_dict.values()]
    summary = compute_gamma_summary(strikes, values)

    ranked = sorted(zip(strikes, values), key=lambda sv: abs(sv[1]), reverse=True)[:top_n]
    wall_lines = "\n".join(f"  - ${s:.0f}: {fmt_money(v)}" for s, v in ranked)

    net_str = fmt_money(summary["net_gex"])
    max_pos = f"${summary['max_pos'][0]:.0f} ({fmt_money(summary['max_pos'][1])})" if summary["max_pos"] else "none"
    max_neg = f"${summary['max_neg'][0]:.0f} ({fmt_money(summary['max_neg'][1])})" if summary["max_neg"] else "none"

    return f"""You are reading options {metric} (gamma exposure) data for {ticker}.
Current spot price: ${spot:.2f}
Total net {metric}: {net_str}
Strongest positive (long-gamma / resistance-leaning) wall: {max_pos}
Strongest negative (short-gamma / support-leaning) wall: {max_neg}

Top {top_n} strikes by magnitude:
{wall_lines}

Write a short (3-4 sentence) plain-English trading-desk-style read of this setup.
Cover: whether the overall positioning is net long or short gamma and what
that implies for volatility (dampened/choppy vs amplified/trending), and
where the nearest significant resistance and support walls are relative to
spot. Use ONLY the numbers given above — do not invent, estimate, or recall
any figure not explicitly provided. If you're unsure of something, say so
rather than guessing."""


def call_gemini(prompt, timeout=20):
    """Returns (success: bool, text_or_error: str). Never raises — any
    failure (missing key, network issue, bad response shape, rate limit)
    comes back as a clean message instead of crashing the app."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return False, "GEMINI_API_KEY not set in Streamlit secrets."

    try:
        r = requests.post(
            f"{GEMINI_API}?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Network error calling Gemini: {e}"

    if r.status_code != 200:
        return False, f"Gemini API error ({r.status_code}): {r.text[:300]}"

    try:
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return True, text.strip()
    except (KeyError, IndexError, TypeError):
        return False, f"Unexpected response shape from Gemini: {str(r.json())[:300]}"


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
if latest and latest.get("last_updated"):
    st.caption(f"📅 Last updated: {latest['last_updated']}")
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
        metric = st.radio("Metric", ["GEX", "VEX"], horizontal=True, key="market_metric")
        st.caption("Same-day expiration only. Refreshes every 15 min during market hours.")

        tickers = list(market_data.keys())
        cols = st.columns(len(tickers))

        for col, ticker in zip(cols, tickers):
            with col:
                info = market_data[ticker]
                st.subheader(display_names.get(ticker, ticker))
                st.metric("Spot", f"${info['spot']:.2f}")

                _strikes = [float(s) for s in info["nodes"].keys()]
                _vals = [v[metric] for v in info["nodes"].values()]
                _summary = compute_gamma_summary(_strikes, _vals)
                st.caption(f"Net {metric}: {fmt_money(_summary['net_gex'])}")

                rows = [
                    {"Strike": float(s), metric: v[metric], "Δ15m": _num(v["pct_change_15m"])}
                    for s, v in info["nodes"].items()
                ]
                df = pd.DataFrame(rows).sort_values("Strike", ascending=False)

                st.dataframe(
                    highlight_spot_row(gradient_style(df, [metric]), df, info["spot"])
                    .format({metric: fmt_money, "Δ15m": "{:+.1f}%"})
                    .map(color_pct, subset=["Δ15m"]),
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )

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
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.selectbox("Ticker", tickers, key="tracked_ticker")
        with col2:
            metric = st.radio("Metric", ["GEX", "VEX"], horizontal=True, key="tracked_metric")

        info = watchlist_data[selected]
        st.metric("Spot Price", f"${info['spot']:.2f}")
        render_gamma_header(info["nodes"], metric=metric)

        if "matrix" not in info:
            st.info("No per-expiration data stored for this ticker yet (next scheduled run will add it).")
        else:
            expirations = info["matrix"]["expirations"]
            metric_data = info["matrix"][metric]  # {strike: {expiration: cell}}

            # Build two aligned grids: numeric values (for color) and 15m %
            # change (for the corner badge). Rows = strike, cols = expiration.
            value_rows, pct_rows = {}, {}
            for strike_str, exps in metric_data.items():
                strike = float(strike_str)
                vrow, prow = {}, {}
                for exp in expirations:
                    cell = exps.get(exp)
                    if cell is None:
                        vrow[exp], prow[exp] = float("nan"), float("nan")
                    elif isinstance(cell, dict):
                        # Tolerate old stored shape (plain number) too, so
                        # stale repo data from before an update doesn't crash.
                        vrow[exp] = _num(cell.get("value"))
                        prow[exp] = _num(cell.get("pct_15m"))
                    else:
                        vrow[exp], prow[exp] = cell, float("nan")
                value_rows[strike] = vrow
                pct_rows[strike] = prow

            value_grid = pd.DataFrame(value_rows).T.reindex(columns=expirations).sort_index(ascending=False)
            pct_grid = pd.DataFrame(pct_rows).T.reindex(columns=expirations).sort_index(ascending=False)

            html = render_matrix_grid_html(value_grid, pct_grid, info["spot"])
            st.markdown(html, unsafe_allow_html=True)
            st.caption(
                "Strikes down, expirations across — scroll right for later "
                "dates, scroll up/down for other strikes. Colored badge = "
                "15-minute % change. Gold cell = spot's strike. Gold fill = "
                "strongest long-gamma node, purple fill = strongest "
                "short-gamma node, both marked with 👑."
            )

        st.divider()
        st.subheader("Price Chart with GEX Walls")
        render_price_gex_overlay(selected, info["nodes"], info["spot"], metric=metric)

        st.divider()
        st.subheader("🤖 AI Read")
        if st.button("Get AI Read", key="gemini_read_button"):
            with st.spinner("Asking Gemini..."):
                try:
                    prompt = build_gamma_summary_prompt(selected, info["spot"], info["nodes"], metric=metric)
                    ok, result = call_gemini(prompt)
                except Exception as e:
                    ok, result = False, f"Unexpected error building/sending the request: {e}"
            if ok:
                st.success(result)
            else:
                st.error(result)
        st.caption(
            "Reads the last stored snapshot (same data as the grid above), "
            "not a live fetch — if the market's closed, this describes the "
            "last available close. Only the exact numbers already shown in "
            "this app are given to the model; nothing is invented."
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
    mode = st.radio("Expiration mode", ["daily", "weekly", "all"], horizontal=True, index=1)
    max_exp = None
    if mode in ("daily", "weekly"):
        uncapped = st.checkbox("No cap (pull every matching expiration)", value=False)
        if not uncapped:
            max_exp = st.number_input("Max expirations", min_value=1, value=12, step=1)

    if st.button("Fetch", type="primary") and ticker_input:
        with st.spinner(f"Fetching live option chain for {ticker_input}..."):
            try:
                df_raw, spot = fetch_heatseeker_data(ticker_input, mode=mode, max_expirations=max_exp)
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

        matrix = extract_matrix(
            result["df_raw"], result["spot"], mode="GEX", bin_size=result["bin_size"],
            zoom_range=(result["df_raw"]["strike"].min(), result["df_raw"]["strike"].max()),
        )
        empty_pct = pd.DataFrame(float("nan"), index=matrix.index, columns=matrix.columns)
        html = render_matrix_grid_html(matrix, empty_pct, result["spot"])
        st.markdown(html, unsafe_allow_html=True)
        st.caption(
            "Strikes down, expirations across — scroll right for later dates, "
            "scroll up/down for other strikes. No 15m badge here (live "
            "one-off lookup, nothing to compare against yet). Gold cell = "
            "spot's strike. Gold fill = strongest long-gamma node, purple "
            "fill = strongest short-gamma node, both marked with 👑."
        )

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
