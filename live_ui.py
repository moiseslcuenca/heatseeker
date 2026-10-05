"""
heatseeker/live_ui.py

Auto-refreshing price widgets for the Streamlit app.

These use st.fragment(run_every=...), which re-runs ONLY the fragment on a
timer — the rest of the page (tabs, grids, selectbox state) is untouched,
so nothing flickers and no session state is lost.

- While the market is closed the timer is switched off (run_every=None), so
  an idle open tab doesn't keep polling Yahoo all weekend.
- Prices come from core.fetch_live_quote, which is cached and shared across
  viewers, so more open browsers don't mean more Yahoo requests.
- If a refresh fails, the last good price stays on screen, clearly marked
  stale, instead of the number disappearing.

GEX/VEX levels are NOT live — they are still the last 15-min snapshot. Only
the price is live. The snapshot spot is shown beside it so the two aren't
confused.
"""

from datetime import datetime, timezone

import pytz
import streamlit as st

from core import fetch_live_quote

LIVE_SPOT_SECONDS = 15
LIVE_CHART_SECONDS = 60
ET = pytz.timezone("America/New_York")


def _market_open_safe():
    """If the market-hours check itself breaks, assume open: a wasted poll
    is better than a frozen price."""
    try:
        from market_hours import is_market_open
        return is_market_open()
    except Exception:
        return True


def _now_et_str():
    return datetime.now(timezone.utc).astimezone(ET).strftime("%H:%M:%S")


def _spot_body(ticker, label, snapshot_spot, key, market_open):
    quote = fetch_live_quote(ticker)
    state_key = f"live_last_{key}"

    if quote["price"] is not None:
        st.session_state[state_key] = (quote["price"], quote["prev_close"], _now_et_str())
        stale = False
    else:
        stale = True

    last = st.session_state.get(state_key)

    if last is None:
        # Never got a live price this session: show the snapshot value and say why.
        st.metric(label, f"${snapshot_spot:.2f}")
        st.caption(f"⚠️ Live price unavailable, showing snapshot. {quote['error']}")
        return

    price, prev_close, as_of = last
    delta = None
    if prev_close:
        chg = price - prev_close
        delta = f"{chg:+.2f} ({chg / prev_close * 100:+.2f}%)"
    st.metric(label, f"${price:,.2f}", delta)

    if stale:
        st.caption(f"⚠️ Stale since {as_of} ET, refresh failing: {quote['error']}")
    elif market_open:
        st.caption(f"🟢 Live · {as_of} ET · levels computed at ${snapshot_spot:.2f}")
    else:
        st.caption(f"Market closed · last price {as_of} ET · levels computed at ${snapshot_spot:.2f}")


def render_live_spot(ticker, label, snapshot_spot, key=None):
    """Drop-in replacement for st.metric('Spot', f'${spot}')."""
    market_open = _market_open_safe()
    run_every = LIVE_SPOT_SECONDS if market_open else None
    st.fragment(run_every=run_every)(_spot_body)(ticker, label, snapshot_spot, key or ticker, market_open)


def _chart_body(render_fn, ticker, nodes, snapshot_spot, metric):
    quote = fetch_live_quote(ticker)
    spot = quote["price"] if quote["price"] is not None else snapshot_spot
    render_fn(ticker, nodes, spot, metric=metric)


def render_live_chart(render_fn, ticker, nodes, snapshot_spot, metric):
    """Runs render_fn(ticker, nodes, spot, metric=metric) inside a fragment
    that re-runs every minute with the live spot, so the candles and the
    spot line keep moving. Wall levels stay at their last snapshot values."""
    run_every = LIVE_CHART_SECONDS if _market_open_safe() else None
    st.fragment(run_every=run_every)(_chart_body)(render_fn, ticker, nodes, snapshot_spot, metric)
