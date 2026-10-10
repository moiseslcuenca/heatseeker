"""
heatseeker/ai_read.py

The AI Read section: builds a fact sheet (facts.py), wraps it in strict
guardrails, asks Gemini to narrate it, then CHECKS the reply: every figure
in the answer must appear in the data sheet, otherwise the app says so.

Design rule (same as Skylit's Talon): the AI describes, it does not decide.
All structure (king, contested, floor, ceiling...) is computed in code.
"""

import math
import re
from datetime import datetime

import pytz
import streamlit as st

from facts import (TIMEFRAMES, TIMEFRAME_HELP, build_facts, render_facts_text,
                   matrices_from_nodes)

ET = pytz.timezone("America/New_York")

RULES = """You are an options-structure narrator for a GEX/VEX (dealer gamma / vanna exposure) tool. \
Below is a DATA SHEET computed by software. Your job is to describe it in plain English.

HARD RULES
1. You describe; you do not decide. Never give buy, sell, hold or exit calls, position sizing, price targets, \
or predictions of where price will go. If asked for any of these, say in one line that you only describe structure.
2. Use ONLY figures that appear in the DATA SHEET, copied exactly as written there. Do not calculate anything \
new (no sums, differences, ratios or percentages) and do not recall any figure from memory.
3. Take the sheet's labels as given (king, contested, stacked, floor, ceiling, gatekeeper). Do not re-derive them.
4. Positive nodes tend to dampen moves; negative nodes tend to amplify them. Use "potential" and "tends to" \
language, not certainty. Do not simply call positive nodes resistance and negative nodes support.
5. If the sheet says a level is "none" or there is no structure, say exactly that. An empty answer is a valid answer.
6. If the king is contested, say it could switch between reads.
7. Mention the snapshot time at most once, briefly (e.g. "As of 12:20 EDT"), and only if the requested format has room for it (the Brief format skips it unless the market is CLOSED). If the market is CLOSED, frame everything as context for the next session, never as live action.
7b. If the sheet says the king is far from spot, lead with the nearer floor and ceiling instead of the king.
8. Mention a data limit only when it matters to the point you are making (prior-day open interest, 15m changes \
being mostly repricing, no flow data).
9. Last line, exactly: Educational, not financial advice."""

STYLES = {
    "Brief": (
        "Format: BRIEF. Write at most 4 short lines (each under 25 words), no headings, no labels:\n"
        "1. Net positioning and what it implies (dampened vs amplified moves).\n"
        "2. The king: where it sits versus spot, and whether it is contested.\n"
        "3. The nearest floor and ceiling versus spot. If the gatekeeper is the same node as the floor or ceiling, "
        "do not repeat it. Mention the stacked king only if it is YES.\n"
        "4. Only if the sheet lists movers: the single biggest mover. Otherwise leave this line out.\n"
        "Do not add a 'what would change' line, a snapshot-time line (unless the market is CLOSED), or data-limit notes."
    ),
    "Detailed": (
        "Format: AT MOST 6 short lines (each under 30 words), each starting with a label and a colon. Use only these "
        "labels, in this order, and skip any the sheet has nothing for:\n"
        "Net positioning / King (say whether it is contested) / Floor and ceiling / Stacked (only if YES) / "
        "Movers (the biggest one or two) / What would change this picture (structure only, e.g. the king switching "
        "or price moving through a level)."
    ),
    "Prose": (
        "Format: one short paragraph of 4-6 sentences in plain English, covering net positioning, the king and "
        "whether it is contested, where the nearest floor and ceiling sit relative to spot, and any notable movers."
    ),
}


def _md(text):
    """Make a model reply safe and readable in Streamlit Markdown: escape $ (a
    pair of dollar signs is treated as LaTeX and mangles the text) and turn
    single newlines into hard line breaks (Markdown otherwise collapses a
    multi-line reply into one dense paragraph)."""
    return text.replace("\r\n", "\n").replace("$", "\\$").replace("\n", "  \n")


def build_prompt(facts_text, style="Brief"):
    return f"{RULES}\n\n{STYLES.get(style, STYLES['Brief'])}\n\n{facts_text}"


# ------------------------------------------------------ figure checker

_STRIP = [re.compile(p, re.I) for p in (
    r"\d{4}-\d{2}-\d{2}",            # ISO dates (expirations)
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b",  # clock times
    r"(?<!\$)\b20[2-3]\d\b",          # years (a $-prefixed number is a strike, kept)
    r"S&P\s*500", r"Nasdaq[- ]?100",
    r"\b\d+\s?DTE\b",
)]
_NUM = re.compile(
    r"(?<![\w.])[-+]?(\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(billion|million|thousand|%|[KMBkmb](?![A-Za-z]))?",
    re.I)
_MULT = {"billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
_ALLOWED_CONSTANTS = {5.0, 10.0, 15.0, 25.0}  # thresholds named in the rules / sheet


def extract_figures(text):
    """Returns [(raw_text, magnitude, marked, scaled)] for numbers in text.
    `marked` = carried a $, %, or K/M/B suffix; `scaled` = carried a K/M/B
    (or million/billion/thousand) suffix, i.e. a rounded display figure."""
    for pat in _STRIP:
        text = pat.sub(" ", text)
    out = []
    for m in _NUM.finditer(text):
        dollar, whole, frac, suffix = m.group(1), m.group(2), m.group(3), (m.group(4) or "").lower()
        try:
            val = float(whole.replace(",", "") + (frac or ""))
        except ValueError:
            continue
        if suffix in _MULT:
            val *= _MULT[suffix]
        out.append((m.group(0).strip(), val, bool(dollar or suffix), suffix in _MULT))
    return out


def check_figures(reply, facts_text, rounding_tol=0.01):
    """Figures in the reply that do not appear in the data sheet. Ignores
    plain small integers (counts like '3 expirations'), dates, times, and the
    named thresholds. Signs are not checked (magnitudes only).

    Matching is deliberately strict where it matters: strikes and prices
    (no K/M/B suffix) must match a sheet figure EXACTLY, because neighbouring
    strikes are within 1% of each other and a tolerance would let a wrong
    strike through. Only abbreviated amounts ('$850M', '$1.2 billion') get a
    1% rounding tolerance, so '$850M' matches '$849.6M'."""
    sheet_vals = [v for _, v, _, _ in extract_figures(facts_text)]
    bad, seen = [], set()
    for raw, val, marked, scaled in extract_figures(reply):
        if val == 0 or val in _ALLOWED_CONSTANTS:
            continue
        if not marked and val <= 10:
            continue
        if scaled:
            ok = any(math.isclose(val, s, rel_tol=rounding_tol) for s in sheet_vals)
        else:
            ok = any(abs(val - s) < 0.005 for s in sheet_vals)
        if ok:
            continue
        if raw not in seen:
            seen.add(raw)
            bad.append(raw)
    return bad


# ------------------------------------------------------------------ glue

def build_for_ticker(ticker, info, metric, timeframe, market_open, snapshot_label=None, live_spot=None, today=None):
    """info = one ticker's entry from latest.json['watchlist']. Returns (facts, text)."""
    matrix = info.get("matrix")
    if matrix and matrix.get("expirations"):
        matrices = {"GEX": matrix.get("GEX", {}), "VEX": matrix.get("VEX", {})}
        expirations = matrix["expirations"]
    else:
        matrices, expirations = matrices_from_nodes(info.get("nodes", {}))
    facts = build_facts(ticker, metric, timeframe, matrices, expirations, info["spot"],
                        live_spot=live_spot, snapshot_label=snapshot_label,
                        market_open=market_open, today=today or _today())
    return facts, render_facts_text(facts)


def _today():
    return datetime.now(ET).date()


def _live_spot(ticker):
    try:
        from core import fetch_live_quote
        return fetch_live_quote(ticker)["price"]
    except Exception:
        return None


def render_ai_read_section(selected, info, metric, market_open, call_gemini, snapshot_label=None, key_prefix="ai"):
    """Drop-in replacement for the whole AI Read block in the tracked tab."""
    st.subheader("🤖 AI Read")
    c1, c2 = st.columns(2)
    timeframe = c1.selectbox("Timeframe", TIMEFRAMES, index=1, key=f"{key_prefix}_timeframe")
    style = c2.radio("Style", list(STYLES), horizontal=True, key=f"{key_prefix}_style")
    st.caption(TIMEFRAME_HELP[timeframe])
    if not (info.get("matrix") or {}).get("expirations"):
        st.caption("No per-expiration data stored yet for this ticker, so the read uses all expirations combined.")

    label = "Get AI Read" if market_open else "Get AI Read (next-session preview)"
    if st.button(label, key=f"{key_prefix}_button"):
        with st.spinner("Asking Gemini..."):
            try:
                _, text = build_for_ticker(selected, info, metric, timeframe, market_open,
                                           snapshot_label=snapshot_label, live_spot=_live_spot(selected))
                ok, result = call_gemini(build_prompt(text, style))
            except Exception as e:
                text, ok, result = "", False, f"Unexpected error building/sending the request: {e}"
        if ok:
            # Escape $ so Streamlit's Markdown doesn't treat pairs as LaTeX.
            st.success(_md(result))
            unmatched = check_figures(result, text)
            if unmatched:
                st.warning("⚠️ These figures in the read were not found in the data sheet, so don't rely on them: "
                           + ", ".join(unmatched).replace("$", "\\$"))
            else:
                st.caption("✅ Every figure in this read matches the data sheet.")
            with st.expander("Data the AI was given"):
                st.code(text, language=None)
        else:
            st.error(result)
    st.caption(
        "The levels (king, floor, ceiling...) are computed by the app, not the AI; Gemini only describes them. "
        "It describes structure and never gives trade calls. Educational, not financial advice."
    )
