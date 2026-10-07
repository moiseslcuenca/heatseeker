"""
heatseeker/facts.py

Computes the *structure* of a GEX/VEX board in plain Python, so the AI never
has to. Everything an AI read says about levels (king, contested, stacked,
floor, ceiling, gatekeeper, movers) is decided here, deterministically and
testably; Gemini only narrates the resulting data sheet.

No Streamlit / network imports: this module is pure computation.

Definitions (modelled on Skylit's public Heatseeker guide):
- Node: one strike x expiration cell.
- King: the cell with the largest absolute value among the selected
  expirations.
- Contested king: the king is less than 10% larger than the runner-up.
- Stacked king: the GEX king and the VEX king are the same strike AND the
  same expiration.
- Large node: at least 25% of the king's absolute size.
- Floor / Ceiling: nearest large node below / above spot.
- Gatekeeper: nearest large node strictly between spot and the king.
- Positive nodes tend to dampen moves; negative nodes tend to amplify them.
"""

import math
from datetime import date, datetime

LARGE_FRACTION = 0.25
CONTESTED_MARGIN = 0.10
MOVER_MIN_FRACTION = 0.10
SWING_DAYS = 14
TOP_NODES = 5
TIMEFRAMES = ("Scalp", "Day", "Swing", "Position")
TIMEFRAME_HELP = {
    "Scalp": "Nearest expiration only.",
    "Day": "Nearest 3 expirations.",
    "Swing": f"Expirations within {SWING_DAYS} days.",
    "Position": "Monthly OPEX (3rd Friday) expirations only.",
}

NAN = float("nan")


def fmt_money(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
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


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def _valid(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _to_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def is_monthly_opex(date_str):
    try:
        d = _to_date(date_str)
    except (ValueError, TypeError):
        return False
    return d.weekday() == 4 and 15 <= d.day <= 21


# ----------------------------------------------------------------- inputs

def select_expirations(expirations, timeframe, today):
    """Returns (selected_expirations, note_or_None). Expired dates are
    dropped; each timeframe falls back to something sensible, with a note
    saying so, instead of returning nothing."""
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    parsed = []
    for e in sorted(set(expirations)):
        try:
            parsed.append((e, _to_date(e)))
        except (ValueError, TypeError):
            continue
    future = [x for x in parsed if x[1] >= today]
    note = None
    if not parsed and expirations:
        # No real dates at all, e.g. the aggregated-nodes fallback uses a single
        # pseudo-expiration called "all". Use it as-is; timeframes can't apply.
        return sorted(set(expirations)), "no per-expiration data stored; all expirations are combined, so the timeframe does not apply"
    if not future:
        if not parsed:
            return [], "no usable expirations stored"
        future = parsed
        note = "all stored expirations are in the past; using them as stored"

    if timeframe == "Scalp":
        sel = future[:1]
    elif timeframe == "Day":
        sel = future[:3]
    elif timeframe == "Swing":
        sel = [x for x in future if (x[1] - today).days <= SWING_DAYS]
        if not sel:
            sel = future[:1]
            note = f"nothing within {SWING_DAYS} days; using the nearest expiration"
    else:  # Position
        sel = [x for x in future if is_monthly_opex(x[0])]
        if not sel:
            sel = future
            note = "no monthly OPEX expirations stored; using all stored expirations"
    return [e for e, _ in sel], note


def parse_cells(metric_matrix, expirations):
    """metric_matrix: {strike_str: {expiration: cell}} exactly as stored in
    latest.json. A cell may be a dict {"value", "pct_15m", ...} (current), a
    plain number (older stored shape), or None. Zero / NaN / missing cells
    are dropped."""
    cells = []
    for strike_str, exps in (metric_matrix or {}).items():
        strike = _num(strike_str)
        if not _valid(strike) or not isinstance(exps, dict):
            continue
        for exp in expirations:
            cell = exps.get(exp)
            if cell is None:
                continue
            if isinstance(cell, dict):
                v, p = _num(cell.get("value")), _num(cell.get("pct_15m"))
            else:
                v, p = _num(cell), NAN
            if not _valid(v) or v == 0:
                continue
            cells.append({"strike": strike, "exp": exp, "value": v, "pct15": p})
    return cells


def matrices_from_nodes(nodes):
    """Fallback when a ticker has no stored per-expiration matrix: treat the
    aggregated nodes as one pseudo-expiration called 'all'. 15m % is left
    unknown, because the stored node % isn't labelled per metric."""
    out = {"GEX": {}, "VEX": {}}
    for s, v in (nodes or {}).items():
        for m in out:
            if m in v:
                out[m][str(s)] = {"all": {"value": v[m], "pct_15m": None}}
    return out, ["all"]


# ------------------------------------------------------------- structure

def _king_info(cells):
    if not cells:
        return None
    ranked = sorted(cells, key=lambda c: abs(c["value"]), reverse=True)
    king = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin, contested = None, False
    if second is not None:
        s = abs(second["value"])
        if s > 0:
            margin = (abs(king["value"]) - s) / s
            contested = margin < CONTESTED_MARGIN
    return {"king": king, "runner_up": second, "margin": margin, "contested": contested}


def _kind(value):
    return "positive node, tends to dampen moves" if value > 0 else "negative node, tends to amplify moves"


def build_facts(ticker, metric, timeframe, matrices, expirations, snapshot_spot,
                live_spot=None, snapshot_label=None, market_open=True, today=None):
    """Returns a plain dict of everything the AI read may talk about."""
    if metric not in ("GEX", "VEX"):
        raise ValueError(f"unknown metric {metric!r}")
    today = today or date.today()
    exps, note = select_expirations(expirations, timeframe, today)

    using_live = _valid(live_spot) and live_spot > 0
    spot = float(live_spot) if using_live else float(snapshot_spot)

    cells = parse_cells(matrices.get(metric), exps)
    other = "VEX" if metric == "GEX" else "GEX"
    other_cells = parse_cells(matrices.get(other), exps)

    facts = {
        "ticker": ticker, "metric": metric, "timeframe": timeframe,
        "expirations": exps, "expiration_note": note,
        "snapshot_label": snapshot_label, "market_open": bool(market_open),
        "spot": spot, "snapshot_spot": float(snapshot_spot), "spot_is_live": using_live,
        "net": float(sum(c["value"] for c in cells)),
        "node_count": len(cells),
    }

    ki = _king_info(cells)
    facts["king_info"] = ki
    if ki is None:
        facts.update(floor=None, ceiling=None, gatekeeper=None, stacked=None,
                     growing=[], shrinking=[], top_nodes=[], other_king=None)
        return facts

    king = ki["king"]
    king_abs = abs(king["value"])
    large = [c for c in cells if abs(c["value"]) >= LARGE_FRACTION * king_abs]

    def pick_below():
        c = [x for x in large if x["strike"] < spot]
        return max(c, key=lambda x: (x["strike"], abs(x["value"]))) if c else None

    def pick_above():
        c = [x for x in large if x["strike"] > spot]
        return min(c, key=lambda x: (x["strike"], -abs(x["value"]))) if c else None

    facts["floor"], facts["ceiling"] = pick_below(), pick_above()

    gate = None
    lo, hi = sorted((spot, king["strike"]))
    between = [x for x in large if lo < x["strike"] < hi]
    if between:
        gate = (min(between, key=lambda x: (x["strike"], -abs(x["value"]))) if king["strike"] > spot
                else max(between, key=lambda x: (x["strike"], abs(x["value"]))))
    facts["gatekeeper"] = gate

    oki = _king_info(other_cells)
    facts["other_king"] = oki["king"] if oki else None
    if oki is None:
        facts["stacked"] = None  # unknown
    else:
        k2 = oki["king"]
        gk, vk = (king, k2) if metric == "GEX" else (k2, king)
        facts["stacked"] = (gk["strike"] == vk["strike"] and gk["exp"] == vk["exp"])

    movers = [c for c in cells if abs(c["value"]) >= MOVER_MIN_FRACTION * king_abs and _valid(c["pct15"])]
    facts["growing"] = sorted([c for c in movers if c["pct15"] > 0], key=lambda c: c["pct15"], reverse=True)[:3]
    facts["shrinking"] = sorted([c for c in movers if c["pct15"] < 0], key=lambda c: c["pct15"])[:3]
    facts["top_nodes"] = sorted(cells, key=lambda c: abs(c["value"]), reverse=True)[:TOP_NODES]
    return facts


# ------------------------------------------------------------------ text

def _dist(strike, spot):
    return (strike - spot) / spot * 100.0


def _node_txt(c, spot):
    return (f"${c['strike']:g} exp {c['exp']}, {fmt_money(c['value'])} "
            f"({_kind(c['value'])}), {_dist(c['strike'], spot):+.2f}% from spot")


def render_facts_text(f):
    spot = f["spot"]
    L = []
    L.append(f"DATA SHEET: {f['ticker']} | metric: {f['metric']} | timeframe: {f['timeframe']}")
    exp_txt = ", ".join(f["expirations"]) if f["expirations"] else "none"
    L.append(f"Expirations included: {exp_txt}" + (f" (note: {f['expiration_note']})" if f["expiration_note"] else ""))
    mk = "OPEN" if f["market_open"] else "CLOSED (this is the last snapshot, not live action)"
    L.append(f"Snapshot taken: {f['snapshot_label'] or 'unknown time'} | Market: {mk}")
    if f["spot_is_live"]:
        L.append(f"Spot (live price): ${spot:,.2f}. Levels were computed at snapshot spot ${f['snapshot_spot']:,.2f}.")
    else:
        L.append(f"Spot (snapshot price): ${spot:,.2f}")

    if f["king_info"] is None:
        L.append("No non-zero nodes in the included expirations. There is no structure to describe.")
        return "\n".join(L)

    net = f["net"]
    if f["metric"] == "GEX":
        tend = "positive: dampening tendency" if net > 0 else "negative: amplifying tendency" if net < 0 else "flat"
    else:
        tend = "positive net vanna" if net > 0 else "negative net vanna" if net < 0 else "flat"
    L.append(f"Net {f['metric']} across included expirations: {fmt_money(net)} ({tend})")

    ki = f["king_info"]
    k = ki["king"]
    L.append(f"KING NODE: {_node_txt(k, spot)}")
    ru = ki["runner_up"]
    if ru is None or ki["margin"] is None:
        L.append("Contested king: NO (there is no runner-up node)")
    else:
        word = "YES" if ki["contested"] else "NO"
        tail = "; the king could switch between reads" if ki["contested"] else ""
        L.append(f"Contested king: {word}, king is {ki['margin']*100:.0f}% larger than the runner-up "
                 f"(${ru['strike']:g} exp {ru['exp']}, {fmt_money(ru['value'])}){tail}")

    if f["stacked"] is None:
        L.append("Stacked king (GEX king and VEX king on the same strike and expiry): unknown, other metric unavailable")
    elif f["stacked"]:
        L.append(f"Stacked king: YES, GEX and VEX kings are both at ${k['strike']:g} exp {k['exp']}")
    else:
        ok = f["other_king"]
        L.append(f"Stacked king: NO (the other metric's king is ${ok['strike']:g} exp {ok['exp']})")

    L.append(f"Large nodes are those at least {LARGE_FRACTION*100:.0f}% of the king's size.")
    for name, key in (("Floor (nearest large node below spot)", "floor"),
                      ("Ceiling (nearest large node above spot)", "ceiling"),
                      ("Gatekeeper (large node between spot and the king)", "gatekeeper")):
        L.append(f"{name}: " + (_node_txt(f[key], spot) if f[key] else "none"))

    for name, key in (("Fastest growing nodes, last 15m", "growing"), ("Fastest shrinking nodes, last 15m", "shrinking")):
        if f[key]:
            L.append(f"{name}: " + "; ".join(
                f"${c['strike']:g} exp {c['exp']} {c['pct15']:+.0f}% (now {fmt_money(c['value'])})" for c in f[key]))
        else:
            L.append(f"{name}: none")
    L.append("Note on 15m changes: open interest is prior-day, so these mostly reflect repricing, not new positioning.")

    L.append(f"Top {len(f['top_nodes'])} nodes by size:")
    for c in f["top_nodes"]:
        L.append(f"  - {_node_txt(c, spot)}")

    L.append("Known data limits: open interest is prior-day (not intraday); no options-flow or dark-pool data; "
             "call/put sign convention (calls positive, puts negative) is an assumption.")
    return "\n".join(L)
