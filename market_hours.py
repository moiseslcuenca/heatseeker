"""
heatseeker/market_hours.py

Belt-and-suspenders guard: even though GitHub Actions cron is limited to
weekday windows, this double-checks US market hours + holidays so a stray
run (e.g. cron drift, manual trigger) is a safe no-op outside trading hours.
"""

from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pytz

NYSE = mcal.get_calendar("NYSE")
ET = pytz.timezone("America/New_York")

# --- Run windows (used by run_snapshot.py) ---------------------------------
# A run reaches the check ~30-60s after it is triggered, and triggers can
# jitter a little, so every window edge gets GRACE_MINUTES of slack.
PRE_OPEN_MINUTES = 15         # first window opens this long before the open
POST_CLOSE_MINUTES = 15       # last window closes this long after the close
POST_CLOSE_START_MINUTES = 5  # ignore the run that fires right at the bell
GRACE_MINUTES = 5


def is_market_open(now_utc=None):
    """
    Returns True only during regular NYSE trading hours on a trading day
    (i.e. no pre-market, no after-hours, no weekends, no holidays).
    """
    if now_utc is None:
        now_utc = datetime.now(pytz.utc)
    now_et = now_utc.astimezone(ET)

    schedule = NYSE.schedule(start_date=now_et.date(), end_date=now_et.date())
    if schedule.empty:
        return False  # holiday or weekend

    market_open = schedule.iloc[0]["market_open"].tz_convert(ET)
    market_close = schedule.iloc[0]["market_close"].tz_convert(ET)
    return market_open <= now_et <= market_close


def get_run_phase(now_utc=None):
    """
    Which run window this moment falls in, on a NYSE trading day:

      "open"  regular session (open <= now <= close)
      "pre"   from PRE_OPEN_MINUTES (+grace) before the open, up to the open
      "post"  from POST_CLOSE_START_MINUTES after the close, up to
              POST_CLOSE_MINUTES (+grace) after the close
      None    anything else: overnight, weekend, holiday, or the gaps between

    Uses the NYSE calendar, so holidays and early-close days (e.g. the day
    after Thanksgiving) shift the windows automatically.
    """
    if now_utc is None:
        now_utc = datetime.now(pytz.utc)
    now_et = now_utc.astimezone(ET)

    schedule = NYSE.schedule(start_date=now_et.date(), end_date=now_et.date())
    if schedule.empty:
        return None  # holiday or weekend

    market_open = schedule.iloc[0]["market_open"].tz_convert(ET)
    market_close = schedule.iloc[0]["market_close"].tz_convert(ET)

    if market_open <= now_et <= market_close:
        return "open"

    pre_start = market_open - timedelta(minutes=PRE_OPEN_MINUTES + GRACE_MINUTES)
    if pre_start <= now_et < market_open:
        return "pre"

    post_start = market_close + timedelta(minutes=POST_CLOSE_START_MINUTES)
    post_end = market_close + timedelta(minutes=POST_CLOSE_MINUTES + GRACE_MINUTES)
    if post_start < now_et <= post_end:
        return "post"

    return None


def is_last_run_of_day(now_utc=None, interval_minutes=15):
    """
    True if this run is the last scheduled 15-min run before today's close
    (used to tag the daily-close snapshot for day-over-day % change).
    """
    if now_utc is None:
        now_utc = datetime.now(pytz.utc)
    now_et = now_utc.astimezone(ET)

    schedule = NYSE.schedule(start_date=now_et.date(), end_date=now_et.date())
    if schedule.empty:
        return False

    market_close = schedule.iloc[0]["market_close"].tz_convert(ET)
    minutes_to_close = (market_close - now_et).total_seconds() / 60
    return 0 <= minutes_to_close < interval_minutes
