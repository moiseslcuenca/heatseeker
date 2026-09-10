"""
heatseeker/market_hours.py

Belt-and-suspenders guard: even though GitHub Actions cron is limited to
weekday windows, this double-checks US market hours + holidays so a stray
run (e.g. cron drift, manual trigger) is a safe no-op outside trading hours.
"""

from datetime import datetime
import pandas_market_calendars as mcal
import pytz

NYSE = mcal.get_calendar("NYSE")
ET = pytz.timezone("America/New_York")


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
