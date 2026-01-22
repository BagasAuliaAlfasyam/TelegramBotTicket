"""
SLA Calculation Utilities
==========================
Helper functions for SLA metrics computation.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def compute_sla(
    tech_dt: datetime | None, 
    response_dt: datetime | None, 
    tz: ZoneInfo
) -> tuple[str | float, str, str | float]:
    """
    Compute SLA metrics, pausing during configured break windows (12-13, 19-20 local).
    
    Args:
        tech_dt: Datetime when technician sent the message
        response_dt: Datetime when ops responded
        tz: Timezone for local time calculations
        
    Returns:
        Tuple of (response_minutes, status, remaining_minutes)
    """
    if not tech_dt or not response_dt:
        return "", "", ""

    if tech_dt > response_dt:
        tech_dt, response_dt = response_dt, tech_dt

    paused_minutes = _paused_minutes_between(tech_dt, response_dt, tz)
    active_seconds = max((response_dt - tech_dt).total_seconds() - (paused_minutes * 60), 0)
    minutes = round(active_seconds / 60, 2)
    remaining = round(max(15 - minutes, 0), 2)
    status = "OK" if minutes <= 15 else "TERLAMBAT"
    
    return minutes, status, remaining


def _paused_minutes_between(start_utc: datetime, end_utc: datetime, tz: ZoneInfo) -> float:
    """Calculate total paused minutes (breaks) between two UTC timestamps in local time."""
    start_local = start_utc.astimezone(tz)
    end_local = end_utc.astimezone(tz)
    if end_local <= start_local:
        return 0.0

    # Daily break windows (local time)
    breaks = [(12, 13), (19, 20)]

    total_seconds = 0.0
    current_date = start_local.date()
    end_date = end_local.date()

    while current_date <= end_date:
        for hour_start, hour_end in breaks:
            window_start = datetime.combine(
                current_date, 
                datetime.min.time(), 
                tz
            ).replace(hour=hour_start, minute=0, second=0, microsecond=0)
            window_end = datetime.combine(
                current_date, 
                datetime.min.time(), 
                tz
            ).replace(hour=hour_end, minute=0, second=0, microsecond=0)

            overlap_seconds = _overlap_seconds(start_local, end_local, window_start, window_end)
            total_seconds += overlap_seconds

        current_date = current_date.fromordinal(current_date.toordinal() + 1)

    return round(total_seconds / 60, 2)


def _overlap_seconds(
    a_start: datetime, 
    a_end: datetime, 
    b_start: datetime, 
    b_end: datetime
) -> float:
    """Return overlapping seconds between intervals [a_start, a_end] and [b_start, b_end]."""
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    delta = (earliest_end - latest_start).total_seconds()
    return max(delta, 0.0)
