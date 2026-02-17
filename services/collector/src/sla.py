"""
SLA Calculation (microservice copy — zero deps, preserved as-is)
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def compute_sla(tech_dt: datetime | None, response_dt: datetime | None, tz: ZoneInfo):
    if not tech_dt or not response_dt:
        return "", "", ""
    if tech_dt > response_dt:
        tech_dt, response_dt = response_dt, tech_dt
    paused = _paused_minutes_between(tech_dt, response_dt, tz)
    active_seconds = max((response_dt - tech_dt).total_seconds() - (paused * 60), 0)
    minutes = round(active_seconds / 60, 2)
    remaining = round(max(15 - minutes, 0), 2)
    status = "OK" if minutes <= 15 else "TERLAMBAT"
    return minutes, status, remaining


def _paused_minutes_between(start_utc, end_utc, tz):
    start_local = start_utc.astimezone(tz)
    end_local = end_utc.astimezone(tz)
    if end_local <= start_local:
        return 0.0
    breaks = [(12, 13), (19, 20)]
    total_seconds = 0.0
    current_date = start_local.date()
    end_date = end_local.date()
    while current_date <= end_date:
        for h_start, h_end in breaks:
            ws = datetime.combine(current_date, datetime.min.time(), tz).replace(
                hour=h_start, minute=0, second=0, microsecond=0)
            we = datetime.combine(current_date, datetime.min.time(), tz).replace(
                hour=h_end, minute=0, second=0, microsecond=0)
            total_seconds += _overlap_seconds(start_local, end_local, ws, we)
        from datetime import timedelta
        current_date += timedelta(days=1)
    return total_seconds / 60.0


def _overlap_seconds(a_start, a_end, b_start, b_end):
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_start >= overlap_end:
        return 0.0
    return (overlap_end - overlap_start).total_seconds()
