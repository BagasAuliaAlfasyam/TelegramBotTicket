"""
SLA Calculation Utilities
==========================

Helper functions untuk menghitung metrik SLA tiket.

SLA dihitung dengan aturan:
    - Target response time: 15 menit
    - Jam istirahat dikurangi: 12:00-13:00, 19:00-20:00
    - Status OK jika response <= 15 menit, TERLAMBAT jika lebih

Author: Bagas Aulia Alfasyam
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
    Hitung metrik SLA dengan memperhitungkan jam istirahat.
    
    Jam istirahat (pause) yang dikecualikan dari perhitungan:
        - 12:00 - 13:00 (makan siang)
        - 19:00 - 20:00 (makan malam)
    
    Args:
        tech_dt: Datetime saat teknisi mengirim pesan
        response_dt: Datetime saat ops merespons
        tz: Timezone untuk kalkulasi waktu lokal
    
    Returns:
        Tuple (response_minutes, status, remaining_minutes):
            - response_minutes: Waktu respons dalam menit (sudah dikurangi jam istirahat)
            - status: "OK" jika <= 15 menit, "TERLAMBAT" jika lebih
            - remaining_minutes: Sisa waktu SLA (15 - response_minutes, min 0)
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
    """
    Hitung total menit jam istirahat antara dua timestamp.
    
    Mengiterasi setiap hari dari start sampai end dan menghitung
    berapa lama overlap dengan jam istirahat (12-13, 19-20).
    
    Args:
        start_utc: Waktu mulai (UTC)
        end_utc: Waktu selesai (UTC)
        tz: Timezone untuk konversi ke waktu lokal
    
    Returns:
        float: Total menit yang masuk jam istirahat
    """
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
    """
    Hitung detik overlap antara dua interval waktu.
    
    Args:
        a_start, a_end: Interval pertama [a_start, a_end]
        b_start, b_end: Interval kedua [b_start, b_end]
    
    Returns:
        float: Jumlah detik overlap, 0 jika tidak overlap
    """
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    delta = (earliest_end - latest_start).total_seconds()
    return max(delta, 0.0)
