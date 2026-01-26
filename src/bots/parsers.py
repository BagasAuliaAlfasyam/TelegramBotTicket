"""
Message Parsers
================

Utilitas untuk parsing pesan Ops dari chat Telegram.

Format pesan Ops yang diharapkan:
    "solving text, APP -initials"
    
Contoh:
    "Done reset MFA, MIT -bg"
    "Done Reset password Default, MIS -dvd"

Author: Bagas Aulia Alfasyam
"""
from __future__ import annotations

import re
from typing import Collection, Optional

# Regex captures "solving text, APP -initials" with flexible whitespace.
_OPS_PATTERN = re.compile(
    r"^(?P<solving>.+?),\s*(?P<app>[A-Za-z]+)\s+(?P<initials>-[A-Za-z]+)$"
)


def parse_ops_message(
    text: str, 
    allowed_apps: Collection[str]
) -> Optional[dict[str, str]]:
    """
    Parse reply Ops dengan format "solving, APP -xx".
    
    Fungsi ini mengekstrak informasi dari pesan Ops yang mengikuti
    format standar: solving text diikuti kode aplikasi dan inisial.
    
    Args:
        text: Teks mentah dari pesan member Ops
        allowed_apps: Collection kode aplikasi yang valid (MIT, MIS, dll)
    
    Returns:
        Dict dengan keys "solving", "app", "initials" jika match,
        None jika format tidak sesuai atau app tidak diizinkan.
    
    Contoh:
        >>> parse_ops_message("Reset password, MIT -bg", ["MIT", "MIS"])
        {"solving": "Reset password", "app": "MIT", "initials": "-bg"}
    """
    if not text:
        return None

    normalized_apps = {app.upper() for app in allowed_apps}
    stripped_text = text.strip()

    # Normalize any newline/extra whitespace into single spaces so multiline replies still match.
    normalized_text = " ".join(stripped_text.split())

    match = _OPS_PATTERN.match(normalized_text)
    if not match:
        return None

    app_code = match.group("app").upper()
    if app_code not in normalized_apps:
        return None

    solving = match.group("solving").strip()
    initials = match.group("initials").strip()
    if not solving or not initials:
        return None

    return {
        "solving": solving,
        "app": app_code,
        "initials": initials,
    }
