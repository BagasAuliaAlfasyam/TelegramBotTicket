"""Utilities for parsing Ops messages from Telegram chats."""
from __future__ import annotations

import re
from typing import Collection, Dict, Optional

# Regex captures "solving text, APP -initials" with flexible whitespace.
_OPS_PATTERN = re.compile(
    r"^(?P<solving>.+?),\s*(?P<app>[A-Za-z]+)\s+(?P<initials>-[A-Za-z]+)$"
)


def parse_ops_message(text: str, allowed_apps: Collection[str]) -> Optional[Dict[str, str]]:
    """Parse an Ops reply following the "solving, APP -xx" pattern.

    Args:
        text: Raw message text from the Ops member.
        allowed_apps: Collection of app codes that are considered valid.

    Returns:
        Dict with keys "solving", "app", and "initials" if the text matches
        the expected structure and contains an allowed app; otherwise None.
    """

    if not text:
        return None

    normalized_apps = {app.upper() for app in allowed_apps}
    stripped_text = text.strip()
    match = _OPS_PATTERN.match(stripped_text)
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
