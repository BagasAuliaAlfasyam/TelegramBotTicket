"""
Message Parsers (microservice copy — zero deps, preserved as-is)
"""
from __future__ import annotations

import re
from collections.abc import Collection
from typing import Optional

_OPS_PATTERN = re.compile(
    r"^(?P<solving>.+?),\s*(?P<app>[A-Za-z]+)\s+(?P<initials>-[A-Za-z]+)$"
)


def parse_ops_message(text: str, allowed_apps: Collection[str]) -> dict[str, str] | None:
    if not text:
        return None
    normalized_apps = {app.upper() for app in allowed_apps}
    normalized_text = " ".join(text.strip().split())
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
    return {"solving": solving, "app": app_code, "initials": initials}
