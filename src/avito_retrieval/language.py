"""Fast language-script diagnostics for Russian-dominant marketplace text.

This intentionally detects writing system rather than pretending to distinguish
Russian from closely related Cyrillic languages on very short search queries.
"""

from __future__ import annotations

import re

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")


def dominant_script(text: object, threshold: float = 0.60) -> str:
    """Return cyrillic/latin/mixed/other using alphabetic script share."""
    value = "" if text is None else str(text)
    cyrillic = len(CYRILLIC_RE.findall(value))
    latin = len(LATIN_RE.findall(value))
    total = cyrillic + latin
    if total < 2:
        return "other"
    if cyrillic / total >= threshold:
        return "cyrillic"
    if latin / total >= threshold:
        return "latin"
    return "mixed"

