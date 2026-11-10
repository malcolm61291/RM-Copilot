"""Formatting helpers shared across the package."""
from __future__ import annotations

import re

_RANGE_DASH = re.compile(r"(\d)\s*[–—]\s*(\d)")
_LEADING_DASH = re.compile(r"^[ \t]*[–—][ \t]*", re.MULTILINE)
_DASH = re.compile(r"\s*[–—]\s*")


def plain_dashes(text: str) -> str:
    """House style for anything shown to the RM or a customer: no em or en dashes.

    "21–58" becomes "21 to 58", a dash opening a line is dropped, and any other dash
    becomes a comma. Applied to model output and RM-typed text, which we don't control.
    """
    text = _RANGE_DASH.sub(r"\1 to \2", text)
    text = _LEADING_DASH.sub("", text)
    return _DASH.sub(", ", text)


def inr(x: float) -> str:
    """Indian digit grouping: 1234567 -> 12,34,567"""
    s = f"{int(round(x))}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])
