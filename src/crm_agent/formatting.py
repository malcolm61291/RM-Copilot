"""Formatting helpers shared across the package."""
from __future__ import annotations


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
