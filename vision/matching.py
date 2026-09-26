"""Label comparison used when merging views and when matching the clinic list."""

from __future__ import annotations

from rapidfuzz import fuzz


def token_ratio(left: str, right: str) -> float:
    """rapidfuzz token ratio, 0 to 100."""
    return float(fuzz.token_ratio(left, right))
