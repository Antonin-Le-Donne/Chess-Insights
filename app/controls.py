"""Shared display helpers for the opponent-control tables (M2, M3).

The reading rule is the one fixed in the spec ("M2/M3 — Opponent control
comparisons") before the intervals were computed: a 95% CI of the paired
difference that excludes 0 is side-specific, one that contains 0 is
compatible with a pattern shared at my level (not proof of equality).
"""

from __future__ import annotations

import pandas as pd

GROUP_LABELS = {"main": "Blitz / rapid", "bullet": "Bullet"}


def selection_key(games: pd.DataFrame) -> int:
    """Cheap cache key for a filtered selection: the set of game ids it holds.

    Hashing the full moves DataFrame on every rerun would cost seconds; the
    game ids identify the selection just as well.
    """
    return hash(tuple(sorted(games["game_id"].astype(str))))


def format_interval(value: float, lower: float, upper: float, pattern: str) -> str:
    """"value [lower, upper]" with one format pattern, or "—" when undefined."""
    if pd.isna(value):
        return "—"
    if pd.isna(lower) or pd.isna(upper):
        return pattern.format(value)
    return f"{pattern.format(value)} [{pattern.format(lower)}, {pattern.format(upper)}]"


def control_reading(lower: float, upper: float, higher_text: str, lower_text: str) -> str:
    """Apply the spec's reading rule to the paired-difference interval."""
    if pd.isna(lower) or pd.isna(upper):
        return "Not enough data"
    if lower > 0:
        return higher_text
    if upper < 0:
        return lower_text
    return "No detectable difference (shared at my level)"
