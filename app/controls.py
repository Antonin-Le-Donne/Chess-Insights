"""Shared display helpers for the opponent-control tables (M2, M3).

The reading rule is the one fixed in the spec ("M2/M3 — Opponent control
comparisons") before the intervals were computed: a 95% CI of the paired
difference that excludes 0 is side-specific, one that contains 0 is
compatible with a pattern shared at my level (not proof of equality).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

GROUP_LABELS = {"main": "Blitz / rapid", "bullet": "Bullet"}

BADGE_GREEN = "#7BE0A3"
BADGE_RED = "#FF6B6B"
BADGE_GRAY = "#B7C2CC"
BADGE_BLUE = "#7CC4FF"


def render_badge(text: str, color: str) -> None:
    """Small coloured verdict label shown above a page's answer sentence."""
    st.markdown(
        f"<span style='color:{color};font-size:0.8rem;font-weight:600;letter-spacing:0.04em;"
        f"text-transform:uppercase'>● {text}</span>",
        unsafe_allow_html=True,
    )


def verdict_color(lower: float, upper: float, good_when_negative: bool) -> str:
    """Green if the difference interval is on the favourable side of 0, red if on
    the unfavourable side, gray if it contains 0 (spec reading rule)."""
    if pd.isna(lower) or pd.isna(upper) or lower <= 0 <= upper:
        return BADGE_GRAY
    favourable = upper < 0 if good_when_negative else lower > 0
    return BADGE_GREEN if favourable else BADGE_RED


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
