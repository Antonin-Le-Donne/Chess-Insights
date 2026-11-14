"""Shared sidebar filters and game links for the dashboard pages (spec section 6).

The spec asks for sidebar filters (perf_type, date range, color, opening
family) and for charts that drill down to the game URL. Every page that
supports the filters calls ``sidebar_filtered`` right after loading the data;
pages whose metric would be distorted by filtering call
``render_filters_not_applied`` instead, so the sidebar always says which case
applies.

Selections are stored under ``filters_*`` keys in ``st.session_state`` so they
survive page changes: Streamlit drops a widget's own state when the widget is
not rendered on the next page, so each widget is re-seeded from the stored
value on every page.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from src.metrics import eco_to_family

OTHER_FAMILY = "Other"
PERF_TYPE_ORDER = ["blitz", "rapid", "bullet"]
COLOR_OPTIONS = ["both", "white", "black"]

_STATE_PREFIX = "filters_"
_WIDGET_PREFIX = "filters_widget_"


@dataclass(frozen=True)
class GameFilters:
    """Sidebar selection. An empty tuple means "no restriction"."""

    platforms: tuple[str, ...]
    perf_types: tuple[str, ...]
    date_from: date
    date_to: date
    color: str
    opening_families: tuple[str, ...]


def opening_family_labels(eco: pd.Series) -> pd.Series:
    """M5 opening family per game, with ECO codes outside the map as "Other"."""
    return eco.map(eco_to_family).fillna(OTHER_FAMILY)


def _persisted(widget: Callable[..., Any], label: str, name: str, default: Any, **kwargs: Any) -> Any:
    """Render a widget whose value survives page changes."""
    state_key = f"{_STATE_PREFIX}{name}"
    widget_key = f"{_WIDGET_PREFIX}{name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[state_key]
    value = widget(label, key=widget_key, **kwargs)
    st.session_state[state_key] = value
    return value


def _reset_filters() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith(_STATE_PREFIX):
            del st.session_state[key]


def render_sidebar_filters(games: pd.DataFrame) -> GameFilters:
    """Draw the sidebar filters and return the current selection."""
    st.sidebar.header("Filters")

    platforms = sorted(games["platform"].dropna().unique())
    present_perf = set(games["perf_type"].dropna().unique())
    perf_types = [p for p in PERF_TYPE_ORDER if p in present_perf] + sorted(
        present_perf - set(PERF_TYPE_ORDER)
    )
    families = sorted(
        opening_family_labels(games["eco"]).unique(),
        key=lambda family: (family == OTHER_FAMILY, family),
    )
    first_day = games["date_utc"].min().date()
    last_day = games["date_utc"].max().date()

    selected_platforms = _persisted(
        st.sidebar.multiselect, "Platform", "platforms", [],
        options=platforms, help="Leave empty for all platforms.",
    )
    selected_perf = _persisted(
        st.sidebar.multiselect, "Time control", "perf_types", [],
        options=perf_types,
        help="Leave empty for all. Bullet behaves very differently from blitz/rapid.",
    )

    # Clamp a stored range to the data bounds so date_input never rejects it.
    stored_range = st.session_state.get(f"{_STATE_PREFIX}dates", (first_day, last_day))
    if len(stored_range) == 2:
        stored_range = (
            min(max(stored_range[0], first_day), last_day),
            max(min(stored_range[1], last_day), first_day),
        )
        st.session_state[f"{_STATE_PREFIX}dates"] = stored_range
    date_range = _persisted(
        st.sidebar.date_input, "Date range", "dates", (first_day, last_day),
        min_value=first_day, max_value=last_day,
    )
    # While a range is being picked, date_input briefly returns one date.
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        date_from, date_to = date_range
    else:
        date_from, date_to = first_day, last_day

    color = _persisted(
        st.sidebar.radio, "My color", "color", "both",
        options=COLOR_OPTIONS, format_func=str.capitalize, horizontal=True,
    )
    selected_families = _persisted(
        st.sidebar.multiselect, "Opening family", "families", [],
        options=families,
        help='M5 families from the ECO code; everything else is "Other". Leave empty for all.',
    )

    st.sidebar.button("Reset filters", on_click=_reset_filters)

    return GameFilters(
        platforms=tuple(selected_platforms),
        perf_types=tuple(selected_perf),
        date_from=date_from,
        date_to=date_to,
        color=color,
        opening_families=tuple(selected_families),
    )


def apply_filters(
    games: pd.DataFrame,
    moves: pd.DataFrame,
    filters: GameFilters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the selected games and the moves of those games."""
    game_days = games["date_utc"].dt.date
    mask = (game_days >= filters.date_from) & (game_days <= filters.date_to)
    if filters.platforms:
        mask &= games["platform"].isin(filters.platforms)
    if filters.perf_types:
        mask &= games["perf_type"].isin(filters.perf_types)
    if filters.color != "both":
        mask &= games["my_color"].eq(filters.color)
    if filters.opening_families:
        mask &= opening_family_labels(games["eco"]).isin(filters.opening_families)

    filtered_games = games.loc[mask]
    filtered_moves = moves.loc[moves["game_id"].isin(filtered_games["game_id"])]
    return filtered_games, filtered_moves


def sidebar_filtered(
    games: pd.DataFrame,
    moves: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render the filters, apply them, report what is shown; stop if nothing is left."""
    filters = render_sidebar_filters(games)
    filtered_games, filtered_moves = apply_filters(games, moves, filters)

    if len(filtered_games) == len(games):
        st.caption(f"All {len(games):,} games — no sidebar filter active.")
    else:
        st.caption(
            f"Sidebar filters active: {len(filtered_games):,} of {len(games):,} games."
        )

    if filtered_games.empty:
        st.warning("No games match the sidebar filters.")
        st.stop()

    return filtered_games, filtered_moves


def render_filters_not_applied(reason: str) -> None:
    """Tell the reader that the sidebar filters are ignored on this page, and why."""
    st.sidebar.info(f"The sidebar filters do not apply to this page: {reason}")


def game_url(game_id: str, my_color: str | None = None, ply: int | None = None) -> str:
    """Public URL of a game, opened at a given ply on Lichess when one is given.

    Lichess supports board orientation and a ply anchor; chess.com live-game
    URLs do not take a move anchor, so they open at the start of the game.
    """
    platform, _, raw_id = str(game_id).partition("_")
    if platform == "lichess":
        url = f"https://lichess.org/{raw_id}"
        if my_color in ("white", "black"):
            url += f"/{my_color}"
        if ply is not None and not pd.isna(ply):
            url += f"#{int(ply)}"
        return url
    if platform == "chesscom":
        return f"https://www.chess.com/game/live/{raw_id}"
    raise ValueError(f"Unknown platform prefix in game_id {game_id!r}")


def game_urls(df: pd.DataFrame, ply_column: str | None = None) -> pd.Series:
    """Vectorised ``game_url`` over rows with game_id, my_color and an optional ply column."""
    plies = df[ply_column] if ply_column else pd.Series(None, index=df.index)
    return pd.Series(
        [
            game_url(game_id, color, ply)
            for game_id, color, ply in zip(df["game_id"], df["my_color"], plies)
        ],
        index=df.index,
        dtype="object",
    )


def format_time_control(base_seconds: int, increment_seconds: int) -> str:
    """180, 0 -> "3+0"; 30, 0 -> "0.5+0"."""
    minutes = base_seconds / 60
    minutes_text = f"{minutes:g}"
    return f"{minutes_text}+{int(increment_seconds)}"


def link_column(label: str = "Game") -> st.column_config.LinkColumn:
    """Column config that shows a short "Open" link instead of the raw URL."""
    return st.column_config.LinkColumn(label, display_text="Open")


def selected_points(event: Any) -> list[dict]:
    """Points selected on a ``st.plotly_chart(..., on_select="rerun")`` chart."""
    if not event:
        return []
    try:
        return list(event["selection"]["points"])
    except (KeyError, TypeError):
        return []
