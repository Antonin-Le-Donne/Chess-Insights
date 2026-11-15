"""M4 — Time management (spec section 5, M4 + "M4 — Time-pressure onset"),
laid out with the dashboard principles (spec section 6): the answer first,
one main chart, details in expanders.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.controls import (
    GROUP_LABELS,
    control_reading,
    format_interval,
    render_badge,
    selection_key,
    verdict_color,
)
from app.filters import sidebar_filtered
from app.ui import load_ui
from src.metrics import (
    M4_MIN_CROSSING_GAMES_FOR_MEDIAN_CI,
    m4_clock_burn_map,
    m4_onset_by_game,
    m4_onset_cif,
    m4_onset_control,
    m4_reference_peak,
)

ME_COLOR = "#C7F36B"
OPPONENT_COLOR = "#8F9893"
PEAK_COLOR = "#FF6B6B"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Inter", size=14),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    return games, moves


@st.cache_data
def load_time_pressure_pct() -> float:
    """Primary threshold: config.toml is its single source (spec section 4)."""
    with open("config.toml", "rb") as f:
        return float(tomllib.load(f)["thresholds"]["time_pressure_pct"])


@st.cache_data(show_spinner="Comparing with my opponents (1,000 game resamples, about a minute)…")
def cached_m4_control(
    _moves: pd.DataFrame,
    _games: pd.DataFrame,
    key: int,
    time_pressure_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    # Underscored arguments are not hashed; `key` identifies the selection.
    onset = m4_onset_by_game(_moves, _games, time_pressure_pct=time_pressure_pct)
    return (
        m4_onset_control(onset),
        m4_onset_cif(onset),
        m4_reference_peak(_moves, _games),
    )


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def peak_sentence(median_onset: float, median_ci_lower: float, peak: float) -> str:
    """Spec rule: lower CI bound of my median onset vs the end of the M1 peak."""
    if pd.isna(peak):
        return "My blunder peak is undefined for this selection, so the timing cannot be compared."
    if pd.isna(median_ci_lower):
        return (
            f"Too few games reach time trouble (fewer than {M4_MIN_CROSSING_GAMES_FOR_MEDIAN_CI} "
            "on one side) to compare its timing with my blunder peak."
        )
    if median_ci_lower > peak:
        return (
            f"When it happens, it starts around move **{median_onset:.0f}** — after my blunder "
            f"peak (moves {peak - 4:.0f}–{peak:.0f}), so the clock does not explain those blunders."
        )
    return (
        f"When it happens, it can start by my blunder peak (moves {peak - 4:.0f}–{peak:.0f}): "
        "time trouble may contribute to those blunders (not proof of a cause)."
    )


def render_answer(mine: pd.Series, theirs: pd.Series, peak: float, group_label: str, threshold_pct: float) -> None:
    reading = control_reading(
        mine["difference_ci_lower"],
        mine["difference_ci_upper"],
        "I get into time trouble more often than my opponents",
        "I get into time trouble less often than my opponents",
    )
    render_badge(
        reading,
        verdict_color(mine["difference_ci_lower"], mine["difference_ci_upper"], good_when_negative=True),
    )
    st.markdown(
        f"#### In {group_label.lower()}, my clock drops below {threshold_pct:.0%} of the starting "
        f"time in **{mine['proportion']:.0%}** of my games, against **{theirs['proportion']:.0%}** "
        f"for my opponents ({100 * mine['proportion_difference_me_minus_opponent']:+.1f} points, "
        f"95% interval {100 * mine['difference_ci_lower']:+.1f} to "
        f"{100 * mine['difference_ci_upper']:+.1f}). "
        + peak_sentence(mine["median_onset_move"], mine["median_ci_lower"], peak)
    )


def render_cif(cif: pd.DataFrame, peak: float) -> None:
    """Main chart: share of games already in time trouble, by move."""
    rows = cif.copy()
    rows["Side"] = rows["side"].map({"me": "Me", "opponent": "My opponents"})
    rows["error_plus"] = rows["ci_upper"] - rows["cif"]
    rows["error_minus"] = rows["cif"] - rows["ci_lower"]
    fig = px.line(
        rows,
        x="move_number",
        y="cif",
        color="Side",
        markers=True,
        error_y="error_plus",
        error_y_minus="error_minus",
        color_discrete_map={"Me": ME_COLOR, "My opponents": OPPONENT_COLOR},
        labels={"move_number": "Move", "cif": "Games already in time trouble", "Side": ""},
    )
    if pd.notna(peak):
        fig.add_vrect(
            x0=peak - 4, x1=peak, fillcolor=PEAK_COLOR, opacity=0.15, line_width=0,
            annotation_text="my blunder peak", annotation_position="top left",
        )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis=dict(tickformat=".0%", rangemode="tozero", gridcolor="rgba(255,255,255,0.07)"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Share of all games in which the player has already been in time trouble by that move "
        "(games that ended earlier still count). Measured at the moves fixed in the spec, with "
        "95% intervals. The red band is my blunder peak (page M1)."
    )


def render_numbers(mine: pd.Series, theirs: pd.Series) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Side": label,
                    "Games in time trouble [95% CI]": format_interval(
                        row["proportion"], row["ci_lower"], row["ci_upper"], "{:.1%}"
                    ),
                    "Games": f"{int(row['n_crossed']):,} / {int(row['n_games']):,}",
                    "Median first move in time trouble [95% CI]": format_interval(
                        row["median_onset_move"], row["median_ci_lower"], row["median_ci_upper"], "{:.0f}"
                    ),
                }
                for label, row in (("Me", mine), ("My opponents", theirs))
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_burn_map(moves: pd.DataFrame, games: pd.DataFrame, group: str) -> None:
    data = m4_clock_burn_map(moves, games).reset_index()
    data = data[data["perf_group"] == group]
    tab1, tab2 = st.tabs(["Time per move", "Clock left"])
    with tab1:
        fig = px.line(
            data,
            x="move_number",
            y=["median_time_spent", "p90_time_spent"],
            labels={"move_number": "Move", "value": "Seconds", "variable": ""},
        )
        fig.update_layout(**PLOT_LAYOUT, height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with tab2:
        fig = px.line(
            data,
            x="move_number",
            y="median_clock_after",
            labels={"move_number": "Move", "median_clock_after": "Median seconds left"},
        )
        fig.update_layout(**PLOT_LAYOUT, height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Medians over the games still running at each move: late moves only contain long "
        "games, so these curves describe my pace, not when time trouble starts."
    )


def main() -> None:
    st.title("M4 — Time trouble")
    st.markdown("**Question:** do I get into time trouble more than players at my level — and is it what causes my blunders?")

    games, moves = sidebar_filtered(*load_data())
    threshold_pct = load_time_pressure_pct()
    control, cif, peaks = cached_m4_control(moves, games, selection_key(games), threshold_pct)

    groups = [g for g in ("main", "bullet") if control["perf_group"].eq(g).any()]
    if not groups:
        st.info("No eligible game in the current selection.")
        return
    group = st.radio(
        "Time control",
        groups,
        format_func=lambda g: GROUP_LABELS.get(g, g),
        horizontal=True,
        key="m4_group",
        label_visibility="collapsed",
    )
    rows = control[control["perf_group"].eq(group)]
    mine = rows[rows["side"].eq("me")].iloc[0]
    theirs = rows[rows["side"].eq("opponent")].iloc[0]
    peak = peaks.get(group, float("nan"))

    render_answer(mine, theirs, peak, GROUP_LABELS[group], threshold_pct)
    render_cif(cif[cif["perf_group"].eq(group)], peak)

    with st.expander("The numbers"):
        render_numbers(mine, theirs)
    with st.expander("How I spend my time, move by move"):
        render_burn_map(moves, games, group)
    with st.expander("How this is computed"):
        st.markdown(
            f"""
            - **Time trouble:** the player's clock after a move falls below {threshold_pct:.0%}
              of the starting time (`config.toml`). Games without a starting time (e.g. 0+1)
              are excluded.
            - **First move in time trouble:** the first such move, per player and per game.
            - **My opponents (control group):** the same rule on their clock, in the same games.
            - **The confirmatory result** (fixed in the spec before computing) is the
              difference in the share of games reaching time trouble, with a paired bootstrap
              interval (1,000 resamples of whole games). Everything else is exploratory.
            - **Blunder peak:** computed from my own M1 curve (games of 40+ half-moves,
              5-move buckets with at least 100 of my moves); if the lower bound of my median
              first move in time trouble is later than the end of that bucket, the clock cannot
              explain the peak by itself.
            - **Limit:** reaching time trouble less often does not prove better clock
              management: I may simply play faster, or my opponents may spend their time
              trying to beat my defence.
            """
        )


if __name__ == "__main__":
    main()
