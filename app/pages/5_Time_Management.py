from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.controls import GROUP_LABELS, control_reading, format_interval, selection_key
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

ME_COLOR = "#2E7DFF"
OPPONENT_COLOR = "#F4BE55"
PEAK_COLOR = "#F87171"


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


@st.cache_data(show_spinner="Bootstrapping 1,000 game resamples (about a minute)…")
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


st.set_page_config(
    page_title="Chess Insights",
    layout="wide",
)

load_ui()

PLOTLY_LAYOUT = go.Layout(
    font=dict(family="Inter", size=14),
    hovermode="x unified",
    template="plotly_white",
    margin=dict(l=40, r=40, t=60, b=40),
)


def peak_reading(median_ci_lower: float, peak: float) -> str:
    """Spec rule: compare the lower CI bound of my median onset with the end
    of the data-derived M1 peak bucket."""
    if pd.isna(peak):
        return "M1 peak undefined (no 5-move bucket with at least 100 of my moves): rule not applied."
    if pd.isna(median_ci_lower):
        return (
            f"Fewer than {M4_MIN_CROSSING_GAMES_FOR_MEDIAN_CI} games reach time pressure "
            "on one side: rule not applied."
        )
    if median_ci_lower > peak:
        return (
            f"Time pressure starts after the M1 blunder peak (move {peak:.0f}): it cannot "
            "explain that peak by itself."
        )
    return (
        f"Time pressure can start by the M1 blunder peak (move {peak:.0f}): it may "
        "contribute to it (not proof of a cause)."
    )


def render_control(
    control: pd.DataFrame,
    peak: float,
    perf_group: str,
    threshold_pct: float,
) -> None:
    """Me vs opponents, same games: share of games reaching time pressure."""
    rows = control.loc[control["perf_group"].eq(perf_group)]
    if rows.empty:
        st.info("No eligible game in the current selection for this time control.")
        return
    mine = rows.loc[rows["side"].eq("me")].iloc[0]
    theirs = rows.loc[rows["side"].eq("opponent")].iloc[0]

    table = pd.DataFrame(
        [
            {
                "Side": label,
                "Games reaching time pressure [95% CI]": format_interval(
                    row["proportion"], row["ci_lower"], row["ci_upper"], "{:.1%}"
                ),
                "Games": f"{int(row['n_crossed']):,} / {int(row['n_games']):,}",
                "Median onset move [95% CI]": format_interval(
                    row["median_onset_move"],
                    row["median_ci_lower"],
                    row["median_ci_upper"],
                    "{:.0f}",
                ),
            }
            for label, row in (("Me", mine), ("My opponents", theirs))
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    difference = format_interval(
        100 * mine["proportion_difference_me_minus_opponent"],
        100 * mine["difference_ci_lower"],
        100 * mine["difference_ci_upper"],
        "{:+.1f}",
    ) + " pts"
    reading = control_reading(
        mine["difference_ci_lower"],
        mine["difference_ci_upper"],
        "I reach time pressure in MORE of my games than my opponents",
        "I reach time pressure in FEWER of my games than my opponents",
    )
    st.markdown(f"**Difference me − opponents:** {difference} → **{reading}**")
    st.markdown(f"**Against the blunder peak:** {peak_reading(mine['median_ci_lower'], peak)}")
    st.caption(
        f"Time pressure = clock after a move below {threshold_pct:.0%} of the starting "
        "time (config.toml); onset = the first such move, per player and per game. "
        "Same games for both sides; intervals from 1,000 game resamples, the difference "
        "by paired bootstrap. The share of games is the spec's single confirmatory "
        "estimate; the median onset (among games that reach time pressure) is "
        "exploratory, and its interval is shown only with at least "
        f"{M4_MIN_CROSSING_GAMES_FOR_MEDIAN_CI} such games on each side."
    )


def render_cif(cif: pd.DataFrame, peak: float, perf_group: str) -> None:
    """Share of games that have reached time pressure by fixed moves."""
    rows = cif.loc[cif["perf_group"].eq(perf_group)].copy()
    if rows.empty:
        return
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
        labels={
            "move_number": "Move number",
            "cif": "Games already in time pressure",
        },
    )
    if pd.notna(peak):
        fig.add_vline(
            x=peak,
            line_dash="dash",
            line_color=PEAK_COLOR,
            annotation_text=f"M1 blunder peak (move {peak:.0f})",
            annotation_position="top left",
        )
    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Share of ALL eligible games in which the player has already reached time "
        "pressure by that move (a game that ended earlier still counts in the "
        "denominator). Evaluated only at the moves fixed in the spec, with 95% "
        "intervals. Exploratory."
    )


def main() -> None:
    st.title("M4 — Time management")

    st.markdown(
        """
        **Question:** how do I spend my time during the game, and when do I really enter time trouble?

        This page shows how long I spend on moves and how fast my clock runs down.
        """
    )

    games, moves = sidebar_filtered(*load_data())
    data = m4_clock_burn_map(moves, games).reset_index()

    perf_type_choice = st.selectbox(
        "Time control",
        data["perf_group"].unique(),
        format_func=lambda group: GROUP_LABELS.get(group, group),
        key="m4_time_control",
    )
    filtered = data[data["perf_group"] == perf_type_choice]

    tab1, tab2 = st.tabs(["Move duration", "Clock depletion"])

    with tab1:
        fig1 = px.line(
            filtered,
            x="move_number",
            y=["median_time_spent", "p90_time_spent"],
            markers=True,
            labels={
                "move_number": "Move number",
                "value": "Time spent (seconds)",
                "variable": "Metric",
            },
        )
        fig1.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
        st.plotly_chart(fig1, width="stretch")

    with tab2:
        fig2 = px.line(
            filtered,
            x="move_number",
            y="median_clock_after",
            markers=True,
            labels={
                "move_number": "Move number",
                "median_clock_after": "Remaining clock (seconds)",
            },
        )
        fig2.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "Median clock of the games still running at each move: late moves only "
            "contain long games, so this curve is not the time-pressure onset (see below)."
        )

    st.markdown("### Time pressure — me vs my opponents")
    threshold_pct = load_time_pressure_pct()
    control, cif, peaks = cached_m4_control(
        moves, games, selection_key(games), threshold_pct
    )
    peak = peaks.get(perf_type_choice, float("nan"))
    render_control(control, peak, perf_type_choice, threshold_pct)
    render_cif(cif, peak, perf_type_choice)

    st.markdown("### What this tells me")
    st.markdown(
        """
        - Spikes in move duration show where I slow down a lot, often in complex positions.
        - The control table answers "do I get into time trouble more than players at my
          level?" with the same rule applied to my opponents in the same games.
        - Comparing the onset with the M1 blunder peak separates **time trouble** from
          **position complexity**: if time pressure only starts well after the peak, the
          clock cannot be what causes the mid-game blunders.
        - Reaching time pressure less often is not proof of better clock management: a
          faster general pace, or opponents spending their time against my defence, would
          give the same result.
        """
    )


if __name__ == "__main__":
    main()
