from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.ui import load_ui
from src.metrics import elo_expected_score, m7_time_and_session


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet"
    )
    moves = pd.read_parquet(
        ROOT / "data" / "processed" / "moves.parquet"
    )
    return games, moves


def load_m7_settings() -> tuple[str, float, int]:
    """Read M7's timezone and session settings from config.toml."""
    with (ROOT / "config.toml").open("rb") as file:
        config = tomllib.load(file)

    settings = config["m7"]
    return (
        str(settings["timezone"]),
        float(settings["session_gap_minutes"]),
        int(settings["max_game_index"]),
    )


def overall_score_gap(games: pd.DataFrame) -> tuple[float | None, int]:
    """Return the game-weighted mean score gap on games with complete ratings."""
    required_columns = ["result_for_me", "my_rating", "opp_rating"]
    score_data = games[required_columns].apply(pd.to_numeric, errors="coerce")
    score_data = score_data.dropna(subset=required_columns)
    if score_data.empty:
        return None, 0

    expected = elo_expected_score(
        score_data["my_rating"] - score_data["opp_rating"]
    )
    mean_gap = (score_data["result_for_me"] - expected).mean()
    return float(mean_gap), len(score_data)


def format_percent(value: float | int | None) -> str:
    """Format a proportion as a percentage, handling missing values."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def format_percentage_points(value: float | int | None) -> str:
    """Format a proportion difference as percentage points."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:+.1f} pp"


def format_hour(value: int | float | None) -> str:
    """Format a local hour as HH:00."""
    if value is None or pd.isna(value):
        return "Unknown"
    return f"{int(value):02d}:00"


def render_metric_section(
    title: str,
    description: str,
    table: pd.DataFrame,
    group_column: str,
    group_label: str,
    overall_reference: float | None,
    reference_n: int,
) -> None:
    """Render the score gap and blunder rate charts for one M7 table."""
    st.markdown(f"### {title}")
    if overall_reference is None:
        reference_description = (
            "The overall reference line is unavailable because no games have "
            "both a result and complete ratings."
        )
    else:
        reference_description = (
            f"Orange dashed line: overall average score gap "
            f"({format_percentage_points(overall_reference)}, n={reference_n:,}). "
            "Gray dashed line: parity with Elo expectation (0 pp)."
        )
    st.caption(f"{description} {reference_description}")

    if table.empty:
        st.info("No games are available for this breakdown.")
        return

    chart_data = table.copy()
    if group_column == "local_hour":
        chart_data["group_label"] = chart_data[group_column].map(format_hour)
        chart_data = chart_data.sort_values("local_hour", kind="stable")
    else:
        chart_data["group_label"] = chart_data[group_column].astype(str)
        if group_column == "game_index_bucket":
            # Sort numerically so the terminal bucket (e.g. "10+") follows 1-9.
            chart_data["_sort_order"] = pd.to_numeric(
                chart_data["group_label"].str.rstrip("+"),
                errors="raise",
            )
            chart_data = (
                chart_data.sort_values("_sort_order", kind="stable")
                .drop(columns="_sort_order")
            )

    # Explicit ordinal order prevents Vega-Lite from sorting string labels
    # lexicographically (which would put "10+" between "1" and "2").
    group_order = chart_data["group_label"].tolist()

    score_columns = [
        "group_label",
        "n_games",
        "score",
        "expected_score",
        "score_vs_expected",
    ]
    score_data = chart_data[score_columns].copy()

    blunder_data = chart_data[
        ["group_label", "n_games", "n_moves", "blunder_rate"]
    ].copy()

    score_layers = [
        {
            "mark": {
                "type": "rule",
                "color": "gray",
                "strokeDash": [4, 4],
            },
            "encoding": {
                "y": {"datum": 0, "type": "quantitative"},
            },
        },
        {
            "mark": {"type": "bar"},
            "encoding": {
                "x": {
                    "field": "group_label",
                    "type": "ordinal",
                    "title": group_label,
                    "sort": group_order,
                    "axis": {"labelAngle": -35},
                },
                "y": {
                    "field": "score_vs_expected",
                    "type": "quantitative",
                    "title": "Score − Elo expected",
                    "axis": {"format": "+.0%"},
                },
                "tooltip": [
                    {
                        "field": "group_label",
                        "type": "nominal",
                        "title": group_label,
                    },
                    {
                        "field": "n_games",
                        "type": "quantitative",
                        "title": "Games",
                        "format": ",d",
                    },
                    {
                        "field": "score",
                        "type": "quantitative",
                        "title": "Actual score",
                        "format": ".1%",
                    },
                    {
                        "field": "expected_score",
                        "type": "quantitative",
                        "title": "Elo expected",
                        "format": ".1%",
                    },
                    {
                        "field": "score_vs_expected",
                        "type": "quantitative",
                        "title": "Score gap",
                        "format": "+.1%",
                    },
                ],
            },
        },
    ]
    if overall_reference is not None:
        # Draw the overall average after the bars so the reference remains visible.
        score_layers.append(
            {
                "mark": {
                    "type": "rule",
                    "color": "#D97706",
                    "strokeDash": [6, 3],
                    "strokeWidth": 2,
                },
                "encoding": {
                    "y": {
                        "datum": overall_reference,
                        "type": "quantitative",
                    },
                },
            }
        )

    score_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": 280,
        "layer": score_layers,
    }

    blunder_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": 280,
        "mark": {"type": "bar"},
        "encoding": {
            "x": {
                "field": "group_label",
                "type": "ordinal",
                "title": group_label,
                "sort": group_order,
                "axis": {"labelAngle": -35},
            },
            "y": {
                "field": "blunder_rate",
                "type": "quantitative",
                "title": "My blunder rate",
                "axis": {"format": ".1%"},
            },
            "tooltip": [
                {
                    "field": "group_label",
                    "type": "nominal",
                    "title": group_label,
                },
                {
                    "field": "n_games",
                    "type": "quantitative",
                    "title": "Games in group",
                    "format": ",d",
                },
                {
                    "field": "n_moves",
                    "type": "quantitative",
                    "title": "My rated moves",
                    "format": ",d",
                },
                {
                    "field": "blunder_rate",
                    "type": "quantitative",
                    "title": "Blunder rate",
                    "format": ".2%",
                },
            ],
        },
    }

    score_col, blunder_col = st.columns(2)
    with score_col:
        st.markdown("**Score versus Elo expectation**")
        st.vega_lite_chart(
            score_data,
            score_spec,
            use_container_width=True,
        )
    with blunder_col:
        st.markdown("**Blunder rate**")
        if int(chart_data["n_moves"].sum()) == 0:
            st.info("No rated moves are available for this breakdown.")
        else:
            st.vega_lite_chart(
                blunder_data,
                blunder_spec,
                use_container_width=True,
            )

    display_data = chart_data[
        [
            "group_label",
            "n_games",
            "score",
            "expected_score",
            "score_vs_expected",
            "blunder_rate",
            "n_moves",
        ]
    ].copy()
    display_data["score"] = display_data["score"].map(format_percent)
    display_data["expected_score"] = display_data["expected_score"].map(
        format_percent
    )
    display_data["score_vs_expected"] = display_data[
        "score_vs_expected"
    ].map(format_percentage_points)
    display_data["blunder_rate"] = display_data["blunder_rate"].map(
        format_percent
    )
    display_data = display_data.rename(
        columns={
            "group_label": group_label,
            "n_games": "Games",
            "score": "Actual score",
            "expected_score": "Elo expected",
            "score_vs_expected": "Score gap",
            "blunder_rate": "Blunder rate",
            "n_moves": "My rated moves",
        }
    )
    st.dataframe(display_data, hide_index=True, use_container_width=True)


st.set_page_config(
    page_title="Chess Insights",
    layout="wide",
)

load_ui()


def main() -> None:
    st.title("M7 — Time of day and session position")

    st.markdown(
        """
        **Question:** do I play below my usual level at certain hours, deep
        into a session, or immediately after a loss (the "tilt queue")?

        The main signal is **actual score minus Elo-expected score**. The Elo
        expectation controls for the rating gap within each game, so a stronger
        opponent pool at a particular hour is less likely to look like fatigue.
        """
    )

    try:
        timezone, gap_minutes, max_game_index = load_m7_settings()
        games, moves = load_data()
        profile = m7_time_and_session(
            moves,
            games,
            timezone=timezone,
            gap_minutes=gap_minutes,
            max_game_index=max_game_index,
        )
        reference_gap, reference_n = overall_score_gap(games)
    except (KeyError, OSError, ValueError, TypeError) as exc:
        st.error(f"Could not load or calculate M7: {exc}")
        return

    st.caption(
        f"Local timezone: {timezone}. A new session starts after a gap of "
        f"at least {gap_minutes:g} minutes from the estimated end of the "
        f"previous game. Game positions are grouped into {max_game_index}+ "
        "for the tail. Game end times are estimates based on summed move time."
    )

    render_metric_section(
        title="By local start hour",
        description=(
            "Each bar shows how far the score is above or below Elo expectation "
            "for games started in that local hour. Hover over a bar to see the "
            "number of games and both score values."
        ),
        table=profile["by_hour"],
        group_column="local_hour",
        group_label="Local start hour",
        overall_reference=reference_gap,
        reference_n=reference_n,
    )

    render_metric_section(
        title="By game index within a session",
        description=(
            "Game 1 is the first game after a qualifying break. Later positions "
            "can reflect fatigue, but they can also reflect selection effects "
            "such as choosing to keep playing after a close game."
        ),
        table=profile["by_game_index"],
        group_column="game_index_bucket",
        group_label="Game in session",
        overall_reference=reference_gap,
        reference_n=reference_n,
    )

    render_metric_section(
        title="By the previous game's result",
        description=(
            "Compares the first game of a session with games played after a win, "
            "draw, or loss in that same session."
        ),
        table=profile["after_previous"],
        group_column="after_previous",
        group_label="Previous result",
        overall_reference=reference_gap,
        reference_n=reference_n,
    )

    st.markdown("### How I interpret this")
    st.markdown(
        """
        - **Compare with the overall average, not just zero.** The orange
          dashed line is my game-weighted average score gap across all games
          with a result and both ratings. If I usually score about +2 pp above
          Elo expectation, a bucket near +2 pp is typical for me; a bucket below
          that line is weaker than my baseline even if it remains above zero.
          The gray dashed line marks exact Elo expectation (0 pp).
        - **Read the score gap, not the raw score.** A negative value means I
          scored below the Elo expectation for the ratings in those games; a
          positive value means I scored above it. A value of −5 pp means five
          percentage points below expectation.
        - **Check the sample sizes.** Late-night hours and high session indexes
          may contain few games. Small groups can move sharply by chance; hover
          over each bar or use the table to check `n_games` and `n_moves`.
        - **The blunder rate has a different denominator.** It includes my moves
          with a known `move_quality`, not every move or every game. Compare it
          alongside the score gap rather than treating either statistic alone
          as a complete explanation.
        - **This is observational, not a causal test.** Playing after a loss can
          also mean that the previous game was difficult or emotionally costly.
          Time of day, opponent strength beyond the rating gap, opening choice,
          and time control may still contribute to differences.
        - **Sessions are inferred.** Only start times are stored, so the end of
          each game is estimated by summing `time_spent` across its moves. The
          configured gap determines where sessions split; unusual estimates or
          overlapping games can affect the grouping.
        """
    )


if __name__ == "__main__":
    main()
