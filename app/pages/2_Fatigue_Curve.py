from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.filters import (
    format_time_control,
    game_urls,
    link_column,
    selected_points,
    sidebar_filtered,
)
from app.ui import load_ui
from src.metrics import m1_fatigue_curve

DRILL_DOWN_ROW_LIMIT = 200


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    return games, moves


PLOTLY_LAYOUT = go.Layout(
    font=dict(family="Inter", size=14),
    hovermode="x unified",
    template="plotly_white",
    margin=dict(l=40, r=40, t=60, b=40),
)

st.set_page_config(
    page_title="Chess Insights",
    layout="wide",
)

load_ui()


def render_blunder_drill_down(
    games: pd.DataFrame,
    moves: pd.DataFrame,
    perf_group: str,
    move_bucket: int,
) -> None:
    """List my blunders in one 5-move bucket, with a link to each move."""
    game_info = games.set_index("game_id")
    my_blunders = moves[
        moves["is_my_move"]
        & moves["move_quality"].eq("blunder")
        & (moves["move_number"] // 5 * 5).eq(move_bucket)
    ].copy()
    is_bullet = my_blunders["game_id"].map(game_info["perf_type"]).eq("bullet")
    my_blunders = my_blunders[is_bullet if perf_group == "bullet" else ~is_bullet]

    st.markdown(
        f"#### My blunders in moves {move_bucket}–{move_bucket + 4} "
        f"({perf_group}, all game lengths)"
    )
    if my_blunders.empty:
        st.info("No blunder of mine in this bucket for the current selection.")
        return

    for column in ["date_utc", "my_color", "base_seconds", "increment_seconds"]:
        my_blunders[column] = my_blunders["game_id"].map(game_info[column])
    my_blunders = my_blunders.sort_values("date_utc", ascending=False)

    shown = my_blunders.head(DRILL_DOWN_ROW_LIMIT)
    table = pd.DataFrame(
        {
            "Date": shown["date_utc"].dt.strftime("%Y-%m-%d"),
            "Time control": [
                format_time_control(base, inc)
                for base, inc in zip(shown["base_seconds"], shown["increment_seconds"])
            ],
            "Color": shown["my_color"].str.capitalize(),
            "Move": shown["move_number"],
            "Played": shown["san"],
            "Win% before": shown["win_pct"].round(0),
            "Win% after": (shown["win_pct"] + shown["win_pct_delta"]).round(0),
            "Game": game_urls(shown, ply_column="ply"),
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Game": link_column("Game (at the move)")},
    )
    st.caption(
        f"{len(my_blunders):,} blunders in this bucket"
        + (f", the {DRILL_DOWN_ROW_LIMIT} most recent shown." if len(my_blunders) > DRILL_DOWN_ROW_LIMIT else ".")
        + " Lichess links open at the move; chess.com links open the game."
    )


def main() -> None:
    st.title("M1 — Fatigue curve")

    st.markdown(
        """
        **Question:** do I blunder more and more as the game goes on, or is there a specific
        phase where my mistakes spike?

        This page shows how my blunder rate changes over the game, split by game length and time control.
        """
    )

    games, moves = sidebar_filtered(*load_data())
    data = m1_fatigue_curve(moves, games).reset_index()

    perf_type_choice = st.selectbox(
        "Time control",
        data["perf_group"].unique(),
        key="m1_time_control",
    )
    filtered = data[data["perf_group"] == perf_type_choice]

    fig = px.line(
        filtered,
        x="move_bucket",
        y="blunder_rate",
        color="duration_tercile",
        markers=True,
        labels={
            "move_bucket": "Move number (grouped)",
            "blunder_rate": "Blunder rate",
            "duration_tercile": "Game length group",
        },
    )
    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
    event = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="m1_chart",
    )

    points = selected_points(event)
    if points:
        render_blunder_drill_down(games, moves, perf_type_choice, int(points[0]["x"]))
    else:
        st.caption("Click a point to list my blunders in that part of the game, with links.")

    st.markdown("### How I read this")
    st.markdown(
        """
        - Each line is a group of games by length (short / medium / long).
        - The y-axis shows how often I blunder in each part of the game.
        - A clear peak in the middle of the game suggests that complexity there hurts me more than pure fatigue.
        """
    )

    st.markdown("### What this tells me")
    st.markdown(
        """
        In my current data, my blunder rate peaks around the middlegame and then drops in the endgame.
        That means I should focus my training on **middlegame decision-making**, not just “staying fresh”
        for long games.
        """
    )


if __name__ == "__main__":
    main()
