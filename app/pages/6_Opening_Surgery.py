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
    opening_family_labels,
    selected_points,
    sidebar_filtered,
)
from app.ui import load_ui
from src.metrics import m5_opening_surgery

DRILL_DOWN_ROW_LIMIT = 200
RESULT_LABELS = {1.0: "Win", 0.5: "Draw", 0.0: "Loss"}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    return games, moves

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


def render_opening_games(games: pd.DataFrame, family: str, color: str) -> None:
    """Games behind one bar: same family, same color, same n_plies >= 20 rule as M5."""
    rows = games[
        (games["n_plies"] >= 20)
        & opening_family_labels(games["eco"]).eq(family)
        & games["my_color"].eq(color)
    ].sort_values("date_utc", ascending=False)

    st.markdown(f"#### {family} as {color} — {len(rows):,} games")
    if rows.empty:
        st.info("No game for this bar in the current selection.")
        return

    shown = rows.head(DRILL_DOWN_ROW_LIMIT)
    table = pd.DataFrame(
        {
            "Date": shown["date_utc"].dt.strftime("%Y-%m-%d"),
            "Time control": [
                format_time_control(base, inc)
                for base, inc in zip(shown["base_seconds"], shown["increment_seconds"])
            ],
            "ECO": shown["eco"],
            "Opening (Lichess only)": shown["opening_name"],
            "Result": shown["result_for_me"].map(RESULT_LABELS),
            "Game": game_urls(shown),
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Game": link_column()},
    )
    if len(rows) > DRILL_DOWN_ROW_LIMIT:
        st.caption(f"The {DRILL_DOWN_ROW_LIMIT} most recent games shown.")


def main() -> None:
    st.title("M5 — Opening surgery")

    st.markdown(
        """
        **Question:** which opening families are actually good for me, and which ones are real leaks?

        This page scores opening families by my results, split by color.
        """
    )

    games, moves = sidebar_filtered(*load_data())
    data = m5_opening_surgery(moves, games)
    if data.empty:
        st.info("No game of a tracked opening family in the current selection.")
        return

    fig = px.bar(
        data,
        x="opening_family",
        y="score_pct",
        color="my_color",
        barmode="group",
        hover_data=["n_games"],
        custom_data=["opening_family", "my_color"],
        labels={
            "opening_family": "Opening family",
            "score_pct": "Score percentage",
            "my_color": "Played as",
            "n_games": "Games",
        },
    )
    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
    event = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="m5_chart",
    )

    points = selected_points(event)
    if points and points[0].get("customdata"):
        family, color = points[0]["customdata"][:2]
        render_opening_games(games, family, color)
    else:
        st.caption("Click a bar to list its games, with links.")

    st.markdown("### What this tells me")
    st.markdown(
        """
        - High scores show openings where I consistently do well.
        - Low scores with enough games point to openings I should probably fix or replace.
        - Very small sample sizes are just hints, not solid conclusions.
        """
    )

    st.markdown("### Raw opening stats")
    st.dataframe(data, width="stretch")


if __name__ == "__main__":
    main()
