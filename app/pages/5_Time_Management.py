from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import load_ui
from src.metrics import m4_clock_burn_map


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


def main() -> None:
    st.title("M4 — Time management")

    st.markdown(
        """
        **Question:** how do I spend my time during the game, and when do I really enter time trouble?

        This page shows how long I spend on moves and how fast my clock runs down.
        """
    )

    games, moves = load_data()
    data = m4_clock_burn_map(moves, games).reset_index()

    perf_type_choice = st.selectbox(
        "Time control",
        data["perf_group"].unique(),
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
        st.plotly_chart(fig1, use_container_width=True)

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
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### What this tells me")
    st.markdown(
        """
        - Spikes in move duration show where I slow down a lot, often in complex positions.
        - A steep drop in remaining time shows when I enter real time pressure.
        - Comparing this with my blunder patterns helps me see whether I lose games more from
          **time trouble** or from **position complexity**.
        """
    )


if __name__ == "__main__":
    main()
