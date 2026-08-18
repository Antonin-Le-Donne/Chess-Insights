from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import load_ui
from src.metrics import m1_fatigue_curve


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

def main() -> None:
    st.title("M1 — Fatigue curve")

    st.markdown(
        """
        **Question:** do I blunder more and more as the game goes on, or is there a specific
        phase where my mistakes spike?

        This page shows how my blunder rate changes over the game, split by game length and time control.
        """
    )

    games, moves = load_data()
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
    st.plotly_chart(fig, use_container_width=True)

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
