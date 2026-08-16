from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import load_ui
from src.metrics import m5_opening_surgery


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
    st.title("M5 — Opening surgery")

    st.markdown(
        """
        **Question:** which opening families are actually good for me, and which ones are real leaks?

        This page scores opening families by my results, split by color.
        """
    )

    games, moves = load_data()
    data = m5_opening_surgery(moves, games)

    fig = px.bar(
        data,
        x="opening_family",
        y="score_pct",
        color="my_color",
        barmode="group",
        hover_data=["n_games"],
        labels={
            "opening_family": "Opening family",
            "score_pct": "Score percentage",
            "my_color": "Played as",
            "n_games": "Games",
        },
    )
    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What this tells me")
    st.markdown(
        """
        - High scores show openings where I consistently do well.
        - Low scores with enough games point to openings I should probably fix or replace.
        - Very small sample sizes are just hints, not solid conclusions.
        """
    )

    st.markdown("### Raw opening stats")
    st.dataframe(data, use_container_width=True)


if __name__ == "__main__":
    main()
