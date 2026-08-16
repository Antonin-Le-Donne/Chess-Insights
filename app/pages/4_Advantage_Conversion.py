from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import load_ui
from src.metrics import m3_greed_conversion


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
    st.title("M3 — Advantage conversion")

    st.markdown(
        """
        **Question:** when I reach a clearly winning position, how often do I actually finish the job?

        This page looks at games where my win chance goes above a high threshold and checks
        how many of those games I still manage to win.
        """
    )

    games, moves = load_data()
    data = m3_greed_conversion(moves, games)

    col1, col2 = st.columns(2)
    col1.metric("Conversion success rate", f"{data['did_convert'].mean():.1%}")
    col2.metric("Median moves to finish", int(data["plies_to_finish"].median()))

    st.markdown("### How long it takes me to finish")
    fig = px.histogram(
        data,
        x="plies_to_finish",
        color="did_convert",
        barmode="overlay",
        labels={
            "plies_to_finish": "Moves from first big advantage to game end",
            "did_convert": "Game converted to a win",
        },
    )
    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What this tells me")
    st.markdown(
        """
        - The conversion rate shows how reliable I am at closing out winning games.
        - If thrown games tend to last longer, it suggests I hesitate or miss chances instead of
          simplifying and finishing cleanly.
        - This is a direct measure of my **technical conversion**, separate from openings or early blunders.
        """
    )


if __name__ == "__main__":
    main()
