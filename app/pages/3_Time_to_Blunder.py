from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from app.ui import load_ui
from src.metrics import m2_time_to_blunder


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


def main() -> None:
    st.title("M2 — Time to blunder")

    st.markdown(
        """
        **Question:** when I spend a very long time on a move, does it actually help me,
        or does it make a blunder more likely right afterwards?

        This page compares my blunder rate after a “long think” to my normal blunder rate.
        """
    )

    games, moves = load_data()
    stats_by_perf = m2_time_to_blunder(moves, games)

    perf_type_choice = st.selectbox("Perf type", list(stats_by_perf.keys()))
    stats = stats_by_perf[perf_type_choice]

    col1, col2, col3 = st.columns(3)
    col1.metric("Blunder risk multiplier", f"{stats['ratio']:.2f}×")
    col2.metric("Blunder rate after long think", f"{stats['p_given_long_think']:.1%}")
    col3.metric("Baseline blunder rate", f"{stats['baseline']:.1%}")

    st.markdown("### Sample size")
    st.info(
        f"I have {stats['n_long_think']} long-think moves in the data, "
        f"and {stats['n_blunders_after']} of them are followed by a blunder within the next 3 moves."
    )

    st.markdown("### How I interpret this")
    st.markdown(
        """
        - If the multiplier is clearly above 1×, it means long thinks often happen in positions
          where I’m already struggling.
        - It doesn’t mean “thinking more is bad”, but it shows that **when I get stuck**, I’m at
          higher risk of making a big mistake soon after.
        """
    )


if __name__ == "__main__":
    main()