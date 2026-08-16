from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.metrics import m3_greed_conversion


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    return games, moves


def kpi_card(title: str, value, subtitle: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <h3>{title}</h3>
            <p style="font-size: 28px; font-weight: 600; color: #FFFFFF;">{value}</p>
            <p style="color: #DADADA;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    st.set_page_config(
        page_title="Chess Insights",
        layout="wide",
    )

    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown("<style>" + css_path.read_text() + "</style>", unsafe_allow_html=True)

    games, moves = load_data()

    st.title("Chess Insights — Dashboard")
    st.caption("My personal chess performance analytics — v1")

    st.markdown(
        """
        <div class="header-banner">
            <h2 style="margin: 0;">My pipeline is up and running</h2>
            <p style="margin: 0;">Fetch → Parse → Evaluate → Features → Metrics → Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This dashboard helps me understand **my own mistakes**.  
        Each page focuses on one question: fatigue, time usage, converting winning positions, and openings.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        kpi_card("Total games parsed", len(games), "Games fetched and processed")

    with col2:
        eval_coverage = 1 - moves["eval_cp"].isna().mean()
        kpi_card("Engine eval coverage", f"{eval_coverage:.1%}", "Moves with Stockfish evaluation")

    with col3:
        m3 = m3_greed_conversion(moves, games)
        kpi_card("Median moves to finish", int(m3["plies_to_finish"].median()), "From first big advantage to game end")

    st.markdown(
        """
        <br>
        <div class="block">
            <h3>How to use this dashboard</h3>
            <p>- Use the pages in the sidebar to explore each metric.</p>
            <p>- Every page explains what it measures and how to read the charts.</p>
            <p>- Findings, Decisions, and Debug Log summarize what I’ve learned.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()