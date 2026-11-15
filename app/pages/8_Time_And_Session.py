"""M7 — Time of day and session position (spec section 5, M7), laid out
with the dashboard principles (spec section 6): the answer first, one main
chart area, details in expanders.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.controls import BADGE_GRAY, render_badge
from app.filters import render_filters_not_applied
from app.ui import load_ui
from src.metrics import elo_expected_score, m7_time_and_session

BAR_COLOR = "#7CC4FF"
REFERENCE_COLOR = "#F6C85F"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Inter", size=14),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet(ROOT / "data" / "processed" / "games.parquet")
    moves = pd.read_parquet(ROOT / "data" / "processed" / "moves.parquet")
    return games, moves


def load_m7_settings() -> tuple[str, float, int]:
    """Read M7's timezone and session settings from config.toml."""
    with (ROOT / "config.toml").open("rb") as file:
        settings = tomllib.load(file)["m7"]
    return (
        str(settings["timezone"]),
        float(settings["session_gap_minutes"]),
        int(settings["max_game_index"]),
    )


@st.cache_data(show_spinner="Rebuilding my playing sessions…")
def cached_profile(_moves: pd.DataFrame, _games: pd.DataFrame, settings: tuple) -> dict:
    timezone, gap_minutes, max_game_index = settings
    return m7_time_and_session(
        _moves, _games, timezone=timezone, gap_minutes=gap_minutes, max_game_index=max_game_index
    )


def overall_score_gap(games: pd.DataFrame) -> tuple[float | None, int]:
    """Game-weighted mean (score − Elo expected) on games with complete ratings."""
    columns = ["result_for_me", "my_rating", "opp_rating"]
    data = games[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return None, 0
    expected = elo_expected_score(data["my_rating"] - data["opp_rating"])
    return float((data["result_for_me"] - expected).mean()), len(data)


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def ordered(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows in reading order with a display label."""
    table = table.copy()
    if column == "local_hour":
        table["label"] = table[column].map(lambda h: f"{int(h):02d}h")
        return table.sort_values(column)
    if column == "game_index_bucket":
        table["label"] = table[column].astype(str)
        return table.assign(
            _order=pd.to_numeric(table["label"].str.rstrip("+"), errors="coerce")
        ).sort_values("_order").drop(columns="_order")
    order = ["first game", "after win", "after draw", "after loss"]
    table["label"] = table[column].astype(str)
    return table.set_index(column).reindex([o for o in order if o in set(table[column])]).reset_index()


def gap_range(table: pd.DataFrame) -> tuple[float, float]:
    return 100 * table["score_vs_expected"].min(), 100 * table["score_vs_expected"].max()


def render_answer(profile: dict, reference: float | None) -> None:
    render_badge("Measured against my rating's expectation, not against my opponents", BADGE_GRAY)
    by_hour, by_index = profile["by_hour"], profile["by_game_index"]
    after = profile["after_previous"].set_index("after_previous")["score_vs_expected"]
    hour_lo, hour_hi = gap_range(by_hour)
    index_lo, index_hi = gap_range(by_index)
    usual = f" around my usual **{100 * reference:+.1f}**" if reference is not None else ""
    sentence = (
        f"#### Against what my rating predicts, I score between **{hour_lo:+.1f}** and "
        f"**{hour_hi:+.1f} points** depending on the hour, and between **{index_lo:+.1f}** and "
        f"**{index_hi:+.1f}** depending on how many games I have played in a row{usual}."
    )
    if {"after win", "after loss"} <= set(after.index):
        sentence += (
            f" After a win I score **{100 * after['after win']:+.1f}**, after a loss "
            f"**{100 * after['after loss']:+.1f}**."
        )
    st.markdown(sentence)
    st.caption(
        f"Groups hold {int(by_hour['n_games'].min()):,}–{int(by_hour['n_games'].max()):,} games "
        "per hour: with a few hundred games, a group can move by several points by chance alone."
    )


def score_chart(table: pd.DataFrame, reference: float | None, x_title: str) -> None:
    fig = go.Figure(
        go.Bar(
            x=table["label"],
            y=100 * table["score_vs_expected"],
            marker_color=BAR_COLOR,
            customdata=table[["n_games", "score", "expected_score"]].to_numpy(),
            hovertemplate=(
                "%{x}: %{y:+.1f} pts (%{customdata[0]:,} games, scored "
                "%{customdata[1]:.1%} vs %{customdata[2]:.1%} expected)<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color="#8F9893", line_width=1)
    if reference is not None:
        fig.add_hline(
            y=100 * reference, line_dash="dash", line_color=REFERENCE_COLOR,
            annotation_text=f"my usual {100 * reference:+.1f}", annotation_position="top left",
        )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title=x_title, type="category"),
        yaxis=dict(title="Points above my rating's expectation",
                   gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def blunder_chart(table: pd.DataFrame, x_title: str) -> None:
    fig = go.Figure(
        go.Bar(
            x=table["label"],
            y=table["blunder_rate"],
            marker_color="#A992FF",
            customdata=table[["n_moves"]].to_numpy(),
            hovertemplate="%{x}: %{y:.1%} of %{customdata[0]:,} moves<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=x_title, type="category"),
        yaxis=dict(title="My blunder rate", tickformat=".1%", rangemode="tozero",
                   gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


VIEWS = [
    ("By hour of the day", "by_hour", "local_hour", "Local start hour"),
    ("By game in the session", "by_game_index", "game_index_bucket", "Game in the session"),
    ("By previous result", "after_previous", "after_previous", "Previous game"),
]


def main() -> None:
    st.title("M7 — Time of day, sessions and momentum")
    render_filters_not_applied(
        "sessions are rebuilt from the complete game history, and removing games "
        "would split sessions artificially."
    )
    st.markdown("**Question:** do the hour, a long session or my last result change how well I play?")

    settings = load_m7_settings()
    games, moves = load_data()
    profile = cached_profile(moves, games, settings)
    reference, reference_n = overall_score_gap(games)
    tables = {key: ordered(profile[key], column) for _, key, column, _ in VIEWS}

    render_answer(profile, reference)

    tabs = st.tabs([title for title, _, _, _ in VIEWS])
    for tab, (_, key, _, x_title) in zip(tabs, VIEWS):
        with tab:
            score_chart(tables[key], reference, x_title)

    with st.expander("Do I blunder more at some hours or deep into a session?"):
        tabs = st.tabs([title for title, _, _, _ in VIEWS])
        for tab, (_, key, _, x_title) in zip(tabs, VIEWS):
            with tab:
                blunder_chart(tables[key], x_title)

    with st.expander("The numbers"):
        for title, key, _, x_title in VIEWS:
            st.markdown(f"**{title}**")
            table = tables[key]
            st.dataframe(
                pd.DataFrame(
                    {
                        x_title: table["label"],
                        "Games": table["n_games"].astype(int),
                        "Score": table["score"].map("{:.1%}".format),
                        "Expected": table["expected_score"].map("{:.1%}".format),
                        "Gap (pts)": (100 * table["score_vs_expected"]).map("{:+.1f}".format),
                        "Blunder rate": table["blunder_rate"].map("{:.1%}".format),
                    }
                ),
                hide_index=True,
                width="stretch",
            )

    timezone, gap_minutes, max_game_index = settings
    usual = (
        f"{100 * reference:+.1f} points over {reference_n:,} games"
        if reference is not None
        else "unavailable (no game with complete ratings)"
    )
    with st.expander("How this is computed"):
        st.markdown(
            f"""
            - **Score vs expectation:** my result (win = 1, draw = ½) minus the score the Elo
              formula predicts from the rating gap in that game. It removes the effect of
              playing stronger or weaker opponents at certain hours.
            - **My usual level** (yellow line): {usual}. Ratings lag behind an improving
              player, so a group is unusual compared with that line, not with 0.
            - **Sessions:** a new session starts after a break of at least {gap_minutes:g}
              minutes ({timezone} time); game end times are estimated from the moves' clock
              times. Games {max_game_index}+ are grouped together.
            - **chess.com ratings are recorded after the game**, which biases these gaps by
              at most about 0.3 points.
            - **Observational:** a loss often follows a hard game, and a win raises my rating
              and therefore the opponents I meet. No confidence interval and no opponent
              comparison on this page.
            """
        )


if __name__ == "__main__":
    main()
