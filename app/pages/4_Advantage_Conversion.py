from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.controls import GROUP_LABELS, control_reading, format_interval, selection_key
from app.filters import format_time_control, game_urls, link_column, sidebar_filtered
from app.ui import load_ui
from src.metrics import m3_control, m3_greed_conversion

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


@st.cache_data(show_spinner="Bootstrapping 1,000 game resamples (about 15 s)…")
def cached_m3_control(_moves: pd.DataFrame, _games: pd.DataFrame, key: int) -> pd.DataFrame:
    # Underscored arguments are not hashed; `key` identifies the selection.
    return m3_control(_moves, _games)


def render_control(moves: pd.DataFrame, games: pd.DataFrame) -> None:
    """My conversion vs my opponents' (mirror rule), with game-level 95% CIs."""
    st.markdown("### Control group — my opponents from the same threshold")
    control = cached_m3_control(moves, games, selection_key(games))
    if control.empty:
        st.info("No game in the current selection.")
        return

    rows = []
    for group, group_rows in control.groupby("perf_group", sort=False):
        mine = group_rows.loc[group_rows["side"] == "me"].iloc[0]
        theirs = group_rows.loc[group_rows["side"] == "opponent"].iloc[0]
        rows.append(
            {
                "Time control": GROUP_LABELS.get(group, group),
                "My conversion [95% CI]": format_interval(
                    mine["conversion"], mine["ci_lower"], mine["ci_upper"], "{:.1%}"
                ),
                "My games at 80%": int(mine["n_games"]),
                "Opponents' conversion [95% CI]": format_interval(
                    theirs["conversion"], theirs["ci_lower"], theirs["ci_upper"], "{:.1%}"
                ),
                "Their games at 80%": int(theirs["n_games"]),
                "Difference me − opponents [95% CI]": format_interval(
                    mine["conversion_difference_me_minus_opponent"] * 100,
                    mine["difference_ci_lower"] * 100,
                    mine["difference_ci_upper"] * 100,
                    "{:+.1f} pp",
                ),
                "Reading": control_reading(
                    mine["difference_ci_lower"],
                    mine["difference_ci_upper"],
                    "I convert more often",
                    "My opponents convert more often",
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Opponents' winning point: the first position where their win probability "
        "reaches 80% (mine falls to 20%); a draw is a non-conversion for both sides. "
        "Limit: my rate depends on my conversion and on their defence, theirs on their "
        "conversion and on MY defence — my swindles (M6) lower their rate, so the "
        "difference mixes both skills. Intervals: 1,000 game resamples, paired for the "
        "difference."
    )


def render_game_list(data: pd.DataFrame) -> None:
    """Games behind the chart, linked at the move where I first reached 80%."""
    st.markdown("### The games behind the chart")
    choice = st.radio(
        "Show",
        ["Not converted (draws and losses)", "Converted"],
        horizontal=True,
        key="m3_game_list",
    )
    converted = choice == "Converted"
    rows = data[data["did_convert"] == converted].sort_values("date_utc", ascending=False)
    if rows.empty:
        st.info("No game in this group for the current selection.")
        return

    # winning_point_ply's win% is the eval of the position BEFORE that ply,
    # i.e. after ply - 1 half-moves: link there so the board shows it.
    shown = rows.head(DRILL_DOWN_ROW_LIMIT).assign(
        position_ply=lambda df: df["winning_point_ply"] - 1
    )
    table = pd.DataFrame(
        {
            "Date": shown["date_utc"].dt.strftime("%Y-%m-%d"),
            "Time control": [
                format_time_control(base, inc)
                for base, inc in zip(shown["base_seconds"], shown["increment_seconds"])
            ],
            "Color": shown["my_color"].str.capitalize(),
            "Reached 80% after move": (shown["position_ply"] + 1) // 2,
            "Lowest win% afterwards": shown["max_drawdown"].round(0),
            "Plies to finish": shown["plies_to_finish"],
            "Result": shown["result_for_me"].map(RESULT_LABELS),
            "Game": game_urls(shown, ply_column="position_ply"),
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Game": link_column("Game (at the 80% point)")},
    )
    st.caption(
        f"{len(rows):,} games"
        + (f", the {DRILL_DOWN_ROW_LIMIT} most recent shown." if len(rows) > DRILL_DOWN_ROW_LIMIT else ".")
        + " Lichess links open at the move where I first reached 80%; chess.com links open the game."
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

    games, moves = sidebar_filtered(*load_data())
    data = m3_greed_conversion(moves, games)
    if data.empty:
        st.info("No game in the selection reaches a 80% win probability.")
        return

    if games["perf_type"].eq("bullet").any() and games["perf_type"].ne("bullet").any():
        st.caption(
            "This selection mixes bullet with blitz/rapid. The README quotes them "
            "separately (72.6% blitz/rapid, 67.3% bullet): filter the time control "
            "in the sidebar to reproduce those numbers."
        )

    col1, col2 = st.columns(2)
    col1.metric("Conversion success rate", f"{data['did_convert'].mean():.1%}")
    col2.metric("Median moves to finish", int(data["plies_to_finish"].median()))

    render_control(moves, games)

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
    st.plotly_chart(fig, width="stretch")

    render_game_list(data)

    st.markdown("### What this tells me")
    st.markdown(
        """
        - The conversion rate shows how reliable I am at closing out winning games.
        - If thrown games tend to last longer, it suggests I hesitate or miss chances instead of
          simplifying and finishing cleanly.
        - On its own, the rate is not a measure of technical conversion: it also depends on how
          well my opponents defend. The control group above compares it with my opponents
          from the same threshold, and that difference still mixes my conversion with my defence.
        """
    )


if __name__ == "__main__":
    main()
