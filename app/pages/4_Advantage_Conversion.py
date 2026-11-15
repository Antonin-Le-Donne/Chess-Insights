"""M3 — Advantage conversion (spec section 5, M3 + "M2/M3 — Opponent control
comparisons"), laid out with the dashboard principles (spec section 6):
the answer first, one main chart, details in expanders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.controls import GROUP_LABELS, control_reading, selection_key
from app.filters import format_time_control, game_urls, link_column, sidebar_filtered
from app.ui import load_ui
from src.metrics import m3_control, m3_greed_conversion

DRILL_DOWN_ROW_LIMIT = 200
RESULT_LABELS = {1.0: "Win", 0.5: "Draw", 0.0: "Loss"}

ME_COLOR = "#C7F36B"
OPPONENT_COLOR = "#8F9893"
GREEN = "#7BE0A3"
RED = "#FF6B6B"
GRAY = "#B7C2CC"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Inter", size=14),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    return games, moves


@st.cache_data(show_spinner="Comparing with my opponents (1,000 game resamples, about 15 s)…")
def cached_m3_control(_moves: pd.DataFrame, _games: pd.DataFrame, key: int) -> pd.DataFrame:
    # Underscored arguments are not hashed; `key` identifies the selection.
    return m3_control(_moves, _games)


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def render_answer(mine: pd.Series, theirs: pd.Series, group_label: str) -> None:
    """The answer in one sentence, with the spec's reading rule as a badge."""
    reading = control_reading(
        mine["difference_ci_lower"],
        mine["difference_ci_upper"],
        "I convert more often than my opponents",
        "My opponents convert more often than me",
    )
    color = GREEN if mine["difference_ci_lower"] > 0 else RED if mine["difference_ci_upper"] < 0 else GRAY
    st.markdown(
        f"<span style='color:{color};font-size:0.8rem;font-weight:600;letter-spacing:0.04em;"
        f"text-transform:uppercase'>● {reading}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"#### In {group_label.lower()}, once I reach a winning position (80%), I win "
        f"**{mine['conversion']:.0%}** of those games. My opponents, from the same point, "
        f"win **{theirs['conversion']:.0%}**: a difference of "
        f"**{100 * mine['conversion_difference_me_minus_opponent']:+.1f} points** "
        f"(95% interval {100 * mine['difference_ci_lower']:+.1f} to "
        f"{100 * mine['difference_ci_upper']:+.1f})."
    )


def render_comparison_chart(mine: pd.Series, theirs: pd.Series) -> None:
    """Main chart: my conversion vs my opponents', with 95% intervals."""
    rows = [("Me", mine, ME_COLOR), ("My opponents", theirs, OPPONENT_COLOR)]
    fig = go.Figure(
        go.Bar(
            x=[label for label, _, _ in rows],
            y=[row["conversion"] for _, row, _ in rows],
            marker_color=[color for _, _, color in rows],
            error_y=dict(
                type="data",
                array=[row["ci_upper"] - row["conversion"] for _, row, _ in rows],
                arrayminus=[row["conversion"] - row["ci_lower"] for _, row, _ in rows],
                color="#D2D6D0",
            ),
            text=[f"{row['conversion']:.1%}" for _, row, _ in rows],
            textposition="inside",
            customdata=[[int(row["n_games"])] for _, row, _ in rows],
            hovertemplate="%{x}: %{y:.1%} of %{customdata[0]:,} games<extra></extra>",
            width=0.5,
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(tickformat=".0%", range=[0, 1], title="Games won from 80%",
                   gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        f"Me: {int(mine['n_games']):,} games reaching 80% · my opponents: "
        f"{int(theirs['n_games']):,} games reaching 80% against me. "
        "Thin lines: 95% intervals. Note: their rate also depends on MY defence, which is "
        "strong (page Swindle Profile), so the gap mixes my conversion and my defence."
    )


def render_game_list(data: pd.DataFrame) -> None:
    """Games behind the number, linked at the move where I first reached 80%."""
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


def render_duration(data: pd.DataFrame) -> None:
    """How long it takes to finish, converted vs not."""
    plot = data.assign(
        outcome=data["did_convert"].map({True: "Converted", False: "Not converted"})
    )
    fig = px.histogram(
        plot,
        x="plies_to_finish",
        color="outcome",
        barmode="overlay",
        color_discrete_map={"Converted": ME_COLOR, "Not converted": RED},
        labels={"plies_to_finish": "Half-moves from 80% to the end of the game", "outcome": ""},
    )
    fig.update_layout(**PLOT_LAYOUT, height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    converted = data.loc[data["did_convert"], "plies_to_finish"].median()
    thrown = data.loc[~data["did_convert"], "plies_to_finish"].median()
    st.caption(
        f"Median: {converted:.0f} half-moves to finish a converted game, {thrown:.0f} for a "
        "game I did not convert."
    )


def main() -> None:
    st.title("M3 — Converting winning positions")
    st.markdown("**Question:** once I reach a clearly winning position, how often do I finish the job — and is that better or worse than players at my level?")

    games, moves = sidebar_filtered(*load_data())
    data = m3_greed_conversion(moves, games)
    if data.empty:
        st.info("No game in the selection reaches an 80% win probability.")
        return

    groups = [g for g in ("main", "bullet") if (data["perf_type"].eq("bullet") == (g == "bullet")).any()]
    group = st.radio(
        "Time control",
        groups,
        format_func=lambda g: GROUP_LABELS.get(g, g),
        horizontal=True,
        key="m3_group",
        label_visibility="collapsed",
    )
    group_data = data[data["perf_type"].eq("bullet") == (group == "bullet")]

    control = cached_m3_control(moves, games, selection_key(games))
    group_control = control[control["perf_group"].eq(group)]
    if group_control.empty:
        st.info("No game in this time control for the current selection.")
        return
    mine = group_control[group_control["side"].eq("me")].iloc[0]
    theirs = group_control[group_control["side"].eq("opponent")].iloc[0]

    render_answer(mine, theirs, GROUP_LABELS.get(group, group))
    render_comparison_chart(mine, theirs)

    with st.expander(f"The games behind the number ({len(group_data):,} games)"):
        render_game_list(group_data)

    with st.expander("How long it takes me to finish"):
        render_duration(group_data)

    with st.expander("How this is computed"):
        st.markdown(
            """
            - **Winning position:** the first moment my win probability (from the Stockfish
              evaluation) reaches 80%. **Converted** = I won the game; a draw counts as not
              converted.
            - **My opponents (control group):** the same rule from their side, in the same
              games: the first moment their win probability reaches 80% (mine falls to 20%).
            - **Intervals:** 1,000 resamples of whole games, 95%. The difference is computed on
              the same resampled games for both sides (paired bootstrap).
            - **Reading rule (fixed in the spec before computing):** an interval of the
              difference entirely above 0 means I convert more often; entirely below 0, my
              opponents do; containing 0, no detectable difference.
            - **Limit:** my rate depends on my conversion and on my opponents' defence; theirs
              on their conversion and on my defence. The gap therefore mixes both skills.
            """
        )


if __name__ == "__main__":
    main()
