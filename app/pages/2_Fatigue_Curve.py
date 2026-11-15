"""M1 — When I blunder (spec section 5, M1), laid out with the dashboard
principles (spec section 6): the answer first, one main chart, details in
expanders. The peak is the same data-derived peak as the M4 reading rule
(m4_reference_peak), so both pages always quote the same moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.controls import BADGE_GRAY, GROUP_LABELS, render_badge
from app.filters import (
    format_time_control,
    game_urls,
    link_column,
    selected_points,
    sidebar_filtered,
)
from app.ui import load_ui
from src.metrics import (
    M4_PEAK_MIN_MOVES_PER_BUCKET,
    M4_PEAK_MIN_PLIES,
    m1_fatigue_curve,
    m4_reference_peak,
)
from src.scope import m1_peak_by_move

DRILL_DOWN_ROW_LIMIT = 200
LINE_COLOR = "#C7F36B"
PEAK_COLOR = "#FF6B6B"

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


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def blunder_curve(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """My blunder rate per 5-move bucket on games of 40+ half-moves (pooled over
    game lengths), with my move count per bucket."""
    long_games = games[games["n_plies"] >= M4_PEAK_MIN_PLIES]
    long_moves = moves[moves["game_id"].isin(long_games["game_id"])]
    curve = m1_peak_by_move(long_moves, long_games).reset_index()

    mine = long_moves[long_moves["is_my_move"]]
    is_bullet = mine["game_id"].map(long_games.set_index("game_id")["perf_type"]).eq("bullet")
    counts = (
        mine.assign(
            perf_group=is_bullet.map({True: "bullet", False: "main"}),
            move_bucket=(mine["move_number"] // 5) * 5,
        )
        .groupby(["perf_group", "move_bucket"])
        .size()
        .rename("n_my_moves")
        .reset_index()
    )
    curve = curve.merge(counts, on=["perf_group", "move_bucket"], how="left")
    curve["reliable"] = curve["n_my_moves"].fillna(0).ge(M4_PEAK_MIN_MOVES_PER_BUCKET)
    return curve


def bucket_label(bucket: int) -> str:
    return f"{max(int(bucket), 1)}–{int(bucket) + 4}"


def render_answer(curve: pd.DataFrame, peak_end: float, group_label: str) -> None:
    render_badge("Not compared with my opponents yet", BADGE_GRAY)
    reliable = curve[curve["reliable"]].sort_values("move_bucket")
    if reliable.empty or pd.isna(peak_end):
        st.markdown("#### Not enough moves in this selection to locate a peak.")
        return
    peak = reliable[reliable["move_bucket"].eq(peak_end - 4)].iloc[0]
    start = reliable.iloc[0]
    last = reliable.iloc[-1]
    if last["move_bucket"] > peak["move_bucket"] and last["blunder_rate"] < peak["blunder_rate"]:
        conclusion = (
            "The rate falls after the peak, so it is not tiredness building up over the "
            "game: the middle of the game is where I blunder most."
        )
    else:
        conclusion = "The rate does not fall after the peak in this selection."
    st.markdown(
        f"#### In {group_label.lower()}, my blunders peak at moves "
        f"**{bucket_label(peak['move_bucket'])}**: **{peak['blunder_rate']:.1%}** of my moves "
        f"there, against **{start['blunder_rate']:.1%}** in moves {bucket_label(start['move_bucket'])} "
        f"and **{last['blunder_rate']:.1%}** by moves {bucket_label(last['move_bucket'])}. "
        + conclusion
    )


def render_main_chart(curve: pd.DataFrame, peak_end: float):
    reliable = curve[curve["reliable"]].sort_values("move_bucket")
    fig = go.Figure(
        go.Scatter(
            x=reliable["move_bucket"] + 2,
            y=reliable["blunder_rate"],
            mode="lines+markers",
            line=dict(color=LINE_COLOR, width=2),
            customdata=list(zip(
                reliable["move_bucket"],
                reliable["move_bucket"].map(bucket_label),
                reliable["n_my_moves"],
            )),
            hovertemplate=(
                "Moves %{customdata[1]}: %{y:.1%} "
                "(%{customdata[2]:,} of my moves)<extra></extra>"
            ),
        )
    )
    if pd.notna(peak_end):
        fig.add_vrect(
            x0=peak_end - 4, x1=peak_end + 1, fillcolor=PEAK_COLOR, opacity=0.15,
            line_width=0, annotation_text="peak", annotation_position="top left",
        )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Move"),
        yaxis=dict(title="My blunder rate", tickformat=".0%", rangemode="tozero",
                   gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False,
    )
    return st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
        on_select="rerun",
        selection_mode="points",
        key="m1_chart",
    )


def render_blunder_drill_down(
    games: pd.DataFrame,
    moves: pd.DataFrame,
    perf_group: str,
    move_bucket: int,
) -> None:
    """List my blunders in one 5-move bucket, with a link to each move."""
    game_info = games.set_index("game_id")
    my_blunders = moves[
        moves["is_my_move"]
        & moves["move_quality"].eq("blunder")
        & (moves["move_number"] // 5 * 5).eq(move_bucket)
    ].copy()
    is_bullet = my_blunders["game_id"].map(game_info["perf_type"]).eq("bullet")
    my_blunders = my_blunders[is_bullet if perf_group == "bullet" else ~is_bullet]

    st.markdown(f"#### My blunders in moves {bucket_label(move_bucket)}")
    if my_blunders.empty:
        st.info("No blunder of mine in this part of the game for the current selection.")
        return

    for column in ["date_utc", "my_color", "base_seconds", "increment_seconds"]:
        my_blunders[column] = my_blunders["game_id"].map(game_info[column])
    my_blunders = my_blunders.sort_values("date_utc", ascending=False)
    shown = my_blunders.head(DRILL_DOWN_ROW_LIMIT)
    st.dataframe(
        pd.DataFrame(
            {
                "Date": shown["date_utc"].dt.strftime("%Y-%m-%d"),
                "Time control": [
                    format_time_control(base, inc)
                    for base, inc in zip(shown["base_seconds"], shown["increment_seconds"])
                ],
                "Color": shown["my_color"].str.capitalize(),
                "Move": shown["move_number"],
                "Played": shown["san"],
                "Win% before": shown["win_pct"].round(0),
                "Win% after": (shown["win_pct"] + shown["win_pct_delta"]).round(0),
                "Game": game_urls(shown, ply_column="ply"),
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Game": link_column("Game (at the move)")},
    )
    st.caption(
        f"{len(my_blunders):,} blunders"
        + (f", the {DRILL_DOWN_ROW_LIMIT} most recent shown." if len(my_blunders) > DRILL_DOWN_ROW_LIMIT else ".")
        + " Games of any length. Lichess links open at the move; chess.com links open the game."
    )


def render_by_length(moves: pd.DataFrame, games: pd.DataFrame, group: str) -> None:
    data = m1_fatigue_curve(moves, games).reset_index()
    data = data[data["perf_group"] == group]
    fig = px.line(
        data,
        x="move_bucket",
        y="blunder_rate",
        color="duration_tercile",
        markers=True,
        labels={"move_bucket": "Move (5-move groups)", "blunder_rate": "My blunder rate",
                "duration_tercile": "Game length"},
    )
    fig.update_layout(**PLOT_LAYOUT, height=320, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Games split into thirds by total duration. All game lengths are included here, so "
        "the late part of the curve only contains games that lasted that long."
    )


def main() -> None:
    st.title("M1 — When I blunder")
    st.markdown("**Question:** at which point of the game do I blunder most — and is it tiredness?")

    games, moves = sidebar_filtered(*load_data())
    curve = blunder_curve(moves, games)
    peaks = m4_reference_peak(moves, games)

    groups = [g for g in ("main", "bullet") if curve["perf_group"].eq(g).any()]
    if not groups:
        st.info("No game of 40+ half-moves in the current selection.")
        return
    group = st.radio(
        "Time control",
        groups,
        format_func=lambda g: GROUP_LABELS.get(g, g),
        horizontal=True,
        key="m1_group",
        label_visibility="collapsed",
    )
    group_curve = curve[curve["perf_group"].eq(group)]
    peak_end = peaks.get(group, float("nan"))

    render_answer(group_curve, peak_end, GROUP_LABELS[group])
    event = render_main_chart(group_curve, peak_end)
    points = selected_points(event)
    if points and points[0].get("customdata"):
        render_blunder_drill_down(games, moves, group, int(points[0]["customdata"][0]))
    else:
        st.caption("Click a point to list my blunders in that part of the game, with links.")

    with st.expander("By game length (short / medium / long games)"):
        render_by_length(moves, games, group)
    with st.expander("How this is computed"):
        st.markdown(
            f"""
            - **Blunder:** a move that lowers my win probability by 20 points or more.
            - **Curve:** my blunder rate per group of 5 moves, on games of at least
              {M4_PEAK_MIN_PLIES} half-moves only, so that short games ending early (often
              because someone blundered) cannot shape the curve up to move 20.
            - **Peak:** the 5-move group with the highest rate among those with at least
              {M4_PEAK_MIN_MOVES_PER_BUCKET} of my moves (same rule as the time-trouble page).
            - **Why the curve falls late in the game:** part of it is real, but part is the
              measurement: near a won or lost position, win probability barely moves, so a
              20-point drop gets rarer.
            - **Open question:** the middlegame peak may come from complexity (more pieces,
              more options) rather than tiredness. Not tested yet, and not compared with my
              opponents yet.
            """
        )


if __name__ == "__main__":
    main()
