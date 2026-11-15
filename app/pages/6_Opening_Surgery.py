"""M5 — Openings (spec section 5, M5), laid out with the dashboard
principles (spec section 6): the answer first, one main chart, details in
expanders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.controls import BADGE_GRAY, render_badge
from app.filters import (
    format_time_control,
    game_urls,
    link_column,
    opening_family_labels,
    selected_points,
    sidebar_filtered,
)
from app.ui import load_ui
from src.metrics import m5_opening_surgery

MIN_GAMES = 100  # same display rule as the Overview card
DRILL_DOWN_ROW_LIMIT = 200
RESULT_LABELS = {1.0: "Win", 0.5: "Draw", 0.0: "Loss"}
GOOD = "#7BE0A3"
BAD = "#FF6B6B"
FEW = "#8F9893"

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


def label(row: pd.Series) -> str:
    return f"{row['opening_family']} · {row['my_color'].capitalize()}"


def render_answer(data: pd.DataFrame) -> None:
    render_badge("Not compared with my opponents yet", BADGE_GRAY)
    enough = data[data["n_games"] >= MIN_GAMES]
    if enough.empty:
        st.markdown(f"#### No opening has {MIN_GAMES}+ games in this selection.")
        return
    worst = enough.loc[enough["score_pct"].idxmin()]
    best = enough.loc[enough["score_pct"].idxmax()]
    st.markdown(
        f"#### My weakest opening is the **{worst['opening_family']} as "
        f"{worst['my_color'].capitalize()}**: **{worst['score_pct']:.0%}** score over "
        f"{int(worst['n_games']):,} games. My best: the **{best['opening_family']} as "
        f"{best['my_color'].capitalize()}**, **{best['score_pct']:.0%}** over "
        f"{int(best['n_games']):,} games."
    )


def render_main_chart(data: pd.DataFrame):
    """Score per opening family and color, gap to 50%, sorted."""
    plot = data.assign(label=data.apply(label, axis=1)).sort_values("score_pct")
    colors = [
        FEW if n < MIN_GAMES else (GOOD if score >= 0.5 else BAD)
        for n, score in zip(plot["n_games"], plot["score_pct"])
    ]
    fig = go.Figure(
        go.Bar(
            y=plot["label"],
            x=plot["score_pct"] - 0.5,
            base=0.5,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.0%} ({int(n):,})" for s, n in zip(plot["score_pct"], plot["n_games"])],
            textposition="outside",
            customdata=plot[["opening_family", "my_color"]].to_numpy(),
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.add_vline(x=0.5, line_color="#8F9893", line_width=1)
    fig.update_layout(
        **PLOT_LAYOUT,
        height=max(320, 34 * len(plot) + 60),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="My score (win = 1, draw = ½)", tickformat=".0%",
                   gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    event = st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
        on_select="rerun",
        selection_mode="points",
        key="m5_chart",
    )
    st.caption(
        f"Bars start at 50%: green above, red below. Gray = fewer than {MIN_GAMES} games, "
        "too few to conclude (an opening result on fewer than 50 games already reversed "
        "once in this project). In brackets: number of games. Click a bar to list its games."
    )
    return event


def render_opening_games(games: pd.DataFrame, family: str, color: str) -> None:
    """Games behind one bar: same family, same color, same n_plies >= 20 rule as M5."""
    rows = games[
        (games["n_plies"] >= 20)
        & opening_family_labels(games["eco"]).eq(family)
        & games["my_color"].eq(color)
    ].sort_values("date_utc", ascending=False)

    st.markdown(f"#### {family} as {color.capitalize()} — {len(rows):,} games")
    if rows.empty:
        st.info("No game for this bar in the current selection.")
        return
    shown = rows.head(DRILL_DOWN_ROW_LIMIT)
    st.dataframe(
        pd.DataFrame(
            {
                "Date": shown["date_utc"].dt.strftime("%Y-%m-%d"),
                "Time control": [
                    format_time_control(base, inc)
                    for base, inc in zip(shown["base_seconds"], shown["increment_seconds"])
                ],
                "ECO": shown["eco"],
                "Opening (Lichess only)": shown["opening_name"],
                "Result": shown["result_for_me"].map(RESULT_LABELS),
                "Game": game_urls(shown),
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Game": link_column()},
    )
    if len(rows) > DRILL_DOWN_ROW_LIMIT:
        st.caption(f"The {DRILL_DOWN_ROW_LIMIT} most recent games shown.")


def render_out_of_opening(data: pd.DataFrame) -> None:
    table = data.assign(label=data.apply(label, axis=1)).sort_values("mean_my_win_pct_ply20")
    st.dataframe(
        pd.DataFrame(
            {
                "Opening": table["label"],
                "Games": table["n_games"].astype(int),
                "My win% after 10 moves": table["mean_my_win_pct_ply20"].map("{:.0f}%".format),
                "My blunder rate in the first 10 moves": table["blunder_rate_first20"].map("{:.1%}".format),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Tells whether an opening goes wrong straight out of the opening (low win% after "
        "10 moves: a theory gap) or later (fine after 10 moves, poor final score: a plan gap)."
    )


def main() -> None:
    st.title("M5 — Openings")
    st.markdown("**Question:** which openings work for me, and which ones cost me points?")

    games, moves = sidebar_filtered(*load_data())
    data = m5_opening_surgery(moves, games)
    if data.empty:
        st.info("No game of a tracked opening family in the current selection.")
        return

    render_answer(data)
    event = render_main_chart(data)
    points = selected_points(event)
    if points and points[0].get("customdata"):
        family, color = points[0]["customdata"][:2]
        render_opening_games(games, family, color)

    with st.expander("Do I come out of the opening worse?"):
        render_out_of_opening(data)
    with st.expander("How this is computed"):
        st.markdown(
            """
            - **Opening family:** from the ECO code, mapped to six families I play or face
              (Sicilian, French, Dutch, Italian, Catalan, Queen's Gambit). Other openings are
              not shown here.
            - **Score:** win = 1, draw = ½, loss = 0, over games of at least 20 half-moves,
              all time controls together (a deliberate choice, see the tracker), split by the
              color I played.
            - **Raw score, not adjusted for opponent strength:** stronger or weaker opponents
              in an opening shift it (the opponent-strength check is in the README's
              detailed findings, finding 5).
            - **Not compared with my opponents yet**, and no confidence interval: read
              differences of a few points as noise below a few hundred games.
            """
        )


if __name__ == "__main__":
    main()
