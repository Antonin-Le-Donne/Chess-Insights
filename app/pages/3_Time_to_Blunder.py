"""M2 — Time to blunder (spec section 5, M2 + "M2/M3 — Opponent control
comparisons", Option B), laid out with the dashboard principles (spec
section 6): the answer first, one main chart, details in expanders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.controls import (
    GROUP_LABELS,
    control_reading,
    render_badge,
    selection_key,
    verdict_color,
)
from app.filters import sidebar_filtered
from app.ui import load_ui
from src.metrics import m2_control

NORMAL_COLOR = "#7CC4FF"
LONG_THINK_COLOR = "#FF6B6B"

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
def cached_m2_control(_moves: pd.DataFrame, _games: pd.DataFrame, key: int) -> pd.DataFrame:
    # Underscored arguments are not hashed; `key` identifies the selection.
    return m2_control(_moves, _games)


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def render_answer(mine: pd.Series, theirs: pd.Series, group_label: str) -> None:
    reading = control_reading(
        mine["difference_ci_lower"],
        mine["difference_ci_upper"],
        "Stronger in my moves than in my opponents'",
        "Weaker in my moves than in my opponents'",
    )
    if reading.startswith("No detectable"):
        reading = "Shared at my level"
    render_badge(
        reading,
        verdict_color(mine["difference_ci_lower"], mine["difference_ci_upper"], good_when_negative=True),
    )
    shared = mine["difference_ci_lower"] <= 0 <= mine["difference_ci_upper"]
    difference = (
        f"({mine['ratio_difference_me_minus_opponent']:+.2f}, 95% interval "
        f"{mine['difference_ci_lower']:+.2f} to {mine['difference_ci_upper']:+.2f})"
    )
    conclusion = (
        f"My opponents show the same pattern (**{theirs['ratio']:.1f}×**): the difference "
        f"{difference} is not detectable. Long thinks mark hard positions for everyone at my "
        "level; they are not a habit of mine."
        if shared
        else f"My opponents: **{theirs['ratio']:.1f}×**. The difference {difference} is "
        "detectable: this pattern is "
        + ("stronger" if mine["difference_ci_lower"] > 0 else "weaker")
        + " in my games than in my opponents'."
    )
    st.markdown(
        f"#### In {group_label.lower()}, after a very long think I blunder within my next 3 moves "
        f"**{mine['p_given_long_think']:.0%}** of the time, against **{mine['baseline']:.1%}** "
        f"normally: **{mine['ratio']:.1f}× more often**. {conclusion}"
    )


def render_main_chart(mine: pd.Series, theirs: pd.Series) -> None:
    """Blunder rate normally vs after a long think, for me and my opponents."""
    sides = ["Me", "My opponents"]
    fig = go.Figure(
        [
            go.Bar(
                name="Normally",
                x=sides,
                y=[mine["baseline"], theirs["baseline"]],
                marker_color=NORMAL_COLOR,
                text=[f"{mine['baseline']:.1%}", f"{theirs['baseline']:.1%}"],
                textposition="outside",
            ),
            go.Bar(
                name="Within 3 moves after a long think",
                x=sides,
                y=[mine["p_given_long_think"], theirs["p_given_long_think"]],
                marker_color=LONG_THINK_COLOR,
                text=[f"{mine['p_given_long_think']:.1%}", f"{theirs['p_given_long_think']:.1%}"],
                textposition="outside",
            ),
        ]
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        height=330,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(tickformat=".0%", title="Blunder rate", gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        f"Multiplier: me {mine['ratio']:.2f}× (95% interval {mine['ratio_ci_lower']:.2f}–"
        f"{mine['ratio_ci_upper']:.2f}), my opponents {theirs['ratio']:.2f}× "
        f"({theirs['ratio_ci_lower']:.2f}–{theirs['ratio_ci_upper']:.2f})."
    )


def main() -> None:
    st.title("M2 — Long thinks and blunders")
    st.markdown("**Question:** when I think for a very long time, do I blunder right after — and is that special to me?")

    games, moves = sidebar_filtered(*load_data())
    control = cached_m2_control(moves, games, selection_key(games))
    groups = [
        g for g in ("main", "bullet")
        if not control[control["perf_group"].eq(g) & control["side"].eq("me")].empty
        and control.loc[control["perf_group"].eq(g) & control["side"].eq("me"), "n_long_think"].iloc[0] > 0
    ]
    if not groups:
        st.info("No long-think move in the current selection.")
        return

    group = st.radio(
        "Time control",
        groups,
        format_func=lambda g: GROUP_LABELS.get(g, g),
        horizontal=True,
        key="m2_group",
        label_visibility="collapsed",
    )
    rows = control[control["perf_group"].eq(group)]
    mine = rows[rows["side"].eq("me")].iloc[0]
    theirs = rows[rows["side"].eq("opponent")].iloc[0]

    render_answer(mine, theirs, GROUP_LABELS[group])
    render_main_chart(mine, theirs)

    with st.expander("Sample sizes"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Side": label,
                        "Moves": f"{int(row['n_moves']):,}",
                        "Blunders": f"{int(row['n_blunders']):,}",
                        "Long thinks": f"{int(row['n_long_think']):,}",
                        "Followed by a blunder within 3 moves": f"{int(row['n_blunders_after']):,}",
                    }
                    for label, row in (("Me", mine), ("My opponents", theirs))
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("How this is computed"):
        st.markdown(
            """
            - **Long think:** a move that takes more than 8 times the player's *own* median
              time in that game (spec, Option B), so a fast player is not judged against a
              slower opponent's pace.
            - **Blunder after a long think:** a blunder by the same player within their next
              3 moves. **Normally** = that player's overall blunder rate.
            - **Multiplier** = blunder rate after a long think ÷ normal blunder rate.
            - **My opponents (control group):** the same rule on their moves, in the same games.
            - **Intervals:** 1,000 resamples of whole games, 95%; the difference is computed on
              the same resampled games for both sides (paired bootstrap).
            - **Reading rule (fixed in the spec before computing):** an interval of the
              difference that contains 0 means no detectable difference — compatible with a
              pattern shared at my level, not proof that it is identical.
            - **Not causal:** a long think often happens *because* the position is hard, which
              is also when blunders happen. Thinking more is not shown to be bad.
            """
        )


if __name__ == "__main__":
    main()
