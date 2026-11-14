from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from app.controls import GROUP_LABELS, control_reading, format_interval, selection_key
from app.filters import sidebar_filtered
from app.ui import load_ui
from src.metrics import m2_control, m2_time_to_blunder


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    return games, moves


@st.cache_data(show_spinner="Bootstrapping 1,000 game resamples (about 15 s)…")
def cached_m2_control(
    _moves: pd.DataFrame,
    _games: pd.DataFrame,
    key: int,
) -> pd.DataFrame:
    # Underscored arguments are not hashed; `key` identifies the selection.
    return m2_control(_moves, _games)


def render_control(moves: pd.DataFrame, games: pd.DataFrame) -> None:
    """My ratio vs my opponents' ratio, same rule, with game-level 95% CIs."""
    st.markdown("### Control group — the same measure on my opponents")
    control = cached_m2_control(moves, games, selection_key(games))
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
                "My ratio [95% CI]": format_interval(
                    mine["ratio"], mine["ratio_ci_lower"], mine["ratio_ci_upper"], "{:.2f}×"
                ),
                "My long thinks": int(mine["n_long_think"]),
                "Opponents' ratio [95% CI]": format_interval(
                    theirs["ratio"], theirs["ratio_ci_lower"], theirs["ratio_ci_upper"], "{:.2f}×"
                ),
                "Opponents' long thinks": int(theirs["n_long_think"]),
                "Difference me − opponents [95% CI]": format_interval(
                    mine["ratio_difference_me_minus_opponent"],
                    mine["difference_ci_lower"],
                    mine["difference_ci_upper"],
                    "{:+.2f}",
                ),
                "Reading": control_reading(
                    mine["difference_ci_lower"],
                    mine["difference_ci_upper"],
                    "Stronger in my moves (personal)",
                    "Stronger in my opponents' moves",
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "A long think is measured against each player's OWN median time in the game "
        "(spec, Option B), so a faster player is not compared with a slower "
        "opponent's pace — the same definition as the numbers at the top of the page. "
        "Intervals: 1,000 game resamples; the difference is a "
        "paired bootstrap (both sides on the same resampled games). An interval "
        "containing 0 shows no detectable difference — it does not prove equality."
    )


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

    games, moves = sidebar_filtered(*load_data())
    stats_by_perf = m2_time_to_blunder(moves, games)

    # A time-control group emptied by the sidebar filters has no long think.
    available = [
        perf for perf, stats in stats_by_perf.items() if stats["n_long_think"] > 0
    ]
    if not available:
        st.info("No long-think move in the current selection.")
        return

    perf_type_choice = st.selectbox("Perf type", available)
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

    render_control(moves, games)

    st.markdown("### How I interpret this")
    st.markdown(
        """
        - If the multiplier is clearly above 1×, it means long thinks often happen in positions
          where I’m already struggling.
        - It doesn’t mean “thinking more is bad”, but it shows that **when I get stuck**, I’m at
          higher risk of making a big mistake soon after.
        - The control group decides whether this is **mine** or simply what difficult positions
          do to everyone at my level: if my opponents show the same multiplier, it is not a
          personal signature.
        """
    )


if __name__ == "__main__":
    main()