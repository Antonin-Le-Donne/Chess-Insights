"""M6 — Swindle profile (spec section 5, M6), laid out with the dashboard
principles (spec section 6): the answer first, one main chart, details in
expanders.
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

from app.controls import BADGE_BLUE, GROUP_LABELS, render_badge
from app.filters import sidebar_filtered
from app.ui import load_ui
from src.metrics import m6_swindle_profile

ME_COLOR = "#C7F36B"
OPPONENT_COLOR = "#8F9893"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Inter", size=14),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

TERMINATION_LABELS = {
    "time": "On time",
    "checkmate": "Checkmate",
    "resignation": "Resignation",
    "stalemate": "Stalemate",
    "repetition": "Repetition",
    "insufficient_material": "Insufficient material",
    "agreement": "Agreement",
    "abandoned": "Abandoned",
    "normal_lichess": "Normal (Lichess; reason hidden)",
    "other": "Other / unrecognized",
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet(ROOT / "data" / "processed" / "games.parquet")
    moves = pd.read_parquet(ROOT / "data" / "processed" / "moves.parquet")
    return games, moves


def load_thresholds() -> tuple[int, int, int, int]:
    """Read all M6 thresholds from config.toml."""
    with (ROOT / "config.toml").open("rb") as file:
        thresholds = tomllib.load(file)["thresholds"]
    return (
        int(thresholds["lost_pct"]),
        int(thresholds["min_plies"]),
        int(thresholds["cp_error_threshold"]),
        int(thresholds["swing_to_pct"]),
    )


def pct(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.1%}"


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def share_saved_on_time(terminations: pd.DataFrame) -> float:
    """Share of all my saved games (win or draw) that ended on the clock."""
    total = terminations["n_saved"].sum()
    on_time = terminations.loc[terminations["termination_category"].eq("time"), "n_saved"].sum()
    return on_time / total if total else float("nan")


def render_answer(mine: pd.Series, theirs: pd.Series | None, on_time: float, group_label: str) -> None:
    render_badge("Compared with my opponents · no interval yet", BADGE_BLUE)
    sentence = (
        f"#### In {group_label.lower()}, from a lost position ({int(mine['n']):,} games), I score "
        f"**{mine['score_pct']:.0%}** where the engine expects **{mine['expected_score']:.0%}**: "
        f"**{100 * mine['score_vs_expected']:+.0f} points**."
    )
    if theirs is not None:
        sentence += (
            f" My opponents, in their own lost positions, beat the same expectation by only "
            f"**{100 * theirs['score_vs_expected']:+.0f} points**."
        )
    if pd.notna(on_time):
        sentence += f" About **{on_time:.0%}** of my saves end on the clock."
    st.markdown(sentence)


def render_main_chart(mine: pd.Series, theirs: pd.Series | None) -> None:
    """Main chart: score above the engine's expectation, me vs my opponents."""
    rows = [("Me", mine, ME_COLOR)]
    if theirs is not None:
        rows.append(("My opponents", theirs, OPPONENT_COLOR))
    fig = go.Figure(
        go.Bar(
            x=[label for label, _, _ in rows],
            y=[100 * row["score_vs_expected"] for _, row, _ in rows],
            marker_color=[color for _, _, color in rows],
            text=[f"{100 * row['score_vs_expected']:+.1f} pts" for _, row, _ in rows],
            textposition="outside",
            customdata=[
                [pct(row["score_pct"] if "score_pct" in row else row["actual_score"]),
                 pct(row["expected_score"]), int(row["n"])]
                for _, row, _ in rows
            ],
            hovertemplate=(
                "%{x}: scored %{customdata[0]} vs %{customdata[1]} expected "
                "(%{customdata[2]:,} lost games)<extra></extra>"
            ),
            width=0.5,
        )
    )
    fig.add_hline(y=0, line_color="#8F9893", line_width=1)
    fig.update_layout(
        **PLOT_LAYOUT,
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Points scored above the expectation",
                   gridcolor="rgba(255,255,255,0.07)", rangemode="tozero"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Expectation = the average engine win probability at the moment the game became "
        "lost. The formula is calibrated on Lichess players overall, so part of any gap can "
        "come from calibration — which is why the same rule applied to my opponents is the "
        "reference, not 0."
    )


def render_chances(opportunities: pd.DataFrame, terminations: pd.DataFrame) -> None:
    if opportunities.empty or int(opportunities["n_lost_games"].max()) == 0:
        st.info("No qualifying lost games for this time control.")
        return
    chance = opportunities.loc[opportunities["opponent_gave_chance"]].iloc[0]
    no_chance = opportunities.loc[~opportunities["opponent_gave_chance"]].iloc[0]
    no_chance_on_time = share_saved_on_time(
        terminations.loc[~terminations["opponent_gave_chance"]]
    )
    st.markdown(
        f"- My opponent gave me a real chance in **{pct(chance['share_of_lost_games'])}** of my "
        f"lost games ({int(chance['n_games']):,} of {int(chance['n_lost_games']):,}).\n"
        f"- With that chance, I saved the game (win or draw) **{pct(chance['saved_rate'])}** of "
        "the time.\n"
        f"- Without one, I still saved **{pct(no_chance['saved_rate'])}** "
        f"({int(no_chance['n_saved']):,} games), {pct(no_chance_on_time)} of them on the clock."
    )


def render_terminations(terminations: pd.DataFrame) -> None:
    if terminations.empty:
        st.info("No saved games for this time control.")
        return
    table = terminations.assign(
        **{
            "Chance offered": terminations["opponent_gave_chance"].map({True: "Yes", False: "No"}),
            "How it ended": terminations["termination_category"].map(TERMINATION_LABELS),
            "Share of saved games": terminations["share_of_saved_games"].map(pct),
        }
    ).rename(columns={"n_saved": "Saved games"})
    st.dataframe(
        table.sort_values(["Chance offered", "Saved games"], ascending=[True, False])[
            ["Chance offered", "How it ended", "Saved games", "Share of saved games"]
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Shares are within each chance-offered group. Lichess 'Normal' hides whether the "
        "game ended by mate, resignation or draw, so it is kept as its own category."
    )


def render_opponent_errors(opponent_moves: pd.DataFrame, cp_error_threshold: int) -> None:
    rates = opponent_moves.set_index("state")
    lines = []
    for state, label in (("lost", "when I am lost"), ("winning", "when I am winning")):
        if state in rates.index:
            row = rates.loc[state]
            lines.append(
                f"- Opponent moves losing {cp_error_threshold}+ centipawns {label}: "
                f"**{pct(row['cp_error_rate'])}** ({int(row['n']):,} moves)"
            )
    st.markdown("\n".join(lines))
    st.caption(
        "Secondary evidence: in an already-decided position, a large centipawn loss can "
        "change nothing (e.g. −900 to −650), so part of the gap is the measurement itself."
    )


def main() -> None:
    st.title("M6 — Saving lost positions")
    st.markdown("**Question:** when I am lost, do I save more games than players at my level would — and how?")

    lost_pct, min_plies, cp_error_threshold, swing_to_pct = load_thresholds()
    games, moves = sidebar_filtered(*load_data())
    profile = m6_swindle_profile(
        moves,
        games,
        lost_pct=lost_pct,
        min_plies=min_plies,
        cp_error_threshold=cp_error_threshold,
        swing_to_pct=swing_to_pct,
    )

    group = st.radio(
        "Time control",
        ["main", "bullet"],
        format_func=lambda g: GROUP_LABELS.get(g, g),
        horizontal=True,
        key="m6_group",
        label_visibility="collapsed",
    )

    def for_group(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[frame["perf_group"].eq(group)]

    results = for_group(profile["results"])
    if results.empty or int(results.iloc[0]["n"]) == 0:
        st.info("No game reached a lost position for this time control in the current selection.")
        return
    mine = results.iloc[0]
    control = for_group(profile["control"])
    theirs = control.iloc[0] if not control.empty and int(control.iloc[0]["n"]) > 0 else None
    terminations = for_group(profile["saves_by_termination"])

    render_answer(mine, theirs, share_saved_on_time(terminations), GROUP_LABELS[group])
    render_main_chart(mine, theirs)

    with st.expander("Did my opponents give me chances?"):
        render_chances(for_group(profile["opportunities"]), terminations)
    with st.expander("How my saved games ended"):
        render_terminations(terminations)
    with st.expander("Do my opponents make more mistakes when I am lost?"):
        render_opponent_errors(for_group(profile["opponent_moves"]), cp_error_threshold)
    with st.expander("How this is computed"):
        st.markdown(
            f"""
            - **Lost position:** my win probability stays at or below {lost_pct}% for
              {min_plies} consecutive half-moves. It identifies a game, not one mistake.
            - **Score:** win = 1, draw = ½, loss = 0, over those games ({int(mine['wins'])} wins,
              {int(mine['draws'])} draws, {int(mine['losses'])} losses here).
            - **Expectation:** the average engine win probability when the lost stretch begins.
            - **My opponents (control group):** the same rule applied to the games where
              *they* were lost. Their gap is the reference: the formula is calibrated on
              Lichess players overall, so both sides may beat it a little.
            - **A real chance:** an opponent move that lifts my win probability from
              {lost_pct}% or less to {swing_to_pct}% or more (threshold fixed in advance).
            - **No confidence interval yet** on this comparison, unlike conversion (M3) and
              time pressure (M4): the gap is large compared with its rough standard error
              (about 1 point), but it has not been through the bootstrap protocol.
            - Observational: this does not prove that a given save was caused by my play.
            """
        )


if __name__ == "__main__":
    main()
