from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.ui import load_ui
from src.metrics import m6_swindle_profile


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet"
    )
    moves = pd.read_parquet(
        ROOT / "data" / "processed" / "moves.parquet"
    )
    return games, moves


def load_thresholds() -> tuple[int, int, int, int]:
    """Read all M6 thresholds from config.toml."""
    with (ROOT / "config.toml").open("rb") as file:
        config = tomllib.load(file)

    thresholds = config["thresholds"]
    return (
        int(thresholds["lost_pct"]),
        int(thresholds["min_plies"]),
        int(thresholds["cp_error_threshold"]),
        int(thresholds["swing_to_pct"]),
    )


def format_percent(value: float | int | None) -> str:
    """Format a proportion as a percentage, handling missing values."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def format_gap(value: float | int | None) -> str | None:
    """Format a score difference as percentage points."""
    if value is None or pd.isna(value):
        return None
    return f"{float(value) * 100:+.1f} pp vs expected"


st.set_page_config(
    page_title="Chess Insights",
    layout="wide",
)

load_ui()


def main() -> None:
    st.title("M6 — Swindle profile")

    st.markdown(
        """
        **Question:** when I reach a lost position, how often do I still save
        the game, and how often does my opponent offer a realistic chance to
        recover?

        This page identifies games where my win probability remains low for
        several consecutive plies. It compares my actual score with the
        expected-score proxy at the start of that stretch, then measures how
        often an opponent move substantially improves my winning chances.
        """
    )

    (
        lost_pct,
        min_plies,
        cp_error_threshold,
        swing_to_pct,
    ) = load_thresholds()

    st.caption(
        f"A lost stretch requires the selected side's win probability to stay "
        f"at or below {lost_pct}% for {min_plies} consecutive plies. "
        f"A large CP error is at least {cp_error_threshold} cp. A rescue "
        f"opportunity requires my win probability to rise from at most "
        f"{lost_pct}% to at least {swing_to_pct}% after an opponent move."
    )

    games, moves = load_data()

    profile = m6_swindle_profile(
        moves,
        games,
        lost_pct=lost_pct,
        min_plies=min_plies,
        cp_error_threshold=cp_error_threshold,
        swing_to_pct=swing_to_pct,
    )

    perf_group_label = st.selectbox(
        "Time control",
        ["Blitz / rapid (main)", "Bullet"],
        key="m6_time_control",
    )
    perf_group_choice = (
        "bullet" if perf_group_label == "Bullet" else "main"
    )

    # Results for games where I reached a qualifying lost stretch.
    st.markdown("### Results after a lost stretch")

    results = profile["results"]
    selected_results = results.loc[
        results["perf_group"] == perf_group_choice
    ]

    if selected_results.empty or int(selected_results.iloc[0]["n"]) == 0:
        st.info(
            "No games reached a qualifying lost stretch for this time control."
        )
    else:
        result = selected_results.iloc[0]
        n_games = int(result["n"])
        wins = int(result["wins"])
        draws = int(result["draws"])
        losses = int(result["losses"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lost games", f"{n_games:,}")
        col2.metric("Wins", f"{wins:,}", format_percent(wins / n_games))
        col3.metric("Draws", f"{draws:,}", format_percent(draws / n_games))
        col4.metric("Losses", f"{losses:,}", format_percent(losses / n_games))

    # Control group: apply the same lost-stretch rule to both sides.
    st.markdown("### Control group — score vs expected")

    selected_control = profile["control"].loc[
        profile["control"]["perf_group"] == perf_group_choice
    ]

    if selected_results.empty:
        my_n = 0
        my_score = None
        my_expected = None
        my_gap = None
    else:
        my_stats = selected_results.iloc[0]
        my_n = int(my_stats["n"])
        my_score = my_stats["score_pct"]
        my_expected = my_stats["expected_score"]
        my_gap = my_stats["score_vs_expected"]

    if selected_control.empty:
        opponent_n = 0
        opponent_score = None
        opponent_expected = None
        opponent_gap = None
    else:
        opponent_stats = selected_control.iloc[0]
        opponent_n = int(opponent_stats["n"])
        opponent_score = opponent_stats["actual_score"]
        opponent_expected = opponent_stats["expected_score"]
        opponent_gap = opponent_stats["score_vs_expected"]

    col1, col2 = st.columns(2)

    col1.metric(
        f"My score vs expected (n={my_n:,})",
        format_percent(my_score),
        delta=format_gap(my_gap),
        help=(
            f"Expected score: {format_percent(my_expected)}. "
            "Expected score is the average win probability at the start of "
            "my first qualifying lost stretch."
        ),
    )

    col2.metric(
        f"Opponent score vs expected (n={opponent_n:,})",
        format_percent(opponent_score),
        delta=format_gap(opponent_gap),
        help=(
            f"Expected score: {format_percent(opponent_expected)}. "
            "The same lost-stretch rule is applied from the opponent's POV; "
            "their result is calculated as 1 - result_for_me."
        ),
    )

    st.caption(
        "If both score gaps are similar, that is evidence that the difference "
        "comes from the win-percentage formula's calibration, not necessarily "
        "from my swindling skill. The formula is calibrated on Lichess players "
        "overall, not specifically on my rating."
    )

    # Move-level errors are measured in centipawns rather than move_quality.
    st.markdown("### Opponent CP errors by position state")

    opponent_moves = profile["opponent_moves"]
    selected_moves = opponent_moves.loc[
        opponent_moves["perf_group"] == perf_group_choice
    ]
    lost_moves = selected_moves.loc[selected_moves["state"] == "lost"]
    winning_moves = selected_moves.loc[selected_moves["state"] == "winning"]

    if lost_moves.empty:
        lost_n = 0
        lost_rate = None
        lost_errors = 0
    else:
        lost_stats = lost_moves.iloc[0]
        lost_n = int(lost_stats["n"])
        lost_rate = lost_stats["cp_error_rate"]
        lost_errors = int(lost_stats["n_cp_errors"])

    if winning_moves.empty:
        winning_n = 0
        winning_rate = None
        winning_errors = 0
    else:
        winning_stats = winning_moves.iloc[0]
        winning_n = int(winning_stats["n"])
        winning_rate = winning_stats["cp_error_rate"]
        winning_errors = int(winning_stats["n_cp_errors"])

    col1, col2 = st.columns(2)
    col1.metric(
        f"Opponent error rate — I am lost (n={lost_n:,})",
        format_percent(lost_rate),
        help=(
            f"{lost_errors:,} opponent moves with a CP error out of "
            f"{lost_n:,} moves with valid evaluations."
        ),
    )
    col2.metric(
        f"Opponent error rate — I am winning (n={winning_n:,})",
        format_percent(winning_rate),
        help=(
            f"{winning_errors:,} opponent moves with a CP error out of "
            f"{winning_n:,} moves with valid evaluations."
        ),
    )

    st.caption(
        "Only moves with valid before/after evaluations enter these rates. "
        "CP loss can be inflated in already-decided positions: for example, "
        "an evaluation improving from -900 to -650 cp is a 250 cp loss for "
        "the mover even though the game may remain lost. The difference in "
        "these rates can therefore partly reflect the measurement itself."
    )

    # Game-level opportunity and save rates.
    st.markdown("### Rescue opportunities and save rate")

    opportunities = profile["opportunities"]
    selected_opportunities = opportunities.loc[
        opportunities["perf_group"] == perf_group_choice
    ].copy()

    if (
        selected_opportunities.empty
        or int(selected_opportunities["n_lost_games"].max()) == 0
    ):
        st.info("No qualifying lost games are available for this time control.")
    else:
        chance_rows = selected_opportunities.loc[
            selected_opportunities["opponent_gave_chance"]
        ]
        no_chance_rows = selected_opportunities.loc[
            ~selected_opportunities["opponent_gave_chance"]
        ]
        chance_row = chance_rows.iloc[0]
        no_chance_row = no_chance_rows.iloc[0]
        total_lost_games = int(chance_row["n_lost_games"])

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Opponent offered a chance",
            format_percent(chance_row["share_of_lost_games"]),
            help=(
                f"{int(chance_row['n_games']):,} of "
                f"{total_lost_games:,} qualifying lost games."
            ),
        )
        col2.metric(
            "Saved when a chance was offered",
            format_percent(chance_row["saved_rate"]),
            help=(
                f"{int(chance_row['n_saved']):,} wins or draws out of "
                f"{int(chance_row['n_games']):,} games with a chance."
            ),
        )
        col3.metric(
            "Saved without an identified chance",
            format_percent(no_chance_row["saved_rate"]),
            help=(
                f"{int(no_chance_row['n_saved']):,} wins or draws out of "
                f"{int(no_chance_row['n_games']):,} games without a qualifying swing."
            ),
        )

        opportunity_table = selected_opportunities[
            [
                "opponent_gave_chance",
                "n_games",
                "share_of_lost_games",
                "n_saved",
                "saved_rate",
                "score_pct",
            ]
        ].copy()
        opportunity_table["opponent_gave_chance"] = (
            opportunity_table["opponent_gave_chance"]
            .map({True: "Yes", False: "No"})
        )
        opportunity_table["share_of_lost_games"] = (
            opportunity_table["share_of_lost_games"].map(format_percent)
        )
        opportunity_table["saved_rate"] = (
            opportunity_table["saved_rate"].map(format_percent)
        )
        opportunity_table["score_pct"] = (
            opportunity_table["score_pct"].map(format_percent)
        )
        opportunity_table = opportunity_table.rename(
            columns={
                "opponent_gave_chance": "Chance offered",
                "n_games": "Lost games",
                "share_of_lost_games": "Share of lost games",
                "n_saved": "Games saved (win or draw)",
                "saved_rate": "Save rate",
                "score_pct": "Score",
            }
        )
        st.dataframe(
            opportunity_table,
            hide_index=True,
            use_container_width=True,
        )

    # Termination breakdown only includes qualifying lost games that ended
    # in a save (win or draw), split by chance offered yes/no.
    st.markdown("### How saved games ended")

    saves_by_termination = profile["saves_by_termination"]
    selected_terminations = saves_by_termination.loc[
        saves_by_termination["perf_group"] == perf_group_choice
    ].copy()

    if selected_terminations.empty:
        st.info("No saved games are available for this time control.")
    else:
        selected_terminations["Chance offered"] = (
            selected_terminations["opponent_gave_chance"]
            .map({True: "Yes", False: "No"})
        )
        display_categories = {
            "time": "Time",
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
        selected_terminations["Termination category"] = (
            selected_terminations["termination_category"].map(display_categories)
        )
        selected_terminations["Share of saved games"] = (
            selected_terminations["share_of_saved_games"].map(format_percent)
        )
        selected_terminations = selected_terminations.rename(
            columns={"n_saved": "Saved games"}
        )
        selected_terminations = selected_terminations.sort_values(
            ["Chance offered", "Saved games"],
            ascending=[True, False],
        )
        st.dataframe(
            selected_terminations[
                [
                    "Chance offered",
                    "Termination category",
                    "Saved games",
                    "Share of saved games",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Lichess 'Normal' combines checkmates, resignations and draws without "
        "telling us which occurred, so it is kept as its own category instead "
        "of guessing. Shares are calculated among saved games within each "
        "chance-offered group."
    )

    st.markdown("### How I interpret this")
    st.markdown(
        f"""
        - **A lost stretch identifies a game, not a single mistake.** My win
          probability must remain at or below {lost_pct}% for at least
          {min_plies} consecutive plies.

        - **The control group is essential.** The same rule is applied to games
          where my opponent was lost. If both sides exceed their expected score
          by a similar amount, the difference may come from the win-probability
          formula's calibration rather than from swindling skill.

        - **A rescue opportunity is a meaningful swing.** It is counted when an
          opponent move after the lost stretch raises my win probability from
          at most {lost_pct}% to at least {swing_to_pct}%. The threshold is fixed
          in advance rather than tuned to fit the results.

        - **The save rate measures conversion of an opportunity.** The
          opportunity rate shows how often opponents offer a substantial
          swing; the save rate shows how often I turn those games into a win
          or a draw. The no-opportunity group captures saves without that
          particular kind of swing.

        - **CP errors are secondary evidence.** A large centipawn loss can
          happen in an already-decided position without meaningfully changing
          the outcome. A CP error is not automatically a real rescue chance.

        - **Termination categories distinguish different kinds of saves.**
          A game saved on time is different from one saved through a board
          reversal. Lichess 'Normal' is ambiguous and is not interpreted more
          specifically than the source data allows.

        The expected score uses a win-probability formula calibrated on Lichess
        players overall, not specifically on my rating. These results are
        observational and do not prove that an opponent error alone caused a save.
        """
    )


if __name__ == "__main__":
    main()
