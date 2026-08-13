"""Core metrics (spec section 5) -- the calculations that ARE the project.

Each function: pure, takes DataFrames (+ config where needed), returns a
DataFrame/dict. One dashboard view per metric. Only operates on MY moves
(is_my_move == True) unless the metric's own definition says otherwise.

M1-M5 for now (Milestone 3). M6/M7 come later (Milestone 4).
"""

from __future__ import annotations

import pandas as pd


def m1_fatigue_curve(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Mean win_pct_delta + blunder rate of MY moves, bucketed by move_number
    (bins of 5), split by perf_type. Plus an overlay split by game duration
    tercile (spec section 5, M1)."""

    my_moves = moves[moves["is_my_move"]]
    move_bucket = (my_moves["move_number"] // 5 ) * 5

    game_duration = moves.groupby("game_id")["time_spent"].sum()
    tercile_per_game = pd.qcut(game_duration, q=3, labels=["short", "medium", "long"])
    duration_tercile = my_moves["game_id"].map(tercile_per_game)

    grouped = my_moves.groupby(["perf_type", move_bucket, duration_tercile]).agg(
            mean_win_pct_delta=("win_pct_delta", "mean"),
            blunder_rate=("move_quality", lambda s: (s == "blunder").mean()),
            )

    return grouped


def m2_time_to_blunder(moves: pd.DataFrame) -> dict:
    """P(blunder within my next 3 moves | long_think on current move) vs
    baseline P(blunder), as a ratio WITH counts (spec section 5, M2).
    """

    my_moves = moves[moves["is_my_move"]].sort_values(["game_id", "ply"])
    is_blunder = my_moves["move_quality"] == "blunder"

    next1 = is_blunder.groupby(my_moves["game_id"]).shift(-1)
    next2 = is_blunder.groupby(my_moves["game_id"]).shift(-2)
    next3 = is_blunder.groupby(my_moves["game_id"]).shift(-3)


    blunder_within_3 = next1.fillna(False) | next2.fillna(False) | next3.fillna(False)

    baseline = is_blunder.mean()

    p_given_long_think = blunder_within_3[my_moves["long_think"]].mean()
    n_long_think = my_moves["long_think"].sum()
    n_blunders_after = blunder_within_3[my_moves["long_think"]].sum()

    ratio = p_given_long_think / baseline

    return {
        "ratio": ratio,
        "p_given_long_think": p_given_long_think,
        "baseline": baseline,
        "n_long_think": n_long_think,
        "n_blunders_after": n_blunders_after,
    }

def m3_greed_conversion(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per game where my win% first reaches >= 80 ("winning point"):
    plies_to_finish, did_convert, max_drawdown (spec section 5, M3).

    Steps:
    - for each game_id, filter is_my_move rows (or all rows? -- win_pct
      at ANY row, mine or opponent's, could hit >= 80 first; decide and
      document which convention you use), find the first ply where the
      MY-perspective win% (see eval_state's my_win_pct pattern from
      features.py) crosses 80
    - games that never reach 80 are excluded from this metric entirely
    - plies_to_finish = game's n_plies - winning_point_ply
    - did_convert = games["result_for_me"] == 1 for that game
    - max_drawdown = the minimum my_win_pct reached AFTER the winning
      point ply (within the same game)
    - return one row per qualifying game with these columns
    """


def m4_clock_burn_map(moves: pd.DataFrame) -> pd.DataFrame:
    """Median + p90 time_spent by move_number, median clock_after by
    move_number, split by perf_type (spec section 5, M4).

    Steps:
    - filter to is_my_move rows
    - group by (perf_type, move_number), aggregate:
      time_spent -> median, and the 90th percentile (Series.quantile(0.9))
      clock_after -> median
    - pattern flag ("slow start -> time pressure >= move X" vs "even
      burn") is a judgment call you make LOOKING at the resulting table/
      chart, not something to compute automatically -- leave that
      out of this function, it's dashboard/analysis-time reasoning
    """

ECO_FAMILY_MAP = {
    # TODO: manual ECO-prefix -> family name mapping, spec section 5 M5
    # mentions Sicilian/French/Dutch/Italian/Catalan/QG specifically.
    # Sicilian is B20-B99, French is C00-C19, Dutch is A80-A99, etc. --
    # look up an ECO code reference table, this needs real chess opening
    # knowledge, not something to guess column-by-column.
}


def m5_opening_surgery(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Group by opening family (from ECO code) + color: n_games, score %,
    mean win% at ply 20, blunder rate in first 20 plies (spec section 5, M5).

    Steps:
    - derive opening_family from games["eco"] using ECO_FAMILY_MAP (map
      the ECO prefix, e.g. games["eco"].str[0] for the letter, or a
      range-based lookup for the two-char code -- decide precision level)
    - group games by (opening_family, my_color): n_games, score % (mean
      of result_for_me)
    - win% at ply 20: needs a join back to moves, filtering ply == 20
      (or nearest available) per game, averaging win_pct (my perspective)
    - blunder rate first 20 plies: filter moves to ply <= 20 AND
      is_my_move, group by game_id -> merge opening_family/color from
      games, aggregate mean(move_quality == "blunder")
    - assemble into one table indexed by (opening_family, color)
    """