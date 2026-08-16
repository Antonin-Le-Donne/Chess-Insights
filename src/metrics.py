"""Core metrics (spec section 5) -- the calculations that ARE the project.

Each function: pure, takes DataFrames (+ config where needed), returns a
DataFrame/dict. One dashboard view per metric. Only operates on MY moves
(is_my_move == True) unless the metric's own definition says otherwise.

M1-M5 for now (Milestone 3). M6/M7 come later (Milestone 4).
"""

from __future__ import annotations

import pandas as pd

ECO_FAMILY_MAP = [
    ("A80", "A99", "Dutch"),
    ("B20", "B99", "Sicilian"),
    ("C00", "C19", "French"),
    ("C50", "C59", "Italian"),
    ("D06", "D69", "Queen's Gambit"),
    ("E00", "E09", "Catalan"),
]

def eco_to_family(eco: str) -> str | None:
    for lo,hi, name in ECO_FAMILY_MAP:
        if lo <= eco <= hi:
            return name
    return None

def m1_fatigue_curve(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Mean win_pct_delta + blunder rate of MY moves, bucketed by move_number
    (bins of 5), split by perf_type. Plus an overlay split by game duration
    tercile (spec section 5, M1)."""

    my_moves = moves[moves["is_my_move"]]
    move_bucket = ((my_moves["move_number"] // 5) * 5).rename("move_bucket")
    perf_type = my_moves["game_id"].map(games.set_index("game_id")["perf_type"])
    perf_group = perf_type.where(perf_type == "bullet", "main").rename("perf_group")


    game_duration = moves.groupby("game_id")["time_spent"].sum()
    tercile_per_game = pd.qcut(game_duration, q=3, labels=["short", "medium", "long"])
    duration_tercile = my_moves["game_id"].map(tercile_per_game).rename("duration_tercile")

    grouped = my_moves.groupby([perf_group, move_bucket, duration_tercile]).agg(
        mean_win_pct_delta=("win_pct_delta", "mean"),
        blunder_rate=("move_quality", lambda s: (s == "blunder").mean()),
        )

    return grouped


def m2_time_to_blunder(moves: pd.DataFrame, games: pd.DataFrame) -> dict:
    my_moves = moves[moves["is_my_move"]].sort_values(["game_id", "ply"])
    is_blunder = my_moves["move_quality"] == "blunder"
    perf_type = my_moves["game_id"].map(games.set_index("game_id")["perf_type"]).rename("perf_type")

    next1 = is_blunder.groupby(my_moves["game_id"]).shift(-1)
    next2 = is_blunder.groupby(my_moves["game_id"]).shift(-2)
    next3 = is_blunder.groupby(my_moves["game_id"]).shift(-3)
    blunder_within_3 = next1.fillna(False) | next2.fillna(False) | next3.fillna(False)

    groups = {
        "main": perf_type != "bullet",
        "bullet": perf_type == "bullet",
    }

    results = {}
    for name, mask in groups.items():
        baseline = is_blunder[mask].mean()
        p_given_long_think = blunder_within_3[mask & my_moves["long_think"]].mean()
        n_long_think = my_moves["long_think"][mask].sum()
        n_blunders_after = blunder_within_3[mask & my_moves["long_think"]].sum()
        ratio = p_given_long_think / baseline

        results[name] = {
            "ratio": ratio,
            "p_given_long_think": p_given_long_think,
            "baseline": baseline,
            "n_long_think": n_long_think,
            "n_blunders_after": n_blunders_after,
        }

    return results

def m3_greed_conversion(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per game where my win% first reaches >= 80 ("winning point"):
    plies_to_finish, did_convert, max_drawdown (spec section 5, M3).
    """

    my_win_pct = moves["win_pct"].where(moves["is_my_move"], 100 - moves["win_pct"])

    winning_rows = moves[my_win_pct >= 80]
    winning_point_ply = winning_rows.groupby("game_id")["ply"].min()

    qualifying = games[games["game_id"].isin(winning_point_ply.index)].copy()
    qualifying["winning_point_ply"] = qualifying["game_id"].map(winning_point_ply)
    qualifying["plies_to_finish"] = qualifying["n_plies"] - qualifying["winning_point_ply"]
    qualifying["did_convert"] = qualifying["result_for_me"] == 1

    wp_per_row = moves["game_id"].map(winning_point_ply)
    after = moves[moves["ply"] > wp_per_row]
    max_drawdown = my_win_pct[after.index].groupby(after["game_id"]).min()
    qualifying["max_drawdown"] = qualifying["game_id"].map(max_drawdown)

    return qualifying

def m4_clock_burn_map(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Median + p90 time_spent by move_number, median clock_after by
    move_number, split by perf_type (spec section 5, M4).
    """
    my_moves = moves[moves["is_my_move"]]
    perf_type = my_moves["game_id"].map(games.set_index("game_id")["perf_type"])
    perf_group = perf_type.where(perf_type == "bullet", "main").rename("perf_group")


    grouped = my_moves.groupby([perf_group, "move_number"]).agg(
            median_time_spent=("time_spent", "median"),
            p90_time_spent=("time_spent", lambda s: s.quantile(0.9)),
            median_clock_after=("clock_after", "median"),
            )

    return grouped

def m5_opening_surgery(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Group by opening family (from ECO code) + color: n_games, score %,
    mean MY win% at ply 20, blunder rate in first 20 plies (spec section 5, M5).
    """

    games_20plus = games[games["n_plies"] >= 20].copy()
    games_20plus["opening_family"] = games_20plus["eco"].apply(eco_to_family)

    game_stats = games_20plus.groupby(["opening_family", "my_color"]).agg(
        n_games=("result_for_me", "count"),
        score_pct=("result_for_me", "mean"),
    )

    games_map = games_20plus.set_index("game_id")
    moves_20plus = moves[moves["game_id"].isin(games_20plus["game_id"])].copy()

    moves_20plus["opening_family"] = moves_20plus["game_id"].map(
        games_map["opening_family"]
    )
    moves_20plus["my_color"] = moves_20plus["game_id"].map(
        games_map["my_color"]
    )

    my_win_pct = moves_20plus["win_pct"].where(
        moves_20plus["is_my_move"], 100 - moves_20plus["win_pct"]
    )
    moves_20plus["my_win_pct"] = my_win_pct

    ply20_stats = (
        moves_20plus[moves_20plus["ply"] == 20]
        .groupby(["opening_family", "my_color"])
        .agg(mean_my_win_pct_ply20=("my_win_pct", "mean"))
    )

    early_my_moves = moves_20plus[
        (moves_20plus["ply"] <= 20)
        & (moves_20plus["is_my_move"])
        & (moves_20plus["move_quality"].notna())
    ]

    blunder_stats = early_my_moves.groupby(
        ["opening_family", "my_color"]
    ).agg(
        blunder_rate_first20=(
            "move_quality",
            lambda s: (s == "blunder").mean(),
        )
    )
    return game_stats.join([ply20_stats, blunder_stats]).reset_index()
