"""Core metrics (spec section 5) -- the calculations that ARE the project.

Each function: pure, takes DataFrames (+ config where needed), returns a
DataFrame/dict. One dashboard view per metric. Only operates on MY moves
(is_my_move == True) unless the metric's own definition says otherwise.

M1-M6 metrics. M7 remains planned (Milestone 4).
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
    if pd.isna(eco):
        return None
    for lo, hi, name in ECO_FAMILY_MAP:
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



def mover_cp_loss(
    moves: pd.DataFrame,
    cp_clip: int = 1000,
) -> pd.Series:
    """Centipawn loss of every move, from the perspective of the player
    who made it.

    Uses the evaluation before the move and the evaluation before the
    next move. Both evaluations are converted to the current mover's
    perspective and clipped to +/- cp_clip before differencing.

    Positive values indicate that the mover lost ground.
    """
    if cp_clip <= 0:
        raise ValueError("cp_clip must be positive.")

    ordered = moves.sort_values(["game_id", "ply"]).copy()
    grouped = ordered.groupby("game_id", sort=False)

    next_eval_cp = grouped["eval_cp"].shift(-1)
    next_ply = grouped["ply"].shift(-1)
    next_eval_cp = next_eval_cp.where(
        next_ply.eq(ordered["ply"] + 1)
    )

    # White moves on odd plies; Black moves on even plies.
    # eval_cp is expressed from White's perspective.
    mover_sign = ordered["ply"].mod(2).map({1: 1, 0: -1})

    eval_before = (
        ordered["eval_cp"] * mover_sign
    ).clip(lower=-cp_clip, upper=cp_clip)

    # Keep the current mover's perspective for both evaluations.
    eval_after = (
        next_eval_cp * mover_sign
    ).clip(lower=-cp_clip, upper=cp_clip)

    cp_loss = (eval_before - eval_after).rename("cp_loss")

    # Restore the original row order and index.
    return cp_loss.reindex(moves.index)


def classify_termination(termination: pd.Series) -> pd.Series:
    """Map raw termination strings to stable, readable categories.

    Lichess often reports "Normal" without distinguishing mate,
    resignation, or draws. Keep that category separate instead of guessing.
    When a string mentions both time and insufficient material, classify it
    as time: this breakdown is intended to identify clock-related saves.
    """
    normalized = (
        termination.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    categories = pd.Series(
        "other",
        index=termination.index,
        dtype="object",
        name="termination_category",
    )

    # Order matters: time takes precedence over insufficient material in
    # compound descriptions such as "drawn by timeout vs insufficient material".
    rules = [
        ("time", "time"),
        ("timeout", "time"),
        ("checkmate", "checkmate"),
        ("resignation", "resignation"),
        ("stalemate", "stalemate"),
        ("repetition", "repetition"),
        ("insufficient material", "insufficient_material"),
        ("agreement", "agreement"),
        ("abandoned", "abandoned"),
    ]

    for keyword, category in rules:
        matches = normalized.str.contains(keyword, regex=False, na=False)
        categories.loc[matches & categories.eq("other")] = category

    # Exact Lichess "Normal" remains its own category; it hides the reason.
    categories.loc[normalized.eq("normal")] = "normal_lichess"

    return categories


def find_lost_stretches(
    moves: pd.DataFrame,
    lost_pct: float,
    min_plies: int,
    side: str = "me",
) -> pd.DataFrame:
    """Find the first qualifying lost stretch for me or my opponent.

    The selected side's win probability must remain <= lost_pct for at
    least min_plies consecutive plies. A missing ply breaks the stretch.

    Returns one row per qualifying game with game_id, lost_from_ply, and
    win_pct_at_lost, expressed from the selected side's perspective.
    """
    if min_plies < 1:
        raise ValueError("min_plies must be at least 1.")

    if side not in {"me", "opponent"}:
        raise ValueError("side must be either 'me' or 'opponent'.")

    ordered = moves.sort_values(["game_id", "ply"]).copy()

    # win_pct is from the mover's perspective. Convert each row to my POV.
    my_win_pct = ordered["win_pct"].where(
        ordered["is_my_move"],
        100 - ordered["win_pct"],
    )

    if side == "me":
        ordered["side_win_pct"] = my_win_pct
    else:
        ordered["side_win_pct"] = 100 - my_win_pct

    ordered["_is_lost"] = (
        ordered["side_win_pct"].le(lost_pct).fillna(False)
    )

    grouped = ordered.groupby("game_id", sort=False)
    previous_flag = grouped["_is_lost"].shift().eq(True)
    previous_ply = grouped["ply"].shift()

    # A new run starts when a lost stretch begins or when consecutive
    # lost rows are interrupted by a missing ply.
    starts_run = ordered["_is_lost"] & (
        ~previous_flag | ordered["ply"].ne(previous_ply + 1)
    )

    ordered["_run_id"] = starts_run.groupby(
        ordered["game_id"],
        sort=False,
    ).cumsum()

    lost_rows = ordered.loc[ordered["_is_lost"]]
    runs = (
        lost_rows.groupby(
            ["game_id", "_run_id"],
            as_index=False,
            sort=False,
        )
        .agg(
            run_length=("ply", "size"),
            lost_from_ply=("ply", "first"),
            win_pct_at_lost=("side_win_pct", "first"),
        )
    )

    qualifying_runs = (
        runs.loc[runs["run_length"] >= min_plies]
        .sort_values(["game_id", "lost_from_ply"])
    )

    return (
        qualifying_runs.drop_duplicates("game_id", keep="first")
        [["game_id", "lost_from_ply", "win_pct_at_lost"]]
        .reset_index(drop=True)
    )


def m6_swindle_profile(
    moves: pd.DataFrame,
    games: pd.DataFrame,
    lost_pct: float,
    min_plies: int,
    cp_error_threshold: int,
    swing_to_pct: float = 40,
) -> dict[str, pd.DataFrame]:
    """M6 -- swindle profile.

    A game qualifies when my win probability remains <= lost_pct for at
    least min_plies consecutive plies. The control group applies the same
    rule to the opponent's perspective.

    Opponent errors are measured in centipawns. A game-level rescue
    opportunity is an opponent move after the lost stretch begins that
    raises my win probability from <= lost_pct to >= swing_to_pct.

    Returns DataFrames named "results", "control", "opponent_moves",
    "opportunities", and "saves_by_termination".
    """
    if not 0 <= lost_pct < swing_to_pct <= 100:
        raise ValueError(
            "Thresholds must satisfy 0 <= lost_pct < swing_to_pct <= 100."
        )

    if cp_error_threshold <= 0:
        raise ValueError("cp_error_threshold must be positive.")

    if "termination" not in games.columns:
        raise KeyError(
            "games must contain a 'termination' column for the M6 saved-game breakdown."
        )

    # Keep bullet separate; blitz and rapid are combined into main.
    perf_type_by_game = games.set_index("game_id")["perf_type"]

    # Identify games in which I reached a qualifying lost stretch.
    lost = find_lost_stretches(
        moves,
        lost_pct,
        min_plies,
        side="me",
    )
    lost_games = lost.merge(
        games,
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    lost_games["perf_group"] = lost_games["perf_type"].where(
        lost_games["perf_type"] == "bullet",
        "main",
    )

    # Wins = 1, draws = 0.5, losses = 0 from my perspective.
    lost_games["_win"] = lost_games["result_for_me"].eq(1)
    lost_games["_draw"] = lost_games["result_for_me"].eq(0.5)
    lost_games["_loss"] = lost_games["result_for_me"].eq(0)
    lost_games["is_saved"] = lost_games["result_for_me"].ge(0.5)
    lost_games["expected_score_at_lost"] = (
        lost_games["win_pct_at_lost"] / 100
    )

    grouped_results = (
        lost_games.groupby("perf_group", as_index=False)
        .agg(
            n=("game_id", "size"),
            wins=("_win", "sum"),
            draws=("_draw", "sum"),
            losses=("_loss", "sum"),
            score_pct=("result_for_me", "mean"),
            expected_score=("expected_score_at_lost", "mean"),
        )
    )

    results = pd.DataFrame({"perf_group": ["main", "bullet"]}).merge(
        grouped_results,
        on="perf_group",
        how="left",
    )
    for column in ["n", "wins", "draws", "losses"]:
        results[column] = results[column].fillna(0).astype(int)
    results["score_vs_expected"] = (
        results["score_pct"] - results["expected_score"]
    )

    # Control group: games where the opponent had the same kind of lost
    # stretch. Their actual score is 1 - my result_for_me.
    opponent_lost = find_lost_stretches(
        moves,
        lost_pct,
        min_plies,
        side="opponent",
    )
    opponent_lost_games = opponent_lost.merge(
        games,
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    opponent_lost_games["perf_group"] = (
        opponent_lost_games["perf_type"].where(
            opponent_lost_games["perf_type"] == "bullet",
            "main",
        )
    )
    opponent_lost_games["actual_score"] = (
        1 - opponent_lost_games["result_for_me"]
    )
    opponent_lost_games["expected_score"] = (
        opponent_lost_games["win_pct_at_lost"] / 100
    )

    grouped_control = (
        opponent_lost_games.groupby("perf_group", as_index=False)
        .agg(
            n=("game_id", "size"),
            actual_score=("actual_score", "mean"),
            expected_score=("expected_score", "mean"),
        )
    )
    control = pd.DataFrame({"perf_group": ["main", "bullet"]}).merge(
        grouped_control,
        on="perf_group",
        how="left",
    )
    control["n"] = control["n"].fillna(0).astype(int)
    control["score_vs_expected"] = (
        control["actual_score"] - control["expected_score"]
    )

    # Move-level opponent CP errors. These states do not require a
    # six-ply lost stretch; only valid before/after evaluations count.
    cp_loss = mover_cp_loss(moves)
    opponent_moves = moves.loc[~moves["is_my_move"]].copy()
    opponent_moves["cp_loss"] = cp_loss.reindex(opponent_moves.index)
    opponent_moves["my_win_pct"] = 100 - opponent_moves["win_pct"]
    opponent_moves["perf_type"] = opponent_moves["game_id"].map(
        perf_type_by_game
    )
    opponent_moves["perf_group"] = opponent_moves["perf_type"].where(
        opponent_moves["perf_type"] == "bullet",
        "main",
    )
    opponent_moves["state"] = "other"
    opponent_moves.loc[
        opponent_moves["my_win_pct"].le(lost_pct), "state"
    ] = "lost"
    opponent_moves.loc[
        opponent_moves["my_win_pct"].ge(100 - lost_pct), "state"
    ] = "winning"
    opponent_moves = opponent_moves.dropna(subset=["cp_loss"]).copy()
    opponent_moves["cp_error"] = (
        opponent_moves["cp_loss"] >= cp_error_threshold
    )

    opponent_move_stats = (
        opponent_moves.groupby(
            ["perf_group", "state"],
            as_index=False,
        )
        .agg(
            n=("cp_loss", "size"),
            n_cp_errors=("cp_error", "sum"),
            cp_error_rate=("cp_error", "mean"),
        )
    )

    # Game-level opportunity: after the lost stretch starts, an opponent
    # move raises my win chance from <= lost_pct to >= swing_to_pct.
    ordered_moves = moves.sort_values(["game_id", "ply"]).copy()
    grouped_moves = ordered_moves.groupby("game_id", sort=False)
    ordered_moves["_next_ply"] = grouped_moves["ply"].shift(-1)
    ordered_moves["_next_is_my_move"] = (
        grouped_moves["is_my_move"].shift(-1)
    )
    ordered_moves["_my_win_pct_after"] = (
        grouped_moves["win_pct"].shift(-1)
    )
    ordered_moves["_my_win_pct_before"] = ordered_moves["win_pct"].where(
        ordered_moves["is_my_move"],
        100 - ordered_moves["win_pct"],
    )
    lost_from_by_game = lost_games.set_index("game_id")["lost_from_ply"]
    ordered_moves["_lost_from_ply"] = ordered_moves["game_id"].map(
        lost_from_by_game
    )

    ordered_moves["_opponent_swing"] = (
        ~ordered_moves["is_my_move"]
        & ordered_moves["_lost_from_ply"].notna()
        & ordered_moves["ply"].ge(ordered_moves["_lost_from_ply"])
        & ordered_moves["_my_win_pct_before"].le(lost_pct)
        & ordered_moves["ply"].add(1).eq(ordered_moves["_next_ply"])
        & ordered_moves["_next_is_my_move"].eq(True)
        & ordered_moves["_my_win_pct_after"].ge(swing_to_pct)
    ).fillna(False)

    swing_by_game = ordered_moves.groupby("game_id")["_opponent_swing"].any()
    lost_games["opponent_gave_chance"] = (
        lost_games["game_id"].map(swing_by_game).fillna(False).astype(bool)
    )

    grouped_opportunities = (
        lost_games.groupby(
            ["perf_group", "opponent_gave_chance"],
            as_index=False,
        )
        .agg(
            n_games=("game_id", "size"),
            n_saved=("is_saved", "sum"),
            saved_rate=("is_saved", "mean"),
            score_pct=("result_for_me", "mean"),
        )
    )
    combinations = pd.DataFrame([
        {"perf_group": "main", "opponent_gave_chance": True},
        {"perf_group": "main", "opponent_gave_chance": False},
        {"perf_group": "bullet", "opponent_gave_chance": True},
        {"perf_group": "bullet", "opponent_gave_chance": False},
    ])
    opportunities = combinations.merge(
        grouped_opportunities,
        on=["perf_group", "opponent_gave_chance"],
        how="left",
    )
    for column in ["n_games", "n_saved"]:
        opportunities[column] = opportunities[column].fillna(0).astype(int)

    total_lost_games = lost_games.groupby("perf_group")["game_id"].size()
    opportunities["n_lost_games"] = (
        opportunities["perf_group"].map(total_lost_games).fillna(0).astype(int)
    )
    denominator = opportunities["n_lost_games"].where(
        opportunities["n_lost_games"] > 0
    )
    opportunities["share_of_lost_games"] = (
        opportunities["n_games"] / denominator
    )

    # Termination breakdown among saved games (win or draw), by whether
    # an opponent swing was identified. Shares use saved games in each
    # (perf_group, opportunity) group as the denominator.
    saved_games = lost_games.loc[lost_games["is_saved"]].copy()
    saved_games["termination_category"] = classify_termination(
        saved_games["termination"]
    )
    grouped_terminations = (
        saved_games.groupby(
            ["perf_group", "opponent_gave_chance", "termination_category"],
            as_index=False,
        )
        .agg(n_saved=("game_id", "size"))
    )
    saved_totals = (
        saved_games.groupby(["perf_group", "opponent_gave_chance"])
        .size()
        .rename("total_saved")
        .reset_index()
    )
    saves_by_termination = grouped_terminations.merge(
        saved_totals,
        on=["perf_group", "opponent_gave_chance"],
        how="left",
    )
    saves_by_termination["share_of_saved_games"] = (
        saves_by_termination["n_saved"] / saves_by_termination["total_saved"]
    )

    return {
        "results": results,
        "control": control,
        "opponent_moves": opponent_move_stats,
        "opportunities": opportunities,
        "saves_by_termination": saves_by_termination,
    }


if __name__ == "__main__":
    moves = pd.read_parquet("data/processed/moves.parquet")
    games = pd.read_parquet("data/processed/games.parquet")

    print("--- M1 fatigue curve ---")
    pd.set_option("display.max_rows", None)
    games_40plus = games[games["n_plies"] >= 40]
    moves_40plus = moves[moves["game_id"].isin(games_40plus["game_id"])]
    print(m1_fatigue_curve(moves_40plus, games_40plus))
    print(m1_fatigue_curve(moves, games))

    print("\n--- M2 time to blunder ---")
    print(m2_time_to_blunder(moves, games))

    print("\n--- M3 greed conversion ---")
    m3 = m3_greed_conversion(moves, games)
    print(m3)
    print("Number of qualifying games:", len(m3))
    print("Conversion rate:", m3["did_convert"].mean())
    print("Median plies_to_finish:", m3["plies_to_finish"].median())
    print(
        "Median plies_to_finish (converted only):",
        m3.loc[m3["did_convert"], "plies_to_finish"].median(),
    )

    m3_main = m3[m3["perf_type"] != "bullet"]
    print("Main — number of qualifying games:", len(m3_main))
    print("Main — conversion rate:", m3_main["did_convert"].mean())
    print("Main — median plies_to_finish:", m3_main["plies_to_finish"].median())
    print(
        "Main — median plies_to_finish (converted only):",
        m3_main.loc[m3_main["did_convert"], "plies_to_finish"].median(),
    )

    m3_bullet = m3[m3["perf_type"] == "bullet"]
    print("Bullet — number of qualifying games:", len(m3_bullet))
    print("Bullet — conversion rate:", m3_bullet["did_convert"].mean())
    print("Bullet — median plies_to_finish:", m3_bullet["plies_to_finish"].median())
    print(
        "Bullet — median plies_to_finish (converted only):",
        m3_bullet.loc[m3_bullet["did_convert"], "plies_to_finish"].median(),
    )

    print("\n--- M4 clock burn map ---")
    m4 = m4_clock_burn_map(moves, games)
    print(m4)
    main_group = m4.loc["main"]
    onset = main_group.loc[main_group["median_clock_after"] < 36].index.min()
    print("Time pressure onset (move number):", onset)
    bullet_group = m4.loc["bullet"]
    onset_bullet = bullet_group.loc[
        bullet_group["median_clock_after"] < (0.10 * 60)
    ].index.min()
    print("Bullet — time pressure onset (move number):", onset_bullet)

    print("\n--- M5 opening surgery ---")
    print(m5_opening_surgery(moves, games))

    print("\n--- M1 retest: games of at least 40 plies ---")
    games_40plus = games[games["n_plies"] >= 40]
    moves_40plus = moves[moves["game_id"].isin(games_40plus["game_id"])]
    print(m1_fatigue_curve(moves_40plus, games_40plus))

    print("\n--- M5 retest: rating difference by opening family ---")
    games["rating_diff"] = games["opp_rating"] - games["my_rating"]
    opening_family = games["eco"].apply(eco_to_family)
    print(games.groupby(opening_family)["rating_diff"].mean())

    print("\n--- M6 termination-category diagnostic ---")
    print(classify_termination(games["termination"]).value_counts(dropna=False))

    print("\n--- M6 swindle profile ---")
    m6 = m6_swindle_profile(
        moves,
        games,
        lost_pct=20,
        min_plies=6,
        cp_error_threshold=200,
        swing_to_pct=40,
    )
    print("\nM6 — Results of games that reached a lost stretch")
    print(m6["results"].to_string(index=False))
    print("\nM6 — Control group: opponents' score vs expected")
    print(m6["control"].to_string(index=False))
    print("\nM6 — Opponent centipawn error rates by my position state")
    print(m6["opponent_moves"].to_string(index=False))
    print("\nM6 — Rescue opportunity rate and save rate with / without an opportunity")
    print(m6["opportunities"].to_string(index=False))
    print("\nM6 — Saved games by termination and opportunity")
    print(m6["saves_by_termination"].to_string(index=False))
