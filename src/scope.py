from __future__ import annotations

import numpy as np
import pandas as pd


def filter_current_regime(
    games: pd.DataFrame,
    moves: pd.DataFrame,
    floors: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only games from (platform, perf_type) groups with a configured
    rating floor, and only rows at/above that floor. Groups absent from
    `floors` are dropped entirely (currently: any rapid, chess.com blitz).

    floors: {"lichess_blitz": 1700, ...} -- keys are f"{platform}_{perf_type}",
    matching config.toml's [current_regime] section.
    """

    group_key = games["platform"] + "_" + games["perf_type"]

    rating_floor = group_key.map(floors)

    keep = rating_floor.notna() & (games["my_rating"] >= rating_floor)

    filtered_games = games[keep].copy()

    surviving_game_ids = filtered_games["game_id"]

    filtered_moves = moves[
        moves["game_id"].isin(surviving_game_ids)
    ].copy()

    return filtered_games, filtered_moves


def rating_trajectory(games: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean my_rating per (platform, perf_type), for the plateau chart
    that justifies the floors above. One row per (month, platform, perf_type).
    """

    trajectory = (
        games
        .set_index("date_utc")
        .groupby([
            pd.Grouper(freq="ME"),
            "platform",
            "perf_type"
        ])["my_rating"]
        .agg(
            mean_rating="mean",
            n="count"
        )
        .reset_index()
    )

    return trajectory


def quality_of_play_trajectory(moves: pd.DataFrame,games: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean win% given away per move (-win_pct_delta, MY moves only),
    pooled across platform and perf_type.

    Unlike my_rating, this doesn't depend on which platform/pool a game was
    played in (lichess and chess.com ratings aren't comparable -- see
    chess_insights_spec.md), so it's the one trend that's directly
    comparable across the whole history in a single line. Lower is better
    (fewer win% points given up on average). Not a rating estimate -- ACPL/
    win_pct-loss-to-Elo conversion is a rough community heuristic at best,
    this is a consistent quality-of-play score, not a claimed Elo.
    """

    moves_enriched = moves.merge(
        games[["game_id", "date_utc"]],
        on="game_id",
        how="left"
    )

    my_moves = moves_enriched[
        moves_enriched["is_my_move"]
    ].copy()

    my_moves["win_pct_lost"] = -my_moves["win_pct_delta"]

    trajectory = (
        my_moves
        .set_index("date_utc")
        .groupby(
            pd.Grouper(freq="ME")
        )
        .agg(
            mean_win_pct_lost=("win_pct_lost", "mean"),
            n=("win_pct_lost", "count")
        )
        .reset_index()
    )

    return trajectory


def per_game_quality(
    moves: pd.DataFrame,
    games: pd.DataFrame
) -> pd.DataFrame:
    """One row per game: mean win% lost per move (MY moves only, that game
    only), joined with platform, perf_type, my_rating. Game-level building
    block for performance_rating_estimate -- unlike quality_of_play_trajectory
    (which pools everything into a monthly trend), this keeps one score per
    game so it can be regressed against that game's own my_rating.
    """

    my_moves = moves[
        moves["is_my_move"]
    ].copy()

    my_moves["win_pct_lost"] = -my_moves["win_pct_delta"]

    quality_per_game = (
        my_moves
        .groupby("game_id")["win_pct_lost"]
        .mean()
        .rename("quality_score")
        .reset_index()
    )

    games_quality = games[
        [
            "game_id",
            "platform",
            "perf_type",
            "my_rating"
        ]
    ].merge(
        quality_per_game,
        on="game_id",
        how="inner"
    )

    return games_quality


def performance_rating_estimate(
    games_quality: pd.DataFrame,
    platform: str,
    perf_type: str,
    current_quality: float,
) -> dict:
    """Fit my_rating ~ quality_score within one (platform, perf_type) group
    (games_quality is per_game_quality's output), then predict the rating
    implied by `current_quality` (e.g. the current-regime mean quality for
    that same group).

    Fit separately per (platform, perf_type) -- ratings aren't comparable
    across platforms/time controls (see chess_insights_spec.md), so pooling
    the regression would reintroduce that exact problem.

    Returns e.g. {"predicted_rating": ..., "n_games": ..., "r2": ...} --
    r2 (or another fit-quality indicator) must be reported alongside the
    prediction, not just the number alone: a regression on one coarse,
    noisy feature over a few hundred/thousand games can fit weakly, and
    that needs to be visible, not hidden behind a clean-looking output.
    """

    group = games_quality[
        (games_quality["platform"] == platform)
        & (games_quality["perf_type"] == perf_type)
    ].dropna(
        subset=["quality_score", "my_rating"]
    )

    n_games = len(group)

    if n_games < 2:
        return {
            "predicted_rating": float("nan"),
            "n_games": n_games,
            "r2": float("nan"),
        }

    quality = group["quality_score"].to_numpy()
    ratings = group["my_rating"].to_numpy()

    if len(np.unique(quality)) < 2:
        return {
            "predicted_rating": float("nan"),
            "n_games": n_games,
            "r2": float("nan"),
        }

    slope, intercept = np.polyfit(
        quality,
        ratings,
        deg=1
    )

    predicted_rating = (
        slope * current_quality
        + intercept
    )

    fitted_ratings = (
        slope * quality
        + intercept
    )

    ss_res = ((ratings - fitted_ratings) ** 2).sum()
    ss_tot = ((ratings - ratings.mean()) ** 2).sum()

    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = float("nan")

    return {
        "predicted_rating": predicted_rating,
        "n_games": n_games,
        "r2": r2,
    }


def m1_peak_by_move(
    moves: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Blunder rate of MY moves by move bucket, pooled across all game
    duration terciles. Used by the current-regime comparison page so the
    headline peak is based on the real weighted move population.
    """

    my_moves = moves[moves["is_my_move"]]

    move_bucket = (
        (my_moves["move_number"] // 5) * 5
    ).rename("move_bucket")

    perf_type = my_moves["game_id"].map(
        games.set_index("game_id")["perf_type"]
    )

    perf_group = perf_type.where(
        perf_type == "bullet",
        "main"
    ).rename("perf_group")

    grouped = my_moves.groupby([
        perf_group,
        move_bucket
    ]).agg(
        mean_win_pct_delta=("win_pct_delta", "mean"),
        blunder_rate=("move_quality", lambda s: (s == "blunder").mean()),
    )

    return grouped