"""Derived features (spec section 4). Applied to moves.parquet.

All these columns are computed from moves.parquet (+ eval_cp, now complete
after evaluate.py) and written back to it. Read config.toml for the
move_quality thresholds -- don't hardcode them, the spec says they belong
in config.

Column list (spec section 4):
win_pct, win_pct_delta, move_quality, phase, time_pressure, long_think,
eval_state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tomllib


def add_win_pct(moves: pd.DataFrame) -> pd.Series:
    """Per-row win_pct, from the perspective of whoever moves at fen_before."""
    cp_mover = moves["eval_cp"].where(moves["ply"] % 2 == 1, -moves["eval_cp"])
    win_pct = 50 + 50 * (2 / (1 + np.exp(-0.00368208 * cp_mover)) - 1)

    return win_pct


def add_win_pct_delta(moves: pd.DataFrame) -> pd.Series:
    """win_pct after this move minus win_pct before, from the mover's
    perspective at each row (whoever's turn it was -- me or opponent).
    """
    moves_sorted = moves.sort_values(["game_id", "ply"])
    next_win_pct = moves_sorted.groupby("game_id")["win_pct"].shift(-1)
    delta = (100 - next_win_pct) - moves_sorted["win_pct"]

    return delta

def add_move_quality(moves: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Categorize win_pct_delta into blunder / mistake / inaccuracy / ok,
    from the mover's perspective at each row -- applies to every move,
    mine and my opponent's alike (needed by M6).
    """
    conditions = [
        -moves["win_pct_delta"] >= thresholds["blunder"],
        -moves["win_pct_delta"] >= thresholds["mistake"],
        -moves["win_pct_delta"] >= thresholds["inaccuracy"],
    ]
    choices = ["blunder", "mistake", "inaccuracy"]
    quality = np.select(conditions, choices, default="ok")

    quality_series = pd.Series(quality, index=moves.index)
    quality_masked = quality_series.where(moves["win_pct_delta"].notna(), None)

    return quality_masked

def add_phase(moves: pd.DataFrame) -> pd.Series:
    """opening / middlegame / endgame.

    Decision: kings excluded from the piece count (standard "material"
    convention, only N/B/R/Q count). Decision: opening (ply<=20) takes
    precedence over endgame even if material is already low -- near
    impossible in practice, but explicit rather than accidental.
    """

    piece_placement = moves["fen_before"].str.split(" ").str[0]
    n_pieces = piece_placement.str.count(r"[NBRQnbrq]")

    conditions = [moves["ply"] <= 20, n_pieces <= 6]
    choices = ["opening", "endgame"]
    phase = np.select(conditions, choices, default="middlegame")
    phase_masked = pd.Series(phase, index=moves.index)

    return phase_masked

def add_time_pressure(moves: pd.DataFrame, games: pd.DataFrame, thresholds: dict) -> pd.Series:
    """threshold for the time pressure"""
    base_seconds = moves["game_id"].map(games.set_index("game_id")["base_seconds"])
    return moves["clock_after"] < thresholds["time_pressure_pct"] * base_seconds

def add_long_think(moves: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Flag moves taking > factor x the game's median think time"""
    median_per_game = moves.groupby("game_id")["time_spent"].transform("median")

    return moves["time_spent"] > thresholds["long_think_factor"] * median_per_game

def add_eval_state(moves: pd.DataFrame) -> pd.Series:
    """winning/better/equal/worse/lost from MY win_pct"""
    my_win_pct = moves["win_pct"].where(moves["is_my_move"], 100 - moves["win_pct"])

    conditions = [
        my_win_pct >= 80,
        my_win_pct >= 65,
        my_win_pct >= 35,
        my_win_pct >= 20,
    ]

    choices = ["winning", "better", "equal", "worse"]
    state = np.select(conditions, choices, default="lost")

    state_masked = pd.Series(state, index=moves.index).where(my_win_pct.notna(), None)

    return state_masked

def check_label_sanity(
    moves: pd.DataFrame,
    blunder_rate_min_samples: int,
) -> None:
    """Raise if a source has enough rated moves but no blunders."""
    rated_moves = moves[moves["move_quality"].notna()]

    stats = rated_moves.groupby("eval_source")["move_quality"].agg(
        count="count",
        blunders=lambda qualities: (qualities == "blunder").sum(),
    )

    invalid_sources = stats[
        (stats["count"] >= blunder_rate_min_samples)
        & (stats["blunders"] == 0)
    ]

    if not invalid_sources.empty:
        details = ", ".join(
            f"{source}: {int(row['count'])} rated moves"
            for source, row in invalid_sources.iterrows()
        )
        raise ValueError(
            f"Blunder rate is 0 for: {details}. "
            "Likely an eval_cp convention mismatch "
            "(before vs after the move); check how this source is parsed."
        )


def main() -> None:
    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)
    thresholds = config_data["thresholds"]

    games_path = Path("data/processed/games.parquet")
    moves_path = Path("data/processed/moves.parquet")

    games = pd.read_parquet(games_path)
    moves = pd.read_parquet(moves_path)

    print(f"eval_cp coverage: {(1 - moves['eval_cp'].isna().mean()):.1%}")

    moves["win_pct"] = add_win_pct(moves)
    moves["win_pct_delta"] = add_win_pct_delta(moves)
    moves["move_quality"] = add_move_quality(moves, thresholds)
    check_label_sanity(moves, thresholds["blunder_rate_min_samples"])
    moves["phase"] = add_phase(moves)
    moves["time_pressure"] = add_time_pressure(moves, games, thresholds)
    moves["long_think"] = add_long_think(moves, thresholds)
    moves["eval_state"] = add_eval_state(moves)

    moves.to_parquet(moves_path, index=False)
    print("Features derived and saved to moves.parquet.")

if __name__ == "__main__":
    main()