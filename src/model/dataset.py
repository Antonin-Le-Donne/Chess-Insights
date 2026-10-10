"""V2 dataset construction (chess_insights_spec.md, section "V2 -- Personal
blunder-risk model").

Builds X, y for the blunder-risk classifier: one row per MY move (ply > 10,
opening book excluded), y = 1 if that move was a blunder. Scoped to the
current-regime games only (src/scope.py's filter_current_regime) -- see
PROJECT_TRACKER.md's decisions log for why: training on the full multi-year
history would mix "2023 me" and "2026 me" into one model, the same drift
problem the Actual Me page exists to correct for.

Most contextual features already exist on moves.parquet (features.py already
computed win_pct, win_pct_delta, move_quality, phase, time_pressure,
long_think, eval_state) -- this module assembles and reshapes those rather
than re-deriving them, and adds the genuinely new ones: rolling time
features, eval volatility, material/complexity features (need a python-chess
Board reconstructed from fen_before), and the opening-family + game-level
join.
"""

from __future__ import annotations

import chess
import numpy as np
import pandas as pd

from src.metrics import eco_to_family


def add_target(moves: pd.DataFrame) -> pd.Series:
    """y = 1 if this MY move is a blunder, else 0. Simple version per spec
    (this move only -- the "within next 3 moves" variant is a second
    experiment, not this function).
    """
    y = (moves["move_quality"] == "blunder").astype(int)

    return y


def add_rolling_time_features(moves: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean of MY last 3 time_spent values, previous time spent,
    and a long_think flag on the PREVIOUS move.
    """

    original_index = moves.index

    moves_sorted = moves.sort_values(
        ["game_id", "ply"]
    ).copy()

    groups = moves_sorted.groupby(
        ["game_id", "is_my_move"]
    )

    moves_sorted["rolling_time_mean_3"] = groups[
        "time_spent"
    ].transform(
        lambda s: (
            s.shift(1)
            .rolling(
                3,
                min_periods=1
            )
            .mean()
        )
    )

    moves_sorted["previous_time_spent"] = groups[
        "time_spent"
    ].shift(1)

    moves_sorted["previous_long_think"] = (
        groups["long_think"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    features = moves_sorted[
        [
            "rolling_time_mean_3",
            "previous_time_spent",
            "previous_long_think",
        ]
    ]

    return features.reindex(original_index)


def add_eval_volatility(moves: pd.DataFrame) -> pd.Series:
    """Std of win_pct over the last 6 plies (both movers, not just mine) --
    a cheap proxy for "how sharp/unstable is this position", independent of
    whose move it is.
    """
    moves_sorted = moves.sort_values(
        ["game_id", "ply"]
    )

    volatility = (
        moves_sorted
        .groupby("game_id")["win_pct"]
        .transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    6,
                    min_periods=2
                )
                .std()
            )
        )
    )

    return volatility.reindex(moves.index)


def add_previous_eval_and_clock(moves: pd.DataFrame,) -> pd.DataFrame:
    """Pre-move eval and clock context: what the engine and clock showed
    right BEFORE this move was played, not after. 
    """

    moves_sorted = moves.sort_values(
        ["game_id", "ply"]
    ).copy()

    eval_before_move = (
        moves_sorted
        .groupby("game_id")["win_pct"]
        .transform(
            lambda s: 100 - s.shift(1)
        )
    )

    clock_before_move = (
        moves_sorted
        .groupby(
            ["game_id", "is_my_move"]
        )["clock_after"]
        .transform(
            lambda s: s.shift(1)
        )
    )

    features = pd.DataFrame(
        {
            "eval_before_move": eval_before_move,
            "clock_before_move": clock_before_move,
        },
        index=moves_sorted.index,
    )

    return features.reindex(moves.index)


def add_material_features(moves: pd.DataFrame,games: pd.DataFrame,) -> pd.DataFrame:
    """Material count, material imbalance (mine - opponent's), queens-on-
    board flag -- all read off chess.Board(fen_before). One Board per row,
    so this is the slow part of dataset.py; consider it a one-time cache
    (save the result, don't recompute on every run) once it works.
    """

    moves_with_color = moves.join(
        games.set_index("game_id")["my_color"],
        on="game_id"
    )

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    material_count = []
    material_imbalance = []
    queens_on_board = []

    for _, row in moves_with_color.iterrows():
        board = chess.Board(row["fen_before"])

        white_material = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in piece_values.items()
        )

        black_material = sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in piece_values.items()
        )

        if row["my_color"] == "white":
            my_material = white_material
            opp_material = black_material
        else:
            my_material = black_material
            opp_material = white_material

        material_count.append(
            my_material + opp_material
        )

        material_imbalance.append(
            my_material - opp_material
        )

        queens_on_board.append(
            bool(
                board.pieces(chess.QUEEN, chess.WHITE)
                or board.pieces(chess.QUEEN, chess.BLACK)
            )
        )

    features = pd.DataFrame(
        {
            "material_count": material_count,
            "material_imbalance": material_imbalance,
            "queens_on_board": queens_on_board,
        },
        index=moves_with_color.index,
    )

    return features.reindex(moves.index)


def add_complexity_features(moves: pd.DataFrame,games: pd.DataFrame,) -> pd.DataFrame:
    """Position complexity proxies from chess.Board(fen_before):
    number of legal moves, my hanging/attacked pieces, and checks available
    to whoever is about to move.

    fen_before already represents the position before the current move, so
    the board's side to move is the mover for that row and no shift is needed.
    """

    moves_with_color = moves.join(
        games.set_index("game_id")["my_color"],
        on="game_id"
    )

    n_legal_moves = []
    hanging_pieces = []
    checks_available = []

    for _, row in moves_with_color.iterrows():
        board = chess.Board(row["fen_before"])

        if row["my_color"] == "white":
            my_color = chess.WHITE
            opp_color = chess.BLACK
        else:
            my_color = chess.BLACK
            opp_color = chess.WHITE

        legal_moves = list(board.legal_moves)

        n_legal_moves.append(
            len(legal_moves)
        )

        hanging_count = 0

        for square, piece in board.piece_map().items():
            if piece.color != my_color:
                continue

            if piece.piece_type == chess.KING:
                continue

            attackers = len(
                board.attackers(
                    opp_color,
                    square
                )
            )

            defenders = len(
                board.attackers(
                    my_color,
                    square
                )
            )

            if attackers > defenders:
                hanging_count += 1

        hanging_pieces.append(
            hanging_count
        )

        checks_available.append(
            sum(
                board.gives_check(move)
                for move in legal_moves
            )
        )

    features = pd.DataFrame(
        {
            "n_legal_moves": n_legal_moves,
            "hanging_pieces": hanging_pieces,
            "checks_available": checks_available,
        },
        index=moves_with_color.index,
    )

    return features.reindex(moves.index)


def assemble_features(
    moves: pd.DataFrame,
    games: pd.DataFrame,
    time_pressure_pct: float,
) -> pd.DataFrame:
    """Join everything into one row-per-my-move feature table."""

    rolling_features = add_rolling_time_features(
        moves
    )

    eval_volatility = add_eval_volatility(
        moves
    ).rename("eval_volatility")

    material_features = add_material_features(
        moves,
        games
    )

    complexity_features = add_complexity_features(
        moves,
        games
    )

    previous_eval_and_clock = add_previous_eval_and_clock(
        moves
    )

    dataset = pd.concat(
        [
            moves,
            rolling_features,
            eval_volatility,
            material_features,
            complexity_features,
            previous_eval_and_clock,
        ],
        axis=1
    )

    dataset = dataset[
        dataset["is_my_move"]
        & (dataset["ply"] > 10)
        & dataset["move_quality"].notna()
    ].copy()

    game_context = games[
        [
            "game_id",
            "my_rating",
            "opp_rating",
            "perf_type",
            "eco",
            "date_utc",
            "base_seconds",
            "increment_seconds",
        ]
    ]

    dataset = dataset.merge(
        game_context,
        on="game_id",
        how="left"
    )

    dataset["rating_diff"] = (
        dataset["my_rating"]
        - dataset["opp_rating"]
    )

    dataset["opening_family"] = dataset[
        "eco"
    ].apply(
        eco_to_family
    )

    dataset["time_pressure_before_move"] = (
        (dataset["base_seconds"] == 0)
        | (
            dataset["clock_before_move"]
            < time_pressure_pct * dataset["base_seconds"]
        )
    )

    dataset["clock_before_move_pct"] = (
        dataset["clock_before_move"]
        / dataset["base_seconds"]
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )

    dataset["has_increment"] = (
        dataset["increment_seconds"] > 0
    )

    conditions = [
        dataset["eval_before_move"] >= 80,
        dataset["eval_before_move"] >= 65,
        dataset["eval_before_move"] >= 35,
        dataset["eval_before_move"] >= 20,
    ]

    choices = [
        "winning",
        "better",
        "equal",
        "worse",
    ]

    dataset["eval_state_before_move"] = np.select(
        conditions,
        choices,
        default="lost"
    )

    dataset["y"] = add_target(
        dataset
    )

    return dataset


def temporal_game_split(
    dataset: pd.DataFrame,
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by game_id, never by row (leakage -- moves in the same game are
    correlated). Temporal: oldest 70% of games -> train, next 15% -> val,
    most recent 15% -> test, ordered by games["date_utc"].
    """

    games_sorted = games.sort_values(
        "date_utc"
    )

    n_games = len(games_sorted)

    train_end = int(
        n_games * 0.70
    )

    val_end = int(
        n_games * 0.85
    )

    train_game_ids = games_sorted.iloc[
        :train_end
    ]["game_id"]

    val_game_ids = games_sorted.iloc[
        train_end:val_end
    ]["game_id"]

    test_game_ids = games_sorted.iloc[
        val_end:
    ]["game_id"]

    train = dataset[
        dataset["game_id"].isin(train_game_ids)
    ].copy()

    val = dataset[
        dataset["game_id"].isin(val_game_ids)
    ].copy()

    test = dataset[
        dataset["game_id"].isin(test_game_ids)
    ].copy()

    return train, val, test
