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

from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from src.metrics import eco_to_family
from src.scope import filter_current_regime


def add_target(moves: pd.DataFrame) -> pd.Series:
    """y = 1 if this MY move is a blunder, else 0. Simple version per spec
    (this move only -- the "within next 3 moves" variant is a second
    experiment, not this function).
    """
    y = (moves["move_quality"] == "blunder").astype(int)

    return y


def add_rolling_time_features(
    moves: pd.DataFrame,
    long_think_factor: float,
) -> pd.DataFrame:
    """Rolling mean of MY last 3 time_spent values, previous time spent,
    and a long_think flag on the PREVIOUS move.

    The flag uses a per-player expanding median including that previous
    move, without future information. Keep NaN until five valid times
    are available; otherwise use 1.0 for true and 0.0 for false.
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
                min_periods=1,
            )
            .mean()
        )
    )

    moves_sorted["previous_time_spent"] = groups[
        "time_spent"
    ].shift(1)

    cumulative_median = groups["time_spent"].transform(
    # Five own-move observations match the opening cutoff (ply > 10);
    # missing times or insufficient prior history still produce NaN.
        lambda s: s.expanding(min_periods=5).median()
    )

    moves_sorted["past_only_long_think"] = (
        (
            moves_sorted["time_spent"]
            > long_think_factor * cumulative_median
        )
        .astype(float)
        .where(
            cumulative_median.notna()
            & moves_sorted["time_spent"].notna()
        )
    )

    moves_sorted["previous_long_think"] = moves_sorted.groupby(
        ["game_id", "is_my_move"]
    )["past_only_long_think"].shift(1)

    features = moves_sorted[
        [
            "rolling_time_mean_3",
            "previous_time_spent",
            "previous_long_think",
        ]
    ]

    return features.reindex(original_index)


def add_eval_volatility(moves: pd.DataFrame) -> pd.Series:
    """Compute the standard deviation of win_pct over the current and
    previous five plies, with every evaluation expressed from my perspective.

    The current row is included because win_pct evaluates fen_before,
    which is already known before the move and captures the latest position.
    """

    moves_sorted = moves.sort_values(
        ["game_id", "ply"]
    )

    win_pct_mine = moves_sorted["win_pct"].where(
        moves_sorted["is_my_move"],
        100 - moves_sorted["win_pct"],
    )

    volatility = (
        win_pct_mine
        .groupby(moves_sorted["game_id"])
        .transform(
            lambda s: (
                s.rolling(
                    6,
                    min_periods=2,
                ).std()
            )
        )
    )

    return volatility.reindex(moves.index)


def add_previous_eval_and_clock(moves: pd.DataFrame,) -> pd.DataFrame:
    """Pre-move evaluation and clock context: the position evaluation and
    remaining time available immediately before the move was played.
    Evaluations are expressed from my perspective.
    """

    moves_sorted = moves.sort_values(
        ["game_id", "ply"]
    ).copy()

    eval_before_move = moves_sorted["win_pct"].where(
        moves_sorted["is_my_move"],
        100 - moves_sorted["win_pct"],
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
    long_think_factor: float,
    blunder_threshold: float,
) -> pd.DataFrame:
    """Join everything into one row-per-my-move feature table."""

    rolling_features = add_rolling_time_features(
        moves,
        long_think_factor,
    )

    eval_volatility = add_eval_volatility(
        moves
    ).rename("eval_volatility")

    material_features = add_material_features(
        moves,
        games,
    )

    complexity_features = add_complexity_features(
        moves,
        games,
    )

    previous_eval_and_clock = add_previous_eval_and_clock(
        moves,
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
        axis=1,
    )

    # Keep only moves where a blunder is possible. Below the blunder
    # threshold, a loss of at least that many points is impossible by
    # definition. This removes rows with no possible positive label
    # without discarding any actual blunders.
    dataset = dataset[
        dataset["is_my_move"]
        & (dataset["ply"] > 10)
        & dataset["move_quality"].notna()
        & (dataset["eval_before_move"] >= blunder_threshold)
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
        how="left",
    )

    dataset["rating_diff"] = (
        dataset["my_rating"]
        - dataset["opp_rating"]
    )

    dataset["opening_family"] = dataset["eco"].apply(
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
        np.nan,
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
        default="lost",
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


# Columns available after assemble_features that are used as model inputs.
# Identifiers/metadata (game_id, san, uci, fen_before, date_utc, platform,
# eval_source) and the target itself (move_quality, y) are deliberately excluded.
FEATURE_COLUMNS = [
    "move_number",
    "phase",
    "rolling_time_mean_3",
    "previous_long_think",
    "eval_volatility",
    "material_count",
    "material_imbalance",
    "queens_on_board",
    "n_legal_moves",
    "hanging_pieces",
    "checks_available",
    "my_rating",
    "rating_diff",
    "perf_type",
    "opening_family",
    "eval_before_move",
    "clock_before_move",
    "time_pressure_before_move",
    "eval_state_before_move",
    "clock_before_move_pct",
    "has_increment",
    "previous_time_spent",
]


def prepare_xy(
    dataset: pd.DataFrame,
    reference_columns: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice FEATURE_COLUMNS + y out of one temporal split.

    Categorical features are one-hot encoded. When reference_columns is
    provided (for validation/test), columns are aligned to the training set
    so unseen categories do not create incompatible feature shapes.
    """

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["y"].copy()

    categorical_columns = [
        "phase",
        "perf_type",
        "opening_family",
        "eval_state_before_move",
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=int,
    )

    if reference_columns is not None:
        X = X.reindex(
            columns=reference_columns,
            fill_value=0,
        )

    return X, y


@dataclass(frozen=True)
class ModelSplits:
    """Everything train / evaluate / ablation need, built ONE way.

    Raw splits keep identifiers (game_id, ply, phase...) for slicing,
    bootstrap grouping and SHAP metadata; X/y are model-ready, with val and
    test columns aligned on train's one-hot columns.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_model_splits(config: dict) -> ModelSplits:
    """Single source of truth for the V2 data: parquet -> current-regime
    filter -> assemble_features -> temporal split -> prepare_xy.

    Replaces the block copied in train.py, evaluate_model.py and ablation.py
    (tech debt, PROJECT_TRACKER 2026-10-10): three copies could silently
    drift onto different data. `config` is the parsed config.toml, loaded
    once by the caller (train.py still needs other keys from it).
    """
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    games = pd.read_parquet(
        processed_dir / "games.parquet"
    )
    moves = pd.read_parquet(
        processed_dir / "moves.parquet"
    )

    thresholds = config["thresholds"]

    games_current, moves_current = filter_current_regime(
        games,
        moves,
        config["current_regime"],
    )

    dataset = assemble_features(
        moves_current,
        games_current,
        thresholds["time_pressure_pct"],
        thresholds["long_think_factor"],
        thresholds["blunder"],
    )

    train, val, test = temporal_game_split(
        dataset,
        games_current,
    )

    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(
        val,
        reference_columns=X_train.columns,
    )
    X_test, y_test = prepare_xy(
        test,
        reference_columns=X_train.columns,
    )

    return ModelSplits(
        train=train,
        val=val,
        test=test,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
