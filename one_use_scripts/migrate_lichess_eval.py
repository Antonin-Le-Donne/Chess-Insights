"""One-off migration: shift Lichess [%eval] values by one ply so that eval_cp
means 'eval of fen_before' for every eval_source (Stockfish already does).
Delete after use.

Why: parse.py read [%eval] from node.comment, which is the eval AFTER the
move. features.add_win_pct_delta assumes BEFORE -> every Lichess-eval move
got the delta of the NEXT ply (the opponent's), negated -> 0 blunders.
"""

import pandas as pd

MOVES_PATH = "data/processed/moves.parquet"


def already_migrated(moves: pd.DataFrame) -> bool:
    """Idempotence guard: running this twice would shift by TWO plies."""
    # TODO: pick a signal that tells you the shift was already applied
    #       (marker file? marker column? a property of the data itself?)
    ...


def count_mixed_source_games(moves: pd.DataFrame) -> int:
    """Games whose plies mix 'lichess' and 'stockfish' evals."""
    # TODO: groupby game_id, count distinct eval_source
    # TODO: decide BEFORE shifting what happens to those games
    ...


def shift_lichess_evals(moves: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where, inside each Lichess-eval game sorted by ply,
    new eval_cp(N) = old eval_cp(N-1). Stockfish rows untouched.
    """
    # TODO: select the rows to shift (which games? which rows?)
    # TODO: sort by game_id, ply -- shift is meaningless on unsorted rows
    # TODO: shift within each game, never across game boundaries
    # TODO: ply 1 -> your design decision (same one as in parse.py)
    ...


def validate(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Fail loudly before anything is overwritten."""
    # TODO: assert same number of rows
    # TODO: assert Stockfish rows' eval_cp strictly unchanged
    # TODO: assert Lichess rows: after(N) == before(N-1)
    ...


if __name__ == "__main__":
    # TODO: load moves.parquet
    # TODO: guard -> stop if already_migrated
    # TODO: full-file backup under an explicit name (not eval_backup.parquet)
    # TODO: report count_mixed_source_games
    # TODO: shift -> validate -> write to a temp file -> only then replace MOVES_PATH
    # TODO: afterwards: re-run features.py, then check_drift.py (Lichess blunder rate ~5%)
    ...
