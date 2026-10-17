"""PGN parsing (spec section 2).

Reads the raw .pgn and json files produced by fetch.py and turns them into two
DataFrames: one row per game (games.parquet), one row per ply (moves.parquet).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
import tempfile

import chess.pgn
import pandas as pd

from src.fetch import load_config


def iter_games(pgn_path: Path):
    """Yield one chess.pgn.Game per game found in pgn_path."""

    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)

            if game is None:
                break

            yield game


def iter_chesscom_games(json_path: Path):
    """Yield (game, external_id, time_class) tuples from a JSON archive."""

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    for game_obj in data["games"]:
        pgn_text = game_obj["pgn"]
        game = chess.pgn.read_game(io.StringIO(pgn_text))

        if game is None:
            continue

        external_id = game_obj["url"].rstrip("/").split("/")[-1]
        time_class = game_obj["time_class"]

        yield game, external_id, time_class


def game_to_record(
    game: chess.pgn.Game,
    username: str,
    platform: str,
    external_game_id: str | None = None,
    external_perf_type: str | None = None,
) -> dict:
    """Build one games.parquet row from a parsed game's headers."""

    my_color = (
        "white"
        if game.headers["White"].lower() == username.lower()
        else "black"
    )

    time_control = game.headers["TimeControl"]

    if "+" in time_control:
        base_str, inc_str = time_control.split("+")
    else:
        base_str, inc_str = time_control, "0"

    base_seconds = int(base_str)
    increment_seconds = int(inc_str)

    estimated = base_seconds + 40 * increment_seconds

    if external_perf_type is not None:
        perf_type = external_perf_type
    elif estimated < 180:
        perf_type = "bullet"
    elif estimated < 480:
        perf_type = "blitz"
    elif estimated < 1500:
        perf_type = "rapid"
    else:
        perf_type = "classical"

    eco = game.headers.get("ECO")
    opening_name = game.headers.get("Opening")
    termination = game.headers["Termination"]

    raw_id = (
        external_game_id
        if external_game_id is not None
        else game.headers["GameId"]
    )
    game_id = f"{platform}_{raw_id}"

    n_plies = sum(1 for _ in game.mainline())
    date_utc = f"{game.headers['UTCDate']} {game.headers['UTCTime']}"

    if my_color == "white":
        my_rating = int(game.headers["WhiteElo"])
        opp_rating = int(game.headers["BlackElo"])
    else:
        my_rating = int(game.headers["BlackElo"])
        opp_rating = int(game.headers["WhiteElo"])

    raw_result = game.headers["Result"]

    if raw_result == "1/2-1/2":
        result_for_me = 0.5
    elif (
        (my_color == "white" and raw_result == "1-0")
        or (my_color == "black" and raw_result == "0-1")
    ):
        result_for_me = 1
    else:
        result_for_me = 0

    has_server_evals = (
        "[%eval" in game.comment
        or any("[%eval" in node.comment for node in game.mainline())
    )

    return {
        "platform": platform,
        "game_id": game_id,
        "date_utc": date_utc,
        "time_control": time_control,
        "base_seconds": base_seconds,
        "increment_seconds": increment_seconds,
        "perf_type": perf_type,
        "my_color": my_color,
        "my_rating": my_rating,
        "opp_rating": opp_rating,
        "eco": eco,
        "opening_name": opening_name,
        "result_for_me": result_for_me,
        "termination": termination,
        "n_plies": n_plies,
        "has_server_evals": has_server_evals,
    }


def game_to_move_records(
    game: chess.pgn.Game,
    username: str,
    platform: str,
    external_game_id: str | None = None,
) -> list[dict]:
    """Walk the mainline, emit one dict per ply for moves.parquet.

    eval_cp describes fen_before, from White's point of view.
    clock_after describes the clock after the current move.
    """

    move_records = []

    raw_id = (
        external_game_id
        if external_game_id is not None
        else game.headers["GameId"]
    )
    game_id = f"{platform}_{raw_id}"

    my_color = (
        chess.WHITE
        if game.headers["White"].lower() == username.lower()
        else chess.BLACK
    )

    for ply, node in enumerate(game.mainline(), start=1):
        parent = node.parent
        board_before = parent.board()

        fen_before = board_before.fen()
        san = node.san()
        uci = node.move.uci()
        move_number = board_before.fullmove_number
        is_my_move = board_before.turn == my_color

        clock_match = re.search(
            r"\[%clk ([\d:.]+)\]",
            node.comment,
        )

        if clock_match is None:
            raise ValueError(
                f"Missing clock at ply {ply} in game {game_id}"
            )

        h, m, s = clock_match.group(1).split(":")
        clock_after = int(h) * 3600 + int(m) * 60 + float(s)

        # The parent's evaluation describes the position BEFORE this move.
        # For ply 1, parent is the root: without a root evaluation, keep None
        # so evaluate.py can evaluate the initial position later.
        eval_match = re.search(
            r"\[%eval\s+(#[+-]?\d+|[+-]?\d+(?:\.\d+)?)"
            r"(?:\s*,\s*\d+)?\s*\]",
            parent.comment,
        )

        eval_cp = None
        eval_source = None

        if eval_match:
            raw_eval = eval_match.group(1)
            eval_source = platform

            if raw_eval.startswith("#"):
                moves_to_mate = int(raw_eval[1:])

                if moves_to_mate > 0:
                    eval_cp = 10000
                elif moves_to_mate < 0:
                    eval_cp = -10000
                else:
                    # #0: the side to move is already checkmated.
                    eval_cp = (
                        -10000 if board_before.turn == chess.WHITE else 10000
                    )
            else:
                eval_cp = round(float(raw_eval) * 100)

        move_records.append({
            "platform": platform,
            "game_id": game_id,
            "ply": ply,
            "move_number": move_number,
            "is_my_move": is_my_move,
            "san": san,
            "uci": uci,
            "clock_after": clock_after,
            "eval_cp": eval_cp,
            "eval_source": eval_source,
            "fen_before": fen_before,
        })

    return move_records


def compute_time_spent(
    moves: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.Series:
    """Derive per-move thinking time, preserving the original row index."""

    if moves.empty:
        return pd.Series(index=moves.index, dtype=float)

    # Keep row positions explicitly because merge creates a new index.
    moves_enriched = moves.assign(_row_order=range(len(moves))).merge(
        games[["game_id", "base_seconds", "increment_seconds"]],
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    assert not moves_enriched["base_seconds"].isna().any(), (
        "merge produced NaN base_seconds — game_id mismatch"
    )

    moves_enriched = moves_enriched.sort_values(["game_id", "ply"])

    clock_before = moves_enriched.groupby(
        ["game_id", "is_my_move"]
    )["clock_after"].shift(1)

    clock_before = clock_before.fillna(moves_enriched["base_seconds"])

    moves_enriched["time_spent"] = (
        clock_before
        - moves_enriched["clock_after"]
        + moves_enriched["increment_seconds"]
    ).clip(lower=0)

    time_spent = moves_enriched.sort_values("_row_order")["time_spent"]

    return pd.Series(
        time_spent.to_numpy(),
        index=moves.index,
        name="time_spent",
    )


def parse_new_games(
    config,
    known_game_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """Parse only raw games that are not already present in processed data.

    Walks all Lichess PGN files and chess.com JSON archives. Existing IDs are
    checked before game_to_move_records(), which is the expensive step. IDs
    parsed during this call are also remembered so duplicate raw entries are
    not emitted twice.
    """
    game_records: list[dict] = []
    move_records: list[dict] = []
    seen_game_ids = set(known_game_ids)

    lichess_username = config.username
    lichess_files = Path("data/raw/lichess").glob("*.pgn")

    for pgn_file in lichess_files:
        for game in iter_games(pgn_file):
            game_id = f"lichess_{game.headers['GameId']}"
            if game_id in seen_game_ids:
                continue

            game_record = game_to_record(
                game,
                lichess_username,
                "lichess",
            )
            moves_for_game = game_to_move_records(
                game,
                lichess_username,
                "lichess",
            )

            game_records.append(game_record)
            move_records.extend(moves_for_game)
            seen_game_ids.add(game_id)

    chesscom_files = Path("data/raw/chesscom").glob("*.json")

    for json_path in chesscom_files:
        for game, external_id, time_class in iter_chesscom_games(json_path):
            game_id = f"chesscom_{external_id}"
            if game_id in seen_game_ids:
                continue

            try:
                game_record = game_to_record(
                    game,
                    config.chesscom_username,
                    "chesscom",
                    external_game_id=external_id,
                    external_perf_type=time_class,
                )

                moves_for_game = game_to_move_records(
                    game,
                    config.chesscom_username,
                    "chesscom",
                    external_game_id=external_id,
                )

            except (ValueError, KeyError) as exc:
                print(f"Skipping chess.com game {external_id}: {exc}")
                continue

            game_records.append(game_record)
            move_records.extend(moves_for_game)
            seen_game_ids.add(game_id)

    return game_records, move_records


def load_existing(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load the existing processed files, or return None on a first run.

    moves.parquet already carries Stockfish evals (evaluate.py) and derived
    columns (features.py): an incremental parse must never rewrite existing
    rows. Every move must reference a game. A game may have no move rows only
    when ``n_plies == 0``; for example, a draw by agreement can occur before
    the first move. A game with ``n_plies > 0`` but no move rows indicates an
    inconsistent or partially written processed-data state.
    """
    games_path = processed_dir / "games.parquet"
    moves_path = processed_dir / "moves.parquet"

    games_exists = games_path.is_file()
    moves_exists = moves_path.is_file()

    if not games_exists and not moves_exists:
        return None

    if games_exists != moves_exists:
        raise FileNotFoundError(
            "Incomplete processed-data state: both games.parquet and "
            f"moves.parquet must exist in {processed_dir}. "
            f"games.parquet exists: {games_exists}; "
            f"moves.parquet exists: {moves_exists}."
        )

    games = pd.read_parquet(games_path)
    moves = pd.read_parquet(moves_path)

    if games["game_id"].duplicated().any():
        raise ValueError(
            "games.parquet contains duplicate game_id values."
        )

    if moves.duplicated(["game_id", "ply"]).any():
        raise ValueError(
            "moves.parquet contains duplicate (game_id, ply) rows."
        )

    if games["game_id"].isna().any():
        raise ValueError("games.parquet contains null game_id values.")
    if moves["game_id"].isna().any():
        raise ValueError("moves.parquet contains null game_id values.")

    game_ids = set(games["game_id"])
    move_game_ids = set(moves["game_id"])
    games_without_moves = game_ids - move_game_ids
    moves_without_games = move_game_ids - game_ids

    # Games with zero plies are valid without move rows (for example, a draw
    # by agreement before the first move). All other games without move rows
    # indicate incomplete or inconsistent processed data.
    games_missing_moves_with_plies: set[str] = set()
    if games_without_moves:
        if "n_plies" not in games.columns:
            raise ValueError(
                "games.parquet is missing the n_plies column, so games without "
                "move rows cannot be validated. Expected n_plies == 0 for "
                "games that legitimately have no moves."
            )

        missing_move_rows = games.loc[
            games["game_id"].isin(games_without_moves),
            ["game_id", "n_plies"],
        ]
        n_plies = pd.to_numeric(
            missing_move_rows["n_plies"],
            errors="coerce",
        )
        # Allow only an explicit zero. Positive counts indicate missing move
        # rows; null, non-numeric, or otherwise invalid counts cannot prove
        # that a no-move game is legitimate either.
        invalid_n_plies = n_plies.ne(0) | n_plies.isna()
        games_missing_moves_with_plies = set(
            missing_move_rows.loc[invalid_n_plies, "game_id"]
        )

    if games_missing_moves_with_plies or moves_without_games:
        details = []
        if games_missing_moves_with_plies:
            sample_rows = games.loc[
                games["game_id"].isin(games_missing_moves_with_plies),
                ["game_id", "n_plies"],
            ].head(10)
            sample = sample_rows.to_dict(orient="records")
            details.append(
                "Present in games.parquet but absent from moves.parquet despite "
                f"a non-zero or invalid n_plies value (up to 10): {sample}"
            )
        if moves_without_games:
            sample = sorted(moves_without_games, key=str)[:10]
            details.append(
                "Present in moves.parquet but absent from games.parquet "
                f"(up to 10): {sample}"
            )

        raise ValueError(
            "Inconsistent processed-data state: every move must reference a "
            "game, and every game with n_plies > 0 must have move rows. A game "
            "with n_plies == 0 may legitimately be absent from moves.parquet "
            "(for example, a draw by agreement before the first move). "
            + " ".join(details)
            + " Do not continue parsing until the files are reconciled."
        )

    return games, moves


def append_new(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    key: list[str],
) -> pd.DataFrame:
    """Append new rows without changing any existing values.

    Raises if the concatenated result contains duplicate keys or if the
    existing rows differ after concatenation.
    """
    if not key:
        raise ValueError("The append key cannot be empty.")

    for column in key:
        if column not in existing.columns:
            raise KeyError(f"Existing DataFrame is missing key {column!r}.")
        if column not in new.columns:
            raise KeyError(f"New DataFrame is missing key {column!r}.")

    result = pd.concat(
        [existing, new],
        ignore_index=True,
    )

    if result.duplicated(key).any():
        duplicates = result.loc[
            result.duplicated(key, keep=False),
            key,
        ].head(10)

        raise ValueError(
            "Duplicate keys detected after appending new rows. "
            f"First duplicates:\n{duplicates.to_string(index=False)}"
        )

    # Confirm that every pre-existing column and value remains unchanged.
    # New derived columns may be absent from `new`; concat fills those
    # columns with NaN for the new rows without modifying old rows.
    original_rows = result.iloc[:len(existing)][
        existing.columns
    ].reset_index(drop=True)

    expected_rows = existing.reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            original_rows,
            expected_rows,
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise AssertionError(
            "Appending new rows changed existing data. "
            "Do not write the processed files."
        ) from exc

    if len(result) != len(existing) + len(new):
        raise AssertionError(
            "Unexpected row count after appending new data."
        )

    return result


def write_atomically(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Write to a temporary file next to path, then replace path.

    A failed write leaves the previous destination file untouched.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}_",
            suffix=path.suffix,
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        df.to_parquet(
            temp_path,
            index=False,
        )

        temp_path.replace(path)
        temp_path = None

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

# TODO(incremental): optional `--full` flag (argparse) for a deliberate full
# rebuild -- print loudly that it wipes Stockfish evals.
# TODO(incremental): afterwards run evaluate.py (picks up the new games
# without server evals, it already skips done ids) then features.py.
def main() -> None:
    """Incrementally parse raw games into games.parquet and moves.parquet.

    Existing processed rows are retained. Only unseen game IDs are parsed and
    appended, preserving existing Stockfish evaluations and derived columns.
    """
    config = load_config()
    output_dir = Path("data/processed")

    existing = load_existing(output_dir)

    if existing is None:
        known_game_ids: set[str] = set()
    else:
        existing_games_for_ids, _ = existing
        known_game_ids = set(existing_games_for_ids["game_id"].astype(str))

    new_game_records, new_move_records = parse_new_games(
        config,
        known_game_ids,
    )

    if not new_game_records:
        print(
            "Aucune nouvelle partie à ajouter. "
            "Les fichiers traités restent inchangés."
        )
        return

    new_games = pd.DataFrame(new_game_records)
    new_moves = pd.DataFrame(
        new_move_records,
        columns=[
            "platform",
            "game_id",
            "ply",
            "move_number",
            "is_my_move",
            "san",
            "uci",
            "clock_after",
            "eval_cp",
            "eval_source",
            "fen_before",
        ],
    )

    new_games["date_utc"] = pd.to_datetime(
        new_games["date_utc"],
        format="%Y.%m.%d %H:%M:%S",
    )
    new_moves["time_spent"] = compute_time_spent(
        new_moves,
        new_games,
    )

    if existing is None:
        games = new_games
        moves = new_moves
    else:
        existing_games, existing_moves = existing
        games = append_new(
            existing_games,
            new_games,
            key=["game_id"],
        )
        moves = append_new(
            existing_moves,
            new_moves,
            key=["game_id", "ply"],
        )

    # Write moves first. If the process stops before games.parquet is
    # replaced, load_existing() will detect the mismatched game_id sets and
    # stop loudly rather than silently losing the new moves.
    write_atomically(
        moves,
        output_dir / "moves.parquet",
    )
    write_atomically(
        games,
        output_dir / "games.parquet",
    )

    print(
        f"Parsing incrémental terminé : {len(new_games)} nouvelles parties "
        f"et {len(new_moves)} nouveaux coups ajoutés."
    )
    print(
        f"Total traité : {len(games)} parties et {len(moves)} coups."
    )

if __name__ == "__main__":
    main()