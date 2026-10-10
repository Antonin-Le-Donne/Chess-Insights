"""PGN parsing (spec section 2).

Reads the raw .pgn and json files produced by fetch.py and turns them into two
DataFrames: one row per game (games.parquet), one row per ply (moves.parquet).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

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


def main() -> None:
    """Parse raw files into games.parquet and moves.parquet.

    This writes fresh outputs. Re-parsing with preservation of existing
    Stockfish evaluations is handled by a separate script.
    """

    config = load_config()
    username = config.username
    platform = "lichess"

    pgn_files = Path("data/raw/lichess").glob("*.pgn")

    game_records = []
    move_records = []

    for pgn_file in pgn_files:
        for game in iter_games(pgn_file):
            game_records.append(
                game_to_record(
                    game,
                    username,
                    platform,
                )
            )

            move_records.extend(
                game_to_move_records(
                    game,
                    username,
                    platform,
                )
            )

    chesscom_files = Path("data/raw/chesscom").glob("*.json")

    for json_path in chesscom_files:
        for game, external_id, time_class in iter_chesscom_games(json_path):
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

            except (ValueError, KeyError) as e:
                print(f"Skipping chess.com game {external_id}: {e}")
                continue

            game_records.append(game_record)
            move_records.extend(moves_for_game)

    if not game_records:
        print("Aucune partie trouvée.")
        return

    games = pd.DataFrame(game_records)
    moves = pd.DataFrame(
        move_records,
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

    games["date_utc"] = pd.to_datetime(
        games["date_utc"],
        format="%Y.%m.%d %H:%M:%S",
    )

    moves["time_spent"] = compute_time_spent(moves, games)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    games.to_parquet(output_dir / "games.parquet")
    moves.to_parquet(output_dir / "moves.parquet")

    print(
        f"Parsing finished: {len(games)} games "
        f"and {len(moves)} moves saved."
    )


if __name__ == "__main__":
    main()