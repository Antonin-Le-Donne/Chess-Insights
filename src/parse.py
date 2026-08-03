"""PGN parsing (spec section 2).

Reads the raw .pgn files produced by fetch.py and turns them into two
DataFrames: one row per game (games.parquet), one row per ply (moves.parquet).
"""

from __future__ import annotations

import re
from pathlib import Path

import chess.pgn
import pandas as pd

from src.fetch import load_config


def iter_games(pgn_path: Path):
    """Yield one chess.pgn.Game per game found in pgn_path."""

    with open(pgn_path,"r") as f:
      while True:
          game = chess.pgn.read_game(f)
          if game is None:
            break
          yield game

def game_to_record(game: chess.pgn.Game, username: str, platform: str) -> dict:
    """Build one games.parquet row from a parsed game's headers."""

    my_color = "white" if game.headers["White"].lower() == username.lower() else "black"
    time_control = game.headers["TimeControl"]
    base_str, inc_str = time_control.split("+")
    base_seconds, increment_seconds = int(base_str), int(inc_str)
    eco = game.headers["ECO"]
    opening_name = game.headers["Opening"]
    termination = game.headers["Termination"]
    perf_type = game.headers["Event"].split()[1]
    game_id = f"{platform}_{game.headers['GameId']}"
    n_plies = game.end().board().ply()
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
    elif my_color == "white" and raw_result == "1-0" or my_color == "black" and raw_result == "0-1" :
        result_for_me = 1
    else:
        result_for_me = 0

    has_server_evals = False
    for node in game.mainline():
        if "[%eval" in node.comment:
            has_server_evals = True
            break
    
    return {
        "platform":platform,
        "game_id":game_id,
        "date_utc":date_utc,
        "time_control":time_control,
        "base_seconds":base_seconds,
        "increment_seconds":increment_seconds,
        "perf_type":perf_type,
        "my_color":my_color,
        "my_rating":my_rating,
        "opp_rating":opp_rating,
        "eco":eco,
        "opening_name":opening_name,
        "result_for_me":result_for_me,
        "termination":termination,
        "n_plies":n_plies,
        "has_server_evals":has_server_evals,
    }

def game_to_move_records(game: chess.pgn.Game, username: str, platform: str) -> list[dict]:
    """Walk the mainline, emit one dict per ply for moves.parquet."""

    move_recorded = []
    game_id = f"{platform}_{game.headers['GameId']}"
    my_color = "white" if game.headers["White"].lower() == username.lower() else "black"
    for ply, node in enumerate(game.mainline(), start=1):
        fen_before = node.parent.board().fen()
        san = node.san()
        uci = node.move.uci()
        move_number = (ply + 1) // 2
        if my_color == "white" and ply % 2 == 1 or my_color == "black" and ply % 2 == 0:
            is_my_move = True
        else:
            is_my_move = False

        clock_match = re.search(r"\[%clk ([\d:]+)\]", node.comment)
        assert clock_match is not None
        clock_parts = clock_match.group(1).split(":")
        h,m,s = clock_parts
        clock_after = int(h) * 3600 + int(m) * 60 + int(s)

        eval_match = re.search(r"\[%eval ([\d.#-]+)\]", node.comment)
        if eval_match:
            raw_eval = eval_match.group(1)
            eval_source = "lichess"

            if "#" in raw_eval:
                moves_to_mate = int(raw_eval.replace("#", ""))
                eval_cp = 10000 if moves_to_mate > 0 else -10000
            else:
                eval_cp = int(float(raw_eval) * 100)
        else:
            eval_cp = None
            eval_source = None

        move_recorded.append({
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

    return move_recorded

def compute_time_spent(moves: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """Derive per-move thinking time from clock_after values."""
    
    moves_enriched = moves.merge(
        games[["game_id", "base_seconds", "increment_seconds"]],
        on="game_id",
        how="left"
    )
    assert not moves_enriched["base_seconds"].isna().any(), "merge produced NaN base_seconds — game_id mismatch"
    moves_enriched = moves_enriched.sort_values(["game_id", "ply"])

    clock_before = moves_enriched.groupby(["game_id", "is_my_move"])["clock_after"].shift(1)
    clock_before = clock_before.fillna(moves_enriched["base_seconds"])
    time_spent = clock_before - moves_enriched["clock_after"] + moves_enriched["increment_seconds"]

    return time_spent.clip(lower = 0)


def main() -> None:
    """Parse every raw .pgn file into games.parquet + moves.parquet."""
    config = load_config()
    username = config.username
    platform = "lichess"

    pgn_files = Path("data/raw/lichess").glob("*.pgn")

    game_records = []
    move_records = []

    for pgn_file in pgn_files:
        for game in iter_games(pgn_file):
            game_records.append(game_to_record(game, username, platform))
            move_records.extend(game_to_move_records(game, username, platform))

    if not game_records:
        print("Aucun fichier PGN trouvé.")
        return

    games = pd.DataFrame(game_records)
    moves = pd.DataFrame(move_records)

    games["date_utc"] = pd.to_datetime(games["date_utc"], format="%Y.%m.%d %H:%M:%S")

    moves["time_spent"] = compute_time_spent(moves, games)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    games.to_parquet(output_dir / "games.parquet")
    moves.to_parquet(output_dir / "moves.parquet")

    print(f"Parsing finished : {len(games)} games and {len(moves)} move saved.")

if __name__ == "__main__":
    main()

