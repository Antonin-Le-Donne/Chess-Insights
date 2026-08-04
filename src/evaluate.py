"""Local Stockfish evaluation pass (spec section 3).

For games where has_server_evals is False, evaluate every position with a
local Stockfish instance and fill eval_cp / eval_source="stockfish" in
moves.parquet. Must be resumable: track which game_ids are already done in
a file, skip them on restart, safe to Ctrl+C at any time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine
import pandas as pd

DONE_LIST_PATH = Path("data/interim/evaluated_game_ids.txt")
STOCKFISH_PATH = "stockfish/stockfish.exe"
LIMIT = chess.engine.Limit(depth=20, time=5.0)

N_WORKERS = 5 

def load_done_game_ids(path: Path = DONE_LIST_PATH) -> set[str]:
    """Read the resumability file: game_ids already evaluated in a past run.
    """
    if not path.exists():
        return set()
    with open(path) as f:
        return {line.strip() for line in f}

def mark_game_done(game_id: str, path: Path = DONE_LIST_PATH) -> None:
    """Append one game_id to the resumability file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{game_id}\n")

def games_to_evaluate(games: pd.DataFrame, done_ids: set[str]) -> pd.DataFrame:
    """Filter games needing a local Stockfish pass."""
    pending = games[(games["has_server_evals"] == False) & (~games["game_id"].isin(done_ids))]
    return pending.sort_values("date_utc", ascending=False)

def score_to_cp(score: chess.engine.PovScore) -> int:
    """Convert a python-chess engine Score to centipawns, White's POV."""
    return score.white().score(mate_score=10000)


def evaluate_game(engine: chess.engine.SimpleEngine, fens: list[str], limit: chess.engine.Limit) -> list[int]:
    """Run one engine instance over every position (fen_before) of one game."""
    eval_cps = []

    for fen in fens: 
        board = chess.Board(fen)
        analyse = engine.analyse(board, limit)
        score = analyse["score"]
        score_to_append = score_to_cp(score)
        eval_cps.append(score_to_append)

    return eval_cps

def evaluate_one_game(game_id: str, fens: list[str], limit: chess.engine.Limit) -> tuple[str, list[int]]:
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        eval_cps = evaluate_game(engine, fens, limit)
    return game_id, eval_cps

def main() -> None:
    moves_path = Path("data/processed/moves.parquet")
    games_path = Path("data/processed/games.parquet")

    games = pd.read_parquet(games_path)
    moves = pd.read_parquet(moves_path)

    done_ids = load_done_game_ids()
    pending_games = games_to_evaluate(games, done_ids)

    if pending_games.empty:
        print("No games pending evaluation.")
        return

    print(f"Starting parallel Stockfish evaluation for {len(pending_games)} games ({N_WORKERS} workers)...")

    tasks = []
    for game_id in pending_games["game_id"]:
        mask = moves["game_id"] == game_id
        game_moves = moves[mask].sort_values("ply")
        fens = game_moves["fen_before"].tolist()

        if not fens:
            mark_game_done(game_id)
            continue

        tasks.append((game_id, fens))

    total = len(tasks)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [
            executor.submit(evaluate_one_game, gid, fens, LIMIT)
            for gid, fens in tasks
        ]

        for count, future in enumerate(as_completed(futures), start=1):
            game_id, eval_cps = future.result()

            mask = moves["game_id"] == game_id
            game_moves = moves[mask].sort_values("ply")

            moves.loc[game_moves.index, "eval_cp"] = eval_cps
            moves.loc[game_moves.index, "eval_source"] = "stockfish"

            moves.to_parquet(moves_path, index=False)
            mark_game_done(game_id)
            print(f"[{count}/{total}] Evaluated game {game_id} ({len(eval_cps)} plies)")

    print("Stockfish parallel evaluation pass completed and saved to moves.parquet.")

if __name__ == "__main__":
    main()