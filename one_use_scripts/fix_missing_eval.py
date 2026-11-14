"""Fill the remaining eval_cp gaps (has_server_evals=True but partial coverage).
Same mechanism as before: Lichess didn't provide an eval on every ply. Delete after use."""

import chess
import chess.engine
import pandas as pd

from src.evaluate import score_to_cp

moves = pd.read_parquet("data/processed/moves.parquet")
missing = moves[moves["eval_cp"].isna()]

print(f"Filling {len(missing)} missing eval_cp values...")

with chess.engine.SimpleEngine.popen_uci("stockfish/stockfish.exe") as engine:
    for idx, row in missing.iterrows():
        board = chess.Board(row["fen_before"])
        result = engine.analyse(board, chess.engine.Limit(depth=20, time=5.0))
        eval_cp = score_to_cp(result["score"])
        moves.loc[idx, "eval_cp"] = eval_cp
        moves.loc[idx, "eval_source"] = "stockfish"
        print(f"  {row['game_id']} ply {row['ply']}: {eval_cp}")

moves.to_parquet("data/processed/moves.parquet", index=False)
print("Done. Remaining NaN:", moves["eval_cp"].isna().sum())