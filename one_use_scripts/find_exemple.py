import pandas as pd
moves = pd.read_parquet("data/processed/moves.parquet")
suspects = moves[(moves["eval_source"] == "lichess") & moves["is_my_move"] & (moves["win_pct_delta"] >= 20)]
print(suspects[["game_id", "ply", "san", "eval_cp", "win_pct_delta"]].head(5))
game = moves[moves["game_id"] == "lichess_Kl7iECHe"]
print(game[game["ply"].between(35, 41)][["ply", "san", "eval_cp", "win_pct", "win_pct_delta"]])
