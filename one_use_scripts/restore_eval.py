"""Restore eval_cp/eval_source into the freshly re-parsed moves.parquet.
Run AFTER parse.py, BEFORE evaluate.py. Delete after use."""

import pandas as pd

moves = pd.read_parquet("data/processed/moves.parquet")
backup = pd.read_parquet("data/interim/eval_backup_oct2026.parquet")

backup = backup[backup["eval_source"] == "stockfish"]

merged = moves.merge(backup, on=["game_id", "ply"], how="left", suffixes=("", "_backup"))

# trust the freshly parsed value first (Lichess evals re-extracted from the PGN),
# fall back to the backed-up value only where the fresh parse has nothing
# (that's the Stockfish-computed ones we're restoring)
merged["eval_cp"] = merged["eval_cp"].combine_first(merged["eval_cp_backup"])
merged["eval_source"] = merged["eval_source"].combine_first(merged["eval_source_backup"])
merged = merged.drop(columns=["eval_cp_backup", "eval_source_backup"])

merged.to_parquet("data/processed/moves.parquet", index=False)

print(f"eval_cp present for {merged['eval_cp'].notna().sum()} / {len(merged)} rows after restore.")