"""Backup eval_cp/eval_source before re-running parse.py on the combined
(blitz+rapid+bullet) PGN file. Run this BEFORE touching parse.py. Delete after use."""

import pandas as pd
from pathlib import Path

moves = pd.read_parquet("data/processed/moves.parquet")
backup = moves[["game_id", "ply", "eval_cp", "eval_source"]].copy()

Path("data/interim").mkdir(parents=True, exist_ok=True)
backup.to_parquet("data/interim/eval_backup_oct2026.parquet", index=False)

print(f"Backed up {len(backup)} rows, {backup['eval_cp'].notna().sum()} with a real eval_cp value.")