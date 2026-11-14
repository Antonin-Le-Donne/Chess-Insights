"""Throwaway sanity check for eval_cp coverage after evaluate.py. Delete after use."""

import pandas as pd

games = pd.read_parquet("data/processed/games.parquet")
moves = pd.read_parquet("data/processed/moves.parquet")

missing = moves[moves["eval_cp"].isna()]
missing_game_ids = set(missing["game_id"].unique())

print("nombre de parties :", len(games))
print("game_id dupliqués :", games["game_id"].duplicated().sum())
print("eval_cp manquants :", moves["eval_cp"].isna().sum())
print("parties concernées :", len(missing_game_ids))

if missing_game_ids:
    affected = games[games["game_id"].isin(missing_game_ids)]
    print("\nhas_server_evals des parties concernées :")
    print(affected["has_server_evals"].value_counts())

    print("\nrepartition des lignes manquantes par partie (top 10) :")
    print(missing["game_id"].value_counts().head(10))

    print("\nrepartition par plateforme :")
    print(affected["platform"].value_counts())