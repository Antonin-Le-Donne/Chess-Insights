"""Throwaway check: does the blunder rate / time-control mix drift across splits? Delete after use."""

import tomllib
import pandas as pd
from src.model.dataset import assemble_features, temporal_game_split
from src.scope import filter_current_regime

if __name__ == "__main__":
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)

        floors = config["current_regime"]
        time_pressure_pct = config["thresholds"]["time_pressure_pct"]

        games = pd.read_parquet(
            "data/processed/games.parquet"
        )

        moves = pd.read_parquet(
            "data/processed/moves.parquet"
        )

        games_current, moves_current = filter_current_regime(
            games,
            moves,
            floors,
        )

        dataset = assemble_features(
            moves_current,
            games_current,
            time_pressure_pct,
        )

        train, val, test = temporal_game_split(
            dataset,
            games_current,
        )

        print(train["y"].mean(), val["y"].mean(), test["y"].mean())
        print(train["perf_type"].value_counts(normalize=True))
        print(test["perf_type"].value_counts(normalize=True))
        print(train.groupby("perf_type")["y"].mean())
        print(test.groupby("perf_type")["y"].mean())
        print(train["eval_source"].value_counts(normalize=True))
        print(test["eval_source"].value_counts(normalize=True))
        print(train["platform"].value_counts(normalize=True))
        print(test["platform"].value_counts(normalize=True))
        print(train["my_rating"].mean(), test["my_rating"].mean())
        print(train.groupby("eval_source")["y"].agg(["mean", "size"]))
        print(test.groupby("eval_source")["y"].agg(["mean", "size"]))
        print(dataset.groupby("eval_source")["eval_cp"].describe())
        print(dataset.groupby("eval_source")["win_pct_delta"].describe())



