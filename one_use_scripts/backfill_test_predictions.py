"""One-off: write test_predictions.parquet for an EXISTING evaluation run.

evaluate_model.py rebuilds the test set from the current parquets. After an
import, the 70/15/15 temporal split shifts, so re-running it without
retraining would score the saved model on a different test set. This script
rebuilds the exact test set of the run instead: same steps as
dataset.load_model_splits, on the games played before the run's data cutoff.

Guard: the rebuilt test set must reproduce the run's LightGBM PR-AUC
(metrics_summary.csv) to 1e-9, and the saved feature columns, or nothing is
written.

Usage (from the repo root):
    python one_use_scripts/backfill_test_predictions.py
    python one_use_scripts/backfill_test_predictions.py --run 2026-11-14_17-03-16 --cutoff 2026-10-17
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.dataset import assemble_features, prepare_xy, temporal_game_split  # noqa: E402
from src.model.evaluate_model import load_models, save_test_predictions  # noqa: E402
from src.scope import filter_current_regime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="2026-11-14_17-03-16")
    parser.add_argument(
        "--cutoff",
        default="2026-10-17",
        help="first day NOT in the run's data (the 2026-10-17 import stopped at 2026-10-16)",
    )
    args = parser.parse_args()

    run_dir = Path("model_evaluation") / args.run
    expected = pd.read_csv(run_dir / "metrics_summary.csv").set_index("model")
    expected_pr_auc = float(expected.loc["lightgbm", "pr_auc"])

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    thresholds = config["thresholds"]

    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    games = games[pd.to_datetime(games["date_utc"]) < pd.Timestamp(args.cutoff)]
    moves = moves[moves["game_id"].isin(games["game_id"])]

    # Same steps as dataset.load_model_splits, on the run's data.
    games_current, moves_current = filter_current_regime(
        games, moves, config["current_regime"]
    )
    dataset = assemble_features(
        moves_current,
        games_current,
        thresholds["time_pressure_pct"],
        thresholds["long_think_factor"],
        thresholds["blunder"],
    )
    train, _, test = temporal_game_split(dataset, games_current)
    X_train, _ = prepare_xy(train)
    X_test, y_test = prepare_xy(test, reference_columns=X_train.columns)

    models = load_models()
    if list(models["feature_columns"]) != list(X_train.columns):
        raise SystemExit("Feature columns differ from the saved model: nothing written.")

    pr_auc = average_precision_score(
        y_test, models["lightgbm"].predict_proba(X_test)[:, 1]
    )
    print(
        f"Rebuilt test set: {test['game_id'].nunique():,} games, {len(X_test):,} moves; "
        f"PR-AUC {pr_auc:.6f} vs run {expected_pr_auc:.6f}"
    )
    if abs(pr_auc - expected_pr_auc) > 1e-9:
        raise SystemExit("PR-AUC does not match the run: wrong test set, nothing written.")

    output = run_dir / "test_predictions.parquet"
    predictions = save_test_predictions(
        models["lightgbm"], X_test, y_test, test, output
    )
    print(f"Saved {len(predictions):,} rows to {output}")


if __name__ == "__main__":
    main()
