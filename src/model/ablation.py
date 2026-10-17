"""V2 feature ablation: can a group of weak/redundant features be dropped
without a notable PR-AUC loss?

Non-inferiority test, pre-registered 2026-10-10 (see PROJECT_TRACKER.md
decisions log) BEFORE any ablation run:
- A = reduced model (group dropped), B = full model, same splits and params.
- epsilon = 0.002 PR-AUC (~1.5% of 0.134): the largest loss accepted.
- Seeds 42-46: for EACH seed, train A and B, run the paired game-level
  bootstrap on PR-AUC(A) - PR-AUC(B). Drop the group only if the CI lower
  bound > -epsilon for ALL five seeds (stability across training noise,
  not a single interval covering it).
- The group is tested as a whole. If it fails, per-feature runs are
  exploratory only -- never reported as an independent validation.
"""

from __future__ import annotations

import json
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from src.model.dataset import load_model_splits
from src.model.evaluate_model import paired_bootstrap_pr_auc_diff
from src.model.train import train_lightgbm

EPSILON = 0.002
SEEDS = [42, 43, 44, 45, 46]

# Pre-model feature names; one-hot encoded ones expand into several columns.
ABLATION_GROUP = [
    "time_pressure_before_move",
    "has_increment",
    "perf_type",
    "previous_long_think",
    "eval_state_before_move",
]


def columns_to_drop(
    encoded_columns: pd.Index,
    group: list[str],
) -> list[str]:
    """Map pre-model feature names to the encoded columns they produced.

    Plain features require an exact match. Prefix matching is restricted
    to the known one-hot encoded features in this ablation group.
    """
    categorical_features = {
        "perf_type",
        "eval_state_before_move",
    }

    dropped = []

    for name in group:
        if name in categorical_features:
            matches = [
                column
                for column in encoded_columns
                if column == name or column.startswith(f"{name}_")
            ]
        else:
            matches = [
                column
                for column in encoded_columns
                if column == name
            ]

        if not matches:
            raise ValueError(
                f"No encoded columns found for ablation feature {name!r}"
            )

        dropped.extend(matches)

    return list(dict.fromkeys(dropped))


def run_one_seed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    game_ids: pd.Series,
    dropped: list[str],
    seed: int,
) -> dict:
    """Train full (B) and reduced (A) LightGBM with the same seed, compare
    them with the paired bootstrap, return one result row.
    """
    full_model = train_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=seed,
    )

    reduced_model = train_lightgbm(
        X_train.drop(columns=dropped),
        y_train,
        X_val.drop(columns=dropped),
        y_val,
        random_state=seed,
    )

    proba_a = reduced_model.predict_proba(
        X_test.drop(columns=dropped)
    )[:, 1]

    proba_b = full_model.predict_proba(X_test)[:, 1]

    delta, lower, upper, proportion_a_better = (
        paired_bootstrap_pr_auc_diff(
            proba_a=proba_a,
            proba_b=proba_b,
            y_test=y_test,
            game_ids=game_ids,
        )
    )

    return {
        "seed": seed,
        "dropped_columns": json.dumps(dropped),
        "pr_auc_reduced": float(
            average_precision_score(y_test, proba_a)
        ),
        "pr_auc_full": float(
            average_precision_score(y_test, proba_b)
        ),
        "pr_auc_delta": delta,
        "ci_lower": lower,
        "ci_upper": upper,
        "proportion_a_better": proportion_a_better,
        "epsilon": EPSILON,
        "passes": lower > -EPSILON,
        "best_iteration_reduced": reduced_model.best_iteration_,
        "best_iteration_full": full_model.best_iteration_,
    }


if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    splits = load_model_splits(config)
    game_ids = splits.test["game_id"]

    dropped = columns_to_drop(
        splits.X_train.columns,
        ABLATION_GROUP,
    )

    rows = []

    for seed in SEEDS:
        print(f"\nRunning ablation with seed {seed}...")

        rows.append(
            run_one_seed(
                splits.X_train,
                splits.y_train,
                splits.X_val,
                splits.y_val,
                splits.X_test,
                splits.y_test,
                game_ids,
                dropped,
                seed,
            )
        )

    ablation_summary = pd.DataFrame(rows)
    overall_passes = bool(ablation_summary["passes"].all())
    ablation_summary["overall_passes"] = overall_passes

    print("\n=== Feature ablation: reduced - full ===")
    print(ablation_summary.to_string(index=False))

    if overall_passes:
        print(
            f"\nAll five seeds pass: the group can be dropped "
            f"under the pre-registered tolerance epsilon={EPSILON}."
        )
    else:
        print(
            "\nNon-inferiority was not established for all five seeds: "
            "keep the group. This does not prove that removing it is worse."
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("model_evaluation") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_summary.to_csv(
        output_dir / "ablation.csv",
        index=False,
    )

    print(f"\nAblation results saved in {output_dir}/ablation.csv")