"""V2 model evaluation (chess_insights_spec.md, section "Evaluation").

Loads the models train.py already saved (models/*.joblib) and the same
test set (rebuilt the same way train.py built it -- the split is temporal
and deterministic, so re-running assemble_features + temporal_game_split
reproduces the identical train/val/test partition, no need to persist it
separately).

Per spec: ROC-AUC + PR-AUC (PR-AUC is the honest one under imbalance),
calibration, SHAP ("the actual payoff"), and slice analysis. Every number
here should ship with its uncertainty, not as a bare point estimate --
same discipline as the M3/quality-of-play/performance-rating discussions
this whole project has been built on. ~12,801 test rows but far fewer
positives (~5% rate) is not a lot of ground to stand a precise metric on.
"""

from __future__ import annotations

import tomllib
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model.dataset import load_model_splits


def load_models(model_dir: str = "models") -> dict:
    """Load baseline_logistic.joblib, lightgbm.joblib, feature_columns.joblib."""

    model_path = Path(model_dir)

    baseline = joblib.load(
        model_path / "baseline_logistic.joblib"
    )

    lightgbm = joblib.load(
        model_path / "lightgbm.joblib"
    )

    feature_columns = joblib.load(
        model_path / "feature_columns.joblib"
    )

    return {
        "baseline": baseline,
        "lightgbm": lightgbm,
        "feature_columns": feature_columns,
    }


def rebuild_test_set(
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Reproduce train.py's exact test split using the shared loader.

    Returns (test_raw, X_test, y_test). Checks that the freshly generated
    training feature columns match the saved feature_columns.joblib exactly,
    including their order, to detect models trained on different data/features.
    """

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    splits = load_model_splits(config)

    if list(feature_columns) != list(splits.X_train.columns):
        raise ValueError(
            "Saved feature_columns.joblib does not match the freshly "
            "generated training feature columns. The saved model may have "
            "been trained on different data or features. "
            f"Saved columns: {len(feature_columns)}; "
            f"current columns: {len(splits.X_train.columns)}."
        )

    return splits.test, splits.X_test, splits.y_test


def auc_metrics(model,X_test: pd.DataFrame,y_test: pd.Series,) -> dict:
    """Point estimates: ROC-AUC and PR-AUC (average_precision_score).
    Point estimates only -- see bootstrap_pr_auc_ci for the honest version.
    """

    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    pr_auc = average_precision_score(
        y_test,
        y_proba,
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def bootstrap_pr_auc_ci(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    game_ids: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI on PR-AUC, resampled at the GAME level -- same leakage
    discipline as temporal_game_split itself. Resampling rows independently
    would treat correlated moves from the same game as independent evidence,
    understating the true uncertainty. Returns (point_estimate, lower, upper).
    """

    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    y_array = y_test.to_numpy()
    game_ids_array = game_ids.to_numpy()

    point_estimate = average_precision_score(
        y_array,
        y_proba,
    )

    unique_games = np.unique(
        game_ids_array
    )

    game_positions = {
        game_id: np.flatnonzero(
            game_ids_array == game_id
        )
        for game_id in unique_games
    }

    rng = np.random.default_rng(
        42
    )

    bootstrap_scores = []

    for _ in range(n_bootstrap):
        sampled_games = rng.choice(
            unique_games,
            size=len(unique_games),
            replace=True,
        )

        sampled_positions = np.concatenate(
            [
                game_positions[game_id]
                for game_id in sampled_games
            ]
        )

        y_bootstrap = y_array[
            sampled_positions
        ]

        proba_bootstrap = y_proba[
            sampled_positions
        ]

        score = average_precision_score(
            y_bootstrap,
            proba_bootstrap,
        )

        bootstrap_scores.append(
            score
        )

    alpha = (
        1 - ci
    ) / 2

    lower, upper = np.percentile(
        bootstrap_scores,
        [
            alpha * 100,
            (1 - alpha) * 100,
        ],
    )

    return (
        float(point_estimate),
        float(lower),
        float(upper),
    )


def paired_bootstrap_pr_auc_diff(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    y_test: pd.Series,
    game_ids: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float, float]:
    """Paired, game-level bootstrap of PR-AUC(A) - PR-AUC(B) on the same test rows.

    Both models are scored on the SAME resampled games at each iteration, so
    the shared "this sample is easy/hard" variation cancels in the difference
    (Var(A-B) = Var(A) + Var(B) - 2 Cov(A, B)). Two separate CIs that overlap
    say nothing about the difference; this does.

    Returns (delta, lower, upper, proportion_a_better).
    """
    if not (len(proba_a) == len(proba_b) == len(y_test) == len(game_ids)):
        raise ValueError(
            "proba_a, proba_b, y_test and game_ids must have the same length"
        )

    y_array = y_test.to_numpy()
    game_ids_array = game_ids.to_numpy()

    delta = (
        average_precision_score(y_array, proba_a)
        - average_precision_score(y_array, proba_b)
    )

    unique_games = np.unique(game_ids_array)

    game_positions = {
        game_id: np.flatnonzero(game_ids_array == game_id)
        for game_id in unique_games
    }

    rng = np.random.default_rng(42)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        sampled_games = rng.choice(
            unique_games,
            size=len(unique_games),
            replace=True,
        )

        sampled_positions = np.concatenate(
            [
                game_positions[game_id]
                for game_id in sampled_games
            ]
        )

        y_bootstrap = y_array[sampled_positions]

        score_a = average_precision_score(
            y_bootstrap,
            proba_a[sampled_positions],
        )

        score_b = average_precision_score(
            y_bootstrap,
            proba_b[sampled_positions],
        )

        bootstrap_diffs.append(score_a - score_b)

    diffs_array = np.asarray(bootstrap_diffs)

    alpha = (1 - ci) / 2

    lower, upper = np.percentile(
        diffs_array,
        [
            alpha * 100,
            (1 - alpha) * 100,
        ],
    )

    proportion_a_better = (diffs_array > 0).mean()

    return (
        float(delta),
        float(lower),
        float(upper),
        float(proportion_a_better),
    )


def calibration_data(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted probability vs actual blunder rate, binned -- spec wants the
    output "usable as a real probability", not just a ranking.
    """

    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    bins = pd.qcut(
        y_proba,
        q=n_bins,
        duplicates="drop",
    )

    calibration = (
        pd.DataFrame(
            {
                "y": y_test.to_numpy(),
                "y_proba": y_proba,
                "bin": bins,
            }
        )
        .groupby(
            "bin",
            observed=True,
        )
        .agg(
            predicted_mean=("y_proba", "mean"),
            actual_rate=("y", "mean"),
            n_in_bin=("y", "size"),
        )
        .reset_index(drop=True)
    )

    return calibration


def compute_shap_values(
    model,
    X_sample: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values once and share them between feature ranking and
    plots so both use exactly the same attributions.

    Returns an array with shape (n_rows, n_columns) for the positive class
    (blunder).
    """

    explainer = shap.TreeExplainer(
        model
    )

    explanation = explainer(
        X_sample
    )

    shap_values = explanation.values

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    return np.asarray(
        shap_values
    )


def save_shap_values(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    test_raw: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save feature values and their corresponding SHAP values to a Parquet
    file, including game_id and ply to identify each move in its game.
    """

    metadata = test_raw.loc[
        X_sample.index,
        ["game_id", "ply"],
    ]

    shap_columns = [
        f"shap_{column}"
        for column in X_sample.columns
    ]

    shap_df = pd.DataFrame(
        shap_values,
        index=X_sample.index,
        columns=shap_columns,
    )

    results = pd.concat(
        [
            metadata,
            X_sample,
            shap_df,
        ],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        output_path,
        index=False,
    )


def shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate mean absolute SHAP importance for LightGBM, recombining
    one-hot encoded categorical columns into their original feature groups.
    """

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns,
        name="mean_abs_shap",
    )

    categorical_prefixes: dict[str, str] = {
        "phase_": "phase",
        "perf_type_": "perf_type",
        "opening_family_": "opening_family",
        "eval_state_before_move_": "eval_state_before_move",
    }

    grouped_importance: dict[str, float] = {}

    for column, value in importance.items():
        column_name = str(column)
        feature_name = column_name

        for prefix, original_feature in categorical_prefixes.items():
            if column_name.startswith(prefix):
                feature_name = original_feature
                break

        grouped_importance[feature_name] = (
            grouped_importance.get(feature_name, 0.0)
            + float(value)
        )

    summary = pd.DataFrame(
        {
            "feature": list(grouped_importance.keys()),
            "mean_abs_shap": list(grouped_importance.values()),
        }
    )

    summary = summary.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return summary


def slice_analysis(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_raw: pd.DataFrame,
    slice_columns: list[str] = [
        "phase",
        "perf_type",
        "eval_state_before_move",
    ],
) -> pd.DataFrame:
    """AUC/PR-AUC per slice value (per spec: phase, perf_type, eval_state).
    Report n_positives alongside every slice score -- a slice with 8 positive
    examples doesn't deserve the same confidence as one with 800.
    """

    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    y_proba = pd.Series(
        y_proba,
        index=X_test.index,
    )

    results: list[dict] = []

    min_positives = 20

    for column in slice_columns:
        for value in test_raw[column].dropna().unique():
            mask = (
                test_raw[column] == value
            )

            slice_indices = test_raw.index[
                mask
            ]

            y_slice = y_test.loc[
                slice_indices
            ]

            proba_slice = y_proba.loc[
                slice_indices
            ]

            n_rows = len(
                y_slice
            )

            n_positives = int(
                y_slice.sum()
            )

            n_negatives = (
                n_rows - n_positives
            )

            if (
                n_positives < min_positives
                or n_negatives == 0
            ):
                roc_auc = float("nan")
                pr_auc = float("nan")
            else:
                roc_auc = roc_auc_score(
                    y_slice,
                    proba_slice,
                )

                pr_auc = average_precision_score(
                    y_slice,
                    proba_slice,
                )

            results.append(
                {
                    "slice_column": column,
                    "slice_value": value,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "n_rows": n_rows,
                    "n_positives": n_positives,
                }
            )

    return pd.DataFrame(
        results
    )


if __name__ == "__main__":
    models = load_models()

    baseline_model = models["baseline"]
    lightgbm_model = models["lightgbm"]
    feature_columns = models["feature_columns"]

    test_raw, X_test, y_test = rebuild_test_set(
        feature_columns
    )

    game_ids = test_raw["game_id"]

    baseline_metrics = auc_metrics(
        baseline_model,
        X_test,
        y_test,
    )

    lightgbm_metrics = auc_metrics(
        lightgbm_model,
        X_test,
        y_test,
    )

    baseline_pr_auc, baseline_ci_lower, baseline_ci_upper = (
        bootstrap_pr_auc_ci(
            baseline_model,
            X_test,
            y_test,
            game_ids,
        )
    )

    lightgbm_pr_auc, lightgbm_ci_lower, lightgbm_ci_upper = (
        bootstrap_pr_auc_ci(
            lightgbm_model,
            X_test,
            y_test,
            game_ids,
        )
    )

    proba_lightgbm = lightgbm_model.predict_proba(X_test)[:, 1]
    proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

    delta, lower, upper, proportion_a_better = paired_bootstrap_pr_auc_diff(
        proba_a=proba_lightgbm,
        proba_b=proba_baseline,
        y_test=y_test,
        game_ids=game_ids,
    )

    paired_summary = pd.DataFrame(
        [
            {
                "model_a": "lightgbm",
                "model_b": "baseline_logistic",
                "pr_auc_delta": delta,
                "ci_lower": lower,
                "ci_upper": upper,
                "proportion_a_better": proportion_a_better,
            }
        ]
    )

    print("\n=== Paired bootstrap: LightGBM - baseline ===")
    print(paired_summary.to_string(index=False))

    if lower > 0:
        print("LightGBM clearly beats the baseline.")
    elif upper < 0:
        print("The baseline clearly beats LightGBM.")
    else:
        print("The CI includes 0: no clear difference.")

    metrics_summary = pd.DataFrame(
        [
            {
                "model": "baseline_logistic",
                "roc_auc": baseline_metrics["roc_auc"],
                "pr_auc": baseline_metrics["pr_auc"],
                "pr_auc_ci_lower": baseline_ci_lower,
                "pr_auc_ci_upper": baseline_ci_upper,
                "best_iteration": float("nan"),
            },
            {
                "model": "lightgbm",
                "roc_auc": lightgbm_metrics["roc_auc"],
                "pr_auc": lightgbm_metrics["pr_auc"],
                "pr_auc_ci_lower": lightgbm_ci_lower,
                "pr_auc_ci_upper": lightgbm_ci_upper,
                "best_iteration": lightgbm_model.best_iteration_,
            },
        ]
    )

    baseline_calibration = calibration_data(
        baseline_model,
        X_test,
        y_test,
    )

    baseline_calibration["model"] = (
        "baseline_logistic"
    )

    lightgbm_calibration = calibration_data(
        lightgbm_model,
        X_test,
        y_test,
    )

    lightgbm_calibration["model"] = (
        "lightgbm"
    )

    calibration_summary = pd.concat(
        [
            baseline_calibration,
            lightgbm_calibration,
        ],
        ignore_index=True,
    )

    baseline_slices = slice_analysis(
        baseline_model,
        X_test,
        y_test,
        test_raw,
    )

    baseline_slices["model"] = (
        "baseline_logistic"
    )

    lightgbm_slices = slice_analysis(
        lightgbm_model,
        X_test,
        y_test,
        test_raw,
    )

    lightgbm_slices["model"] = (
        "lightgbm"
    )

    slice_summary = pd.concat(
        [
            baseline_slices,
            lightgbm_slices,
        ],
        ignore_index=True,
    )

    shap_sample = X_test.sample(
        n=min(3000, len(X_test)),
        random_state=42,
    )

    shap_values = compute_shap_values(
        lightgbm_model,
        shap_sample,
    )

    shap_results = shap_summary(
        shap_values,
        shap_sample,
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_dir = (
        Path("model_evaluation")
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_shap_values(
        shap_values,
        shap_sample,
        test_raw,
        output_dir / "shap_values.parquet",
    )

    metrics_summary.to_csv(
        output_dir / "metrics_summary.csv",
        index=False,
    )

    calibration_summary.to_csv(
        output_dir / "calibration.csv",
        index=False,
    )

    slice_summary.to_csv(
        output_dir / "slice_analysis.csv",
        index=False,
    )

    shap_results.to_csv(
        output_dir / "shap_summary.csv",
        index=False,
    )

    paired_summary.to_csv(
        output_dir / "paired_bootstrap.csv",
        index=False,
    )

    print("\n=== Model comparison ===")
    print(
        metrics_summary.to_string(
            index=False
        )
    )

    print("\n=== LightGBM SHAP summary ===")
    print(
        shap_results.to_string(
            index=False
        )
    )

    print(
        "\nEvaluation artifacts saved in "
        f"{output_dir}/"
    )