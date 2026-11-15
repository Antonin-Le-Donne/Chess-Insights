"""Shared access to the V2 test predictions and their risk bands.

One definition of the bands (spec V2, "Risk page layout", 2026-11-15) for
every page that shows them: low = bottom 50% of predicted risk on the test
moves, high = top 10%, medium = in between.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
EVALUATION_DIR = PROJECT_ROOT / "model_evaluation"
PREDICTIONS_FILE = "test_predictions.parquet"

LOW_BAND_QUANTILE = 0.50
HIGH_BAND_QUANTILE = 0.90
BAND_ORDER = ["Low risk", "Medium risk", "High risk"]


def find_latest_predictions() -> Path | None:
    """Most recent model_evaluation/<timestamp>/test_predictions.parquet.

    Run folders are named YYYY-MM-DD_HH-MM-SS, so names sort in time order.
    """
    files = sorted(
        EVALUATION_DIR.glob(f"*/{PREDICTIONS_FILE}"),
        key=lambda path: path.parent.name,
    )
    return files[-1] if files else None


def add_risk_band(predictions: pd.DataFrame) -> pd.Series:
    """Low / Medium / High risk band of each test move."""
    risk = predictions["predicted_risk"]
    low_cut = risk.quantile(LOW_BAND_QUANTILE)
    high_cut = risk.quantile(HIGH_BAND_QUANTILE)
    band = pd.Series("Medium risk", index=predictions.index)
    band[risk < low_cut] = "Low risk"
    band[risk >= high_cut] = "High risk"
    return band


def band_rates(predictions: pd.DataFrame) -> pd.DataFrame:
    """Blunder rate, move count and blunder count per band, in BAND_ORDER."""
    return (
        predictions.assign(band=add_risk_band(predictions))
        .groupby("band")["is_blunder"]
        .agg(rate="mean", n="size", blunders="sum")
        .reindex(BAND_ORDER)
    )
