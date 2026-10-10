"""V2 model training (chess_insights_spec.md, section "Models").

Baseline logistic regression first, then LightGBM. Per spec: if LightGBM
can't clearly beat the baseline, the features are the problem, not the
model -- so the baseline's honest metrics matter as a real checkpoint, not
a formality to skip past.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import sklearn
import tomllib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.model.dataset import (
    assemble_features,
    temporal_game_split,
)
from src.scope import filter_current_regime

# Columns that exist on assemble_features's output but are NOT model inputs:
# identifiers/metadata (game_id, san, uci, fen_before, date_utc, platform,
# eval_source), and the target itself (move_quality, y).
FEATURE_COLUMNS = [
    "move_number",
    "ply",
    "phase",
    "rolling_time_mean_3",
    "previous_long_think",
    "eval_volatility",
    "material_count",
    "material_imbalance",
    "queens_on_board",
    "n_legal_moves",
    "hanging_pieces",
    "checks_available",
    "my_rating",
    "rating_diff",
    "perf_type",
    "opening_family",
    "eval_before_move",
    "clock_before_move",
    "time_pressure_before_move",
    "eval_state_before_move",
    "clock_before_move_pct",
    "has_increment",
    "previous_time_spent",
]

def prepare_xy(dataset: pd.DataFrame,
    reference_columns: pd.Index 
    | None = None,) -> tuple[pd.DataFrame, pd.Series]:
    """Slice FEATURE_COLUMNS + y out of one of temporal_game_split's three
    outputs (train/val/test all go through this the same way).

    Categorical features are one-hot encoded. When reference_columns is
    provided (for val/test), columns are aligned to the training set so
    unseen categories do not create incompatible feature shapes.
    """

    X = dataset[
        FEATURE_COLUMNS
    ].copy()

    y = dataset["y"].copy()

    categorical_columns = [
        "phase",
        "perf_type",
        "opening_family",
        "eval_state_before_move",
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=int,
    )

    if reference_columns is not None:
        X = X.reindex(
            columns=reference_columns,
            fill_value=0,
        )

    return X, y

def train_baseline_logistic(X_train: pd.DataFrame,y_train: pd.Series,):
    """Logistic regression on scaled features -- the checkpoint per spec:
    if LightGBM can't clearly beat this, the features need work, not the
    model choice.
    """

    model = sklearn.pipeline.Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model
    


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    """Main model per spec. LightGBM handles NaN natively - unlike the
    baseline, no imputation/scaling needed here. Categorical features are
    already one-hot encoded by prepare_xy.
    """

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        feature_fraction=0.8,
        min_child_samples=10,
    )

    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False,
            )
        ],
    )

    return model


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

    X_train, y_train = prepare_xy(
        train
    )

    X_val, y_val = prepare_xy(
        val,
        reference_columns=X_train.columns,
    )

    X_test, y_test = prepare_xy(
        test,
        reference_columns=X_train.columns,
    )

    baseline_model = train_baseline_logistic(
        X_train,
        y_train,
    )

    lightgbm_model = train_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    model_dir = Path("models")
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        baseline_model,
        model_dir / "baseline_logistic.joblib",
    )

    joblib.dump(
        lightgbm_model,
        model_dir / "lightgbm.joblib",
    )

    joblib.dump(
        X_train.columns.tolist(),
        model_dir / "feature_columns.joblib",
    )

    print(
        f"Train: {len(train)} rows"
    )

    print(
        f"Validation: {len(val)} rows"
    )

    print(
        f"Test: {len(test)} rows"
    )

    print(
        "Models saved in models/"
    )