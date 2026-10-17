"""V2 model training (chess_insights_spec.md, section "Models").

Baseline logistic regression first, then LightGBM. Per spec: if LightGBM
can't clearly beat the baseline, the features are the problem, not the
model -- so the baseline's honest metrics matter as a real checkpoint, not
a formality to skip past.
"""

from __future__ import annotations

import tomllib
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.model.dataset import load_model_splits


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
    random_state: int = 42,
):
    """Main model per spec. LightGBM handles NaN natively - unlike the
    baseline, no imputation/scaling needed here. Categorical features are
    already one-hot encoded by prepare_xy.
    No class rebalancing is used to keep training aligned with unweighted
    validation logloss (see the decisions log in PROJECT_TRACKER.md).
    """

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        random_state=random_state,
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


def check_early_stopping(
    model,
    best_iteration_warning_threshold: int,
) -> None:
    """Warn when early stopping selects unusually few boosting iterations.
    A best_iteration_ of 0 is the sentinel for no active early stopping,
    so it is excluded from this check.

    Emit a warning rather than an exception because a low best iteration
    can be legitimate when features contain little predictive signal.

    Motivated by the 2026-10-08 incident: class rebalancing conflicted with
    unweighted validation logloss, producing a one-tree model that went
    undetected for two evaluation runs.
    """
    best_iteration = model.best_iteration_

    if 0 < best_iteration < best_iteration_warning_threshold:
        warnings.warn(
            f"best_iteration_={best_iteration} is below "
            f"best_iteration_warning_threshold={best_iteration_warning_threshold}. "
            "Possible mismatch between the training objective and the early-stopping "
            "metric (e.g. class rebalancing with unweighted validation logloss), "
            "or features with little predictive signal. "
            "Check class weighting, the validation metric, and feature quality.",
            UserWarning,
            stacklevel=2,
    )


if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    splits = load_model_splits(config)

    train = splits.train
    val = splits.val
    test = splits.test
    X_train = splits.X_train
    y_train = splits.y_train
    X_val = splits.X_val
    y_val = splits.y_val
    X_test = splits.X_test
    y_test = splits.y_test

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

    check_early_stopping(
        lightgbm_model,
        config["thresholds"]["best_iteration_warning_threshold"],
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