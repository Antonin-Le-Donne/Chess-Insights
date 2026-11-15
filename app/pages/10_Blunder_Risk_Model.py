"""V2 — Blunder-risk model (spec V2, "Evaluation"), laid out with the
dashboard principles (spec section 6): does the model work, then what drives
it; the SHAP dependence plot and the calibration sit in expanders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.controls import BADGE_GRAY, BADGE_GREEN, render_badge
from app.filters import game_url, render_filters_not_applied, selected_points
from app.model_labels import feature_label
from app.ui import load_ui

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVALUATION_DIR = PROJECT_ROOT / "model_evaluation"

TOP_N_FEATURES = 8

CATEGORICAL_PREFIXES: dict[str, str] = {
    "phase_": "phase",
    "perf_type_": "perf_type",
    "opening_family_": "opening_family",
    "eval_state_before_move_": "eval_state_before_move",
}

PLOTLY_LAYOUT = go.Layout(
    font=dict(family="Inter", size=14, color="#DCE3E9"),
    hovermode="closest",
    template="plotly_dark",
    margin=dict(l=40, r=35, t=45, b=35),
)


def find_latest_shap_file() -> Path | None:
    """Most recent model_evaluation/<timestamp>/shap_values.parquet, or None.

    Run folders are named YYYY-MM-DD_HH-MM-SS, so their names sort in time
    order. Ablation runs have no shap_values.parquet -- skip them.
    """
    shap_files = sorted(
        EVALUATION_DIR.glob("*/shap_values.parquet"),
        key=lambda path: path.parent.name,
    )
    return shap_files[-1] if shap_files else None


@st.cache_data
def load_shap_values(path: str) -> pd.DataFrame:
    """Cached on the path: a new evaluation run = a new path = a fresh load."""
    return pd.read_parquet(path)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_base_rate(path: str) -> tuple[float, int]:
    """Blunder rate and move count of the test set (test_predictions.parquet)."""
    labels = pd.read_parquet(path, columns=["is_blunder"])["is_blunder"]
    return float(labels.mean()), len(labels)


def plot_calibration(calibration: pd.DataFrame) -> None:
    """Predicted risk (x) vs observed blunder rate (y), one line per model,
    plus the y = x diagonal (spec: "usable as a real probability")."""
    calibration = calibration.sort_values(["model", "predicted_mean"])
    fig = px.line(
        calibration,
        x="predicted_mean",
        y="actual_rate",
        color="model",
        markers=True,
        hover_data=["n_in_bin"],
        labels={
            "predicted_mean": "Mean predicted risk",
            "actual_rate": "Observed blunder rate",
            "model": "Model",
            "n_in_bin": "Moves in bin",
        },
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="gray", dash="dash"), layer="below",
    )
    axis_max = min(
        1.0,
        1.1 * max(calibration["predicted_mean"].max(), calibration["actual_rate"].max()),
    )
    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
    fig.update_xaxes(tickformat=".0%", range=[0, axis_max], title="Mean predicted risk", constrain="domain")
    fig.update_yaxes(tickformat=".0%", range=[0, axis_max], title="Observed blunder rate",
                     scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, width="stretch")


def prepare_plot_data(shap_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Plot features, with one-hot columns combined into their categorical group
    (value = the category, attribution = the sum of its one-hot SHAP values)."""
    plot_data = shap_data.copy()
    raw_features = [
        column
        for column in plot_data.columns
        if column not in {"game_id", "ply"}
        and not column.startswith("shap_")
        and f"shap_{column}" in plot_data.columns
    ]
    one_hot_columns = {c for c in raw_features if c.startswith(tuple(CATEGORICAL_PREFIXES))}
    available_features = [
        c for c in raw_features
        if c not in one_hot_columns and pd.api.types.is_numeric_dtype(plot_data[c])
    ]
    for prefix, feature_name in CATEGORICAL_PREFIXES.items():
        group_columns = sorted(c for c in raw_features if c.startswith(prefix))
        if not group_columns:
            continue
        values = plot_data[group_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        categories = values.idxmax(axis=1).str.slice(start=len(prefix))
        categories = categories.where(values.max(axis=1) > 0, "Unknown")
        shap_columns = [f"shap_{c}" for c in group_columns if f"shap_{c}" in plot_data.columns]
        if not shap_columns:
            continue
        plot_data[feature_name] = categories
        plot_data[f"shap_{feature_name}"] = plot_data[shap_columns].sum(axis=1)
        if feature_name not in available_features:
            available_features.append(feature_name)
    return plot_data, available_features


def build_feature_summary(
    plot_data: pd.DataFrame,
    available_features: list[str],
    summary_path: Path,
) -> pd.DataFrame:
    """Rank features with the run's SHAP summary, or from the SHAP values."""
    summary = pd.DataFrame()
    if summary_path.exists():
        summary = load_csv(str(summary_path))
    if {"feature", "mean_abs_shap"}.issubset(summary.columns):
        summary = summary[["feature", "mean_abs_shap"]].copy()
        summary["feature"] = summary["feature"].astype(str)
        summary["mean_abs_shap"] = pd.to_numeric(summary["mean_abs_shap"], errors="coerce")
        summary = summary.dropna(subset=["mean_abs_shap"])
        summary = summary[summary["feature"].isin(available_features)].drop_duplicates("feature")
    if summary.empty:
        rows = []
        for feature in available_features:
            column = f"shap_{feature}"
            if column in plot_data.columns:
                importance = pd.to_numeric(plot_data[column], errors="coerce").abs().mean()
                if pd.notna(importance):
                    rows.append({"feature": feature, "mean_abs_shap": float(importance)})
        summary = pd.DataFrame(rows, columns=["feature", "mean_abs_shap"])
    return summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def render_answer(run_dir: Path, top_features: list[str]) -> None:
    """Does the model work? Read from the run's evaluation files."""
    metrics_path = run_dir / "metrics_summary.csv"
    paired_path = run_dir / "paired_bootstrap.csv"
    predictions_path = run_dir / "test_predictions.parquet"
    if not (metrics_path.exists() and paired_path.exists()):
        st.info("This evaluation run has no metrics summary; re-run `python -m src.model.evaluate_model`.")
        return
    metrics = load_csv(str(metrics_path)).set_index("model")
    paired = load_csv(str(paired_path)).iloc[0]
    lightgbm, baseline = metrics.loc["lightgbm"], metrics.loc["baseline_logistic"]
    beats = paired["ci_lower"] > 0

    render_badge(
        "Beats the simple baseline" if beats else "No clear gain over the simple baseline",
        BADGE_GREEN if beats else BADGE_GRAY,
    )
    random_text = ""
    if predictions_path.exists():
        base_rate, n_moves = load_base_rate(str(predictions_path))
        random_text = (
            f" Random guessing would score **{base_rate:.3f}** (the blunder rate of these "
            f"{n_moves:,} test moves)."
        )
    drivers = ", ".join(feature_label(f).lower() for f in top_features[:3])
    st.markdown(
        f"#### On my most recent games, which the model never saw, it ranks risky moves "
        f"{'better than' if beats else 'about as well as'} a simple model: PR-AUC "
        f"**{lightgbm['pr_auc']:.3f}** against **{baseline['pr_auc']:.3f}** "
        f"({paired['pr_auc_delta']:+.3f}, 95% interval {paired['ci_lower']:+.3f} to "
        f"{paired['ci_upper']:+.3f}).{random_text} What weighs most in its predictions: "
        f"{drivers}."
    )


def render_top_factors(top_summary: pd.DataFrame) -> None:
    bar_data = top_summary.copy()
    bar_data["label"] = bar_data["feature"].map(feature_label)
    fig = px.bar(
        bar_data, x="mean_abs_shap", y="label", orientation="h",
        labels={"mean_abs_shap": "Average weight in the predictions (mean |SHAP|)", "label": ""},
    )
    fig.update_traces(marker_color="#C7F36B", texttemplate="%{x:.2f}", textposition="outside",
                      cliponaxis=False)
    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
    fig.update_layout(
        height=max(300, 42 * len(bar_data) + 70),
        showlegend=False,
        margin=dict(l=25, r=75, t=15, b=35),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        yaxis=dict(title="", categoryorder="total ascending", showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "How much each factor moves the predictions on average, up or down. It says what the "
        "model relies on, not whether the factor raises or lowers the risk (see the next section)."
    )


def render_dependence(plot_data: pd.DataFrame, top_features: list[str]) -> None:
    """SHAP dependence plot for one feature, coloured by another."""
    col1, col2 = st.columns(2)
    default_x = top_features.index("eval_before_move") if "eval_before_move" in top_features else 0
    with col1:
        x_feature = st.selectbox("Factor", top_features, index=default_x, format_func=feature_label)
    color_options = [f for f in top_features if f != x_feature] or top_features
    default_color = color_options.index("material_imbalance") if "material_imbalance" in color_options else 0
    with col2:
        color_feature = st.selectbox(
            "Colour points by", color_options, index=default_color, format_func=feature_label
        )

    shap_column = f"shap_{x_feature}"
    data = plot_data.dropna(subset=[x_feature, shap_column, color_feature]).copy()
    if data.empty:
        st.warning("No complete observations are available for these factors.")
        return

    x_label, color_label = feature_label(x_feature), feature_label(color_feature)
    scatter_kwargs: dict[str, object] = {}
    if pd.api.types.is_numeric_dtype(data[color_feature]):
        scatter_kwargs["color_continuous_scale"] = ["#2E7DFF", "#B8E35A", "#F4BE55"]
        if color_feature == "material_imbalance":
            scatter_kwargs["range_color"] = (-10, 10)
        else:
            values = pd.to_numeric(data[color_feature], errors="coerce").dropna()
            if not values.empty and values.quantile(0.02) < values.quantile(0.98):
                scatter_kwargs["range_color"] = (float(values.quantile(0.02)), float(values.quantile(0.98)))
    else:
        scatter_kwargs["color_discrete_sequence"] = ["#2E7DFF", "#B8E35A", "#F4BE55", "#C084FC", "#F87171", "#38BDF8"]

    fig = px.scatter(
        data, x=x_feature, y=shap_column, color=color_feature,
        custom_data=["game_id", "ply", color_feature],
        labels={x_feature: x_label, shap_column: f"Push on the risk — {x_label}", color_feature: color_label},
        opacity=0.75, **scatter_kwargs,
    )
    fig.update_traces(
        marker=dict(size=7, line=dict(width=0)),
        hovertemplate=(
            "<b>Game:</b> %{customdata[0]}<br><b>Half-move:</b> %{customdata[1]}<br>"
            f"<b>{x_label}:</b> %{{x}}<br><b>Push on the risk:</b> %{{y:.3f}}<br>"
            f"<b>{color_label}:</b> %{{customdata[2]}}<extra></extra>"
        ),
    )
    fig.add_hline(y=0, line_dash="dash", line_width=1.5, line_color="#F4BE55")
    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())
    fig.update_layout(
        height=520,
        xaxis=dict(title=x_label, showgrid=False, zeroline=False),
        yaxis=dict(title=f"Push on the risk — {x_label}", gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    event = st.plotly_chart(
        fig, width="stretch", config={"displayModeBar": False},
        on_select="rerun", selection_mode="points", key="shap_dependence_chart",
    )
    points = selected_points(event)
    if points and points[0].get("customdata"):
        game_id, ply = points[0]["customdata"][:2]
        # Open the board BEFORE the move: the model predicts from that position.
        st.markdown(f"Selected move: [{game_id}, half-move {int(ply)}]({game_url(game_id, ply=int(ply) - 1)})")
    st.caption(
        "Each point is one test move. Above the dashed line, that factor pushed the predicted "
        "risk up; below, down (in log-odds). Click a point for a link to the position. "
        "Reading rule (2026-10-17): the 20-point blunder label alone produces a curve peaking "
        "near 60% win probability and symmetric around it, so only a peak clearly elsewhere, "
        "or an asymmetry, is a candidate behavioural pattern."
    )


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def main() -> None:
    st.title("V2 — Blunder-risk model")
    render_filters_not_applied("the model is evaluated once, on a fixed chronological test set.")
    st.markdown("**Question:** can a model tell, before I move, how likely I am to blunder — and what drives that risk?")

    shap_path = find_latest_shap_file()
    if shap_path is None:
        st.warning(
            "No evaluation run found. Run `python -m src.model.train`, then "
            "`python -m src.model.evaluate_model`, from the project root."
        )
        return
    run_dir = shap_path.parent
    plot_data, available_features = prepare_plot_data(load_shap_values(str(shap_path)))
    summary = build_feature_summary(plot_data, available_features, run_dir / "shap_summary.csv")
    top_summary = summary.head(TOP_N_FEATURES)
    top_features = top_summary["feature"].tolist()

    render_answer(run_dir, top_features)
    render_top_factors(top_summary)

    with st.expander("How each factor pushes the risk up or down"):
        render_dependence(plot_data, top_features)
    with st.expander("Is the predicted risk a real probability? (calibration)"):
        calibration_path = run_dir / "calibration.csv"
        if calibration_path.exists():
            plot_calibration(load_csv(str(calibration_path)))
            st.caption(
                "On the diagonal, a predicted 10% means 10% of those moves really were "
                "blunders. Above it, the model under-predicts; below it, it over-predicts."
            )
        else:
            st.info("No calibration data in this run.")
    with st.expander("How this is computed"):
        st.markdown(
            f"""
            - **What it predicts:** whether one of my moves is a blunder (a drop of 20 win%
              points or more), using only what was known before I played it: position,
              clock, recent evaluation swings, my recent pace. No player ratings
              (chess.com records them after the game, which would leak the result).
            - **Which games:** my current level only (rating floors in `config.toml`), my moves
              after move 5, positions where a blunder is possible (win% ≥ 20).
            - **Tested on my most recent games**, held out chronologically: the model never saw
              them while training.
            - **PR-AUC:** how well the model puts the rare blunders at the top of its list;
              random guessing scores the blunder rate. The comparison with the simple model
              (logistic regression) uses a paired bootstrap on whole games.
            - **Training noise:** the same model varies by about 0.01 PR-AUC from one training
              run to the next, so read small differences with care.
            - **Factors (SHAP):** explain how the model uses its inputs, not what causes a
              blunder. Evaluation run: `{run_dir.name}`.
            """
        )


if __name__ == "__main__":
    main()
