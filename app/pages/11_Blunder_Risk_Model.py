from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

FEATURE_LABELS: dict[str, str] = {
    "eval_before_move": "Position evaluation before the move",
    "material_imbalance": "Material imbalance",
    "material_count": "Material remaining on the board",
    "queens_on_board": "Queens remaining on the board",
    "n_legal_moves": "Number of legal moves",
    "hanging_pieces": "Pieces under attack",
    "checks_available": "Available checks",
    "eval_volatility": "Evaluation volatility",
    "rolling_time_mean_3": "Average time spent on the last 3 moves (s)",
    "previous_long_think": "Long think on the previous move",
    "move_number": "Move number",
    "ply": "Half-move number",
    "my_rating": "My rating",
    "rating_diff": "Rating difference between players",
    "clock_before_move": "Time remaining before the move (s)",
    "clock_before_move_pct": "Percentage of time remaining",
    "time_pressure_before_move": "Time pressure before the move",
    "previous_time_spent": "Time spent on the previous move (s)",
    "has_increment": "Time increment available",
    "phase": "Game phase",
    "perf_type": "Time control",
    "opening_family": "Opening family",
    "eval_state_before_move": "Position evaluation state before the move",
}

PLOTLY_LAYOUT = go.Layout(
    font=dict(
        family="Inter",
        size=14,
        color="#DCE3E9",
    ),
    hovermode="closest",
    template="plotly_dark",
    margin=dict(
        l=40,
        r=35,
        t=45,
        b=35,
    ),
)


def feature_label(feature: str) -> str:
    """Return a readable label for a model feature."""

    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    return feature.replace("_", " ").capitalize()


def find_latest_shap_file() -> Path | None:
    """Most recent model_evaluation/<timestamp>/shap_values.parquet, or None.

    Run folders are named YYYY-MM-DD_HH-MM-SS, so their names sort in time
    order. Ablation runs have no shap_values.parquet -- skip them.
    """

    shap_files = sorted(
        EVALUATION_DIR.glob("*/shap_values.parquet"),
        key=lambda path: path.parent.name,
    )

    if not shap_files:
        return None

    return shap_files[-1]


@st.cache_data
def load_shap_values(path: str) -> pd.DataFrame:
    """Cached on the path (a str, hashable): a new evaluation run = a new
    path = a fresh load, without restarting Streamlit.
    """

    return pd.read_parquet(path)


@st.cache_data
def load_shap_summary(path: str) -> pd.DataFrame:
    """Load the SHAP feature-importance ranking for one evaluation run."""

    return pd.read_csv(path)


@st.cache_data
def load_calibration(path: str) -> pd.DataFrame:
    """Calibration table written by evaluate_model.calibration_data, one row
    per probability decile and model: predicted_mean, actual_rate, n_in_bin.

    Read from the SAME run folder as shap_values.parquet, so every chart on
    this page describes the same trained model.
    """
    return pd.read_csv(path)


def plot_calibration(calibration: pd.DataFrame) -> None:
    """Predicted risk (x) vs observed blunder rate (y), one line per model,
    plus the y = x diagonal: on the diagonal, "7% predicted" really means
    7% of those moves were blunders (spec: "usable as a real probability").
    """
    calibration = calibration.sort_values(
        ["model", "predicted_mean"]
    )

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
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="gray", dash="dash"),
        layer="below",
    )

    axis_max = min(
        1.0,
        1.1
        * max(
            calibration["predicted_mean"].max(),
            calibration["actual_rate"].max(),
        ),
    )

    fig.update_layout(**PLOTLY_LAYOUT.to_plotly_json())

    fig.update_xaxes(
        tickformat=".0%",
        range=[0, axis_max],
        title="Mean predicted risk",
        constrain="domain",
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, axis_max],
        title="Observed blunder rate",
        scaleanchor="x",
        scaleratio=1,
    )

    st.plotly_chart(fig, use_container_width=True)


def prepare_plot_data(
    shap_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare plot features, combining one-hot columns into categorical groups.

    Each reconstructed categorical feature gets its original category as its
    value and the sum of its one-hot SHAP contributions as its attribution.
    This avoids displaying separate 0/1 columns in the feature selector.
    """

    plot_data = shap_data.copy()

    raw_features = [
        column
        for column in plot_data.columns
        if column not in {"game_id", "ply"}
        and not column.startswith("shap_")
        and f"shap_{column}" in plot_data.columns
    ]

    one_hot_columns = {
        column
        for column in raw_features
        if column.startswith(tuple(CATEGORICAL_PREFIXES))
    }

    available_features = [
        column
        for column in raw_features
        if column not in one_hot_columns
        and pd.api.types.is_numeric_dtype(plot_data[column])
    ]

    for prefix, feature_name in CATEGORICAL_PREFIXES.items():
        group_columns = sorted(
            column
            for column in raw_features
            if column.startswith(prefix)
        )

        if not group_columns:
            continue

        values = (
            plot_data[group_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        categories = values.idxmax(axis=1).str.slice(
            start=len(prefix)
        )

        categories = categories.where(
            values.max(axis=1) > 0,
            "Unknown",
        )

        shap_columns = [
            f"shap_{column}"
            for column in group_columns
            if f"shap_{column}" in plot_data.columns
        ]

        if not shap_columns:
            continue

        plot_data[feature_name] = categories
        plot_data[f"shap_{feature_name}"] = (
            plot_data[shap_columns].sum(axis=1)
        )

        if feature_name not in available_features:
            available_features.append(feature_name)

    return plot_data, available_features


def build_feature_summary(
    shap_data: pd.DataFrame,
    plot_data: pd.DataFrame,
    available_features: list[str],
    summary_path: Path,
) -> pd.DataFrame:
    """Rank available plot features using the run's SHAP summary.

    If the summary is missing or does not match the available features,
    calculate a fallback ranking from the saved SHAP values.
    """

    summary = pd.DataFrame()

    if summary_path.exists():
        summary = load_shap_summary(str(summary_path))

    if {"feature", "mean_abs_shap"}.issubset(summary.columns):
        summary = summary[
            ["feature", "mean_abs_shap"]
        ].copy()

        summary["feature"] = summary["feature"].astype(str)

        summary["mean_abs_shap"] = pd.to_numeric(
            summary["mean_abs_shap"],
            errors="coerce",
        )

        summary = summary.dropna(
            subset=["mean_abs_shap"]
        )

        summary = summary[
            summary["feature"].isin(available_features)
        ]

        summary = summary.drop_duplicates(
            subset=["feature"]
        )

    if summary.empty:
        summary_rows = []

        for feature in available_features:
            shap_column = f"shap_{feature}"

            if shap_column not in plot_data.columns:
                continue

            importance = pd.to_numeric(
                plot_data[shap_column],
                errors="coerce",
            ).abs().mean()

            if pd.notna(importance):
                summary_rows.append(
                    {
                        "feature": feature,
                        "mean_abs_shap": float(importance),
                    }
                )

        summary = pd.DataFrame(
            summary_rows,
            columns=["feature", "mean_abs_shap"],
        )

    return summary.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)


st.set_page_config(
    page_title="Chess Insights",
    layout="wide",
)

load_ui()


def main() -> None:
    st.title("V2 — Blunder Risk Model")

    st.markdown(
        """
        **Question:** which factors influence my blunder risk, and how does
        each factor affect the model's prediction?

        Each point represents a move from the test set. The chart shows
        whether a given feature value pushes the model towards predicting a
        blunder or away from it. One-hot encoded categorical columns are
        grouped by feature to make the results easier to interpret.
        """
    )

    shap_path = find_latest_shap_file()

    if shap_path is None:
        st.warning(
            "No SHAP values were found. Run "
            "`python -m src.model.evaluate_model` from the project root "
            "to generate the evaluation artifacts."
        )
        return

    shap_data = load_shap_values(str(shap_path))

    if shap_data.empty:
        st.warning("The latest SHAP file is empty.")
        return

    if not {"game_id", "ply"}.issubset(shap_data.columns):
        st.warning(
            "The SHAP file does not contain `game_id` and `ply`. "
            "Run `python -m src.model.evaluate_model` again to regenerate "
            "the file with move metadata."
        )
        return

    plot_data, available_features = prepare_plot_data(shap_data)

    if not available_features:
        st.warning(
            "No usable features were found in the SHAP file."
        )
        return

    summary_path = shap_path.parent / "shap_summary.csv"

    if not summary_path.exists():
        st.warning(
            "`shap_summary.csv` is missing. Feature rankings will be "
            "calculated directly from the available SHAP values."
        )

    feature_summary = build_feature_summary(
        shap_data,
        plot_data,
        available_features,
        summary_path,
    )

    if feature_summary.empty:
        st.warning("Unable to calculate feature importance.")
        return

    top_summary = feature_summary.head(TOP_N_FEATURES).copy()
    top_features = top_summary["feature"].tolist()

    if not top_features:
        st.warning("No important features are available for plotting.")
        return

    st.caption(
        f"Evaluation run: {shap_path.parent.name} "
        f"· {len(shap_data):,} moves analysed"
    )

    st.markdown("### Top risk factors")

    st.caption(
        "Features are ranked by mean absolute SHAP value. This measures "
        "the magnitude of their contribution, not whether they increase "
        "or decrease the predicted risk."
    )

    bar_data = top_summary.copy()
    bar_data["label"] = bar_data["feature"].map(feature_label)

    bar_fig = px.bar(
        bar_data,
        x="mean_abs_shap",
        y="label",
        orientation="h",
        labels={
            "mean_abs_shap": "Mean absolute SHAP importance",
            "label": "Feature",
        },
    )

    bar_fig.update_traces(
        marker_color="#2E7DFF",
        texttemplate="%{x:.3f}",
        textposition="outside",
        cliponaxis=False,
    )

    bar_fig.update_layout(
        **PLOTLY_LAYOUT.to_plotly_json()
    )

    bar_fig.update_layout(
        height=max(300, 42 * len(bar_data) + 70),
        showlegend=False,
        margin=dict(
            l=25,
            r=75,
            t=15,
            b=35,
        ),
        xaxis=dict(
            title="Mean absolute SHAP importance",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            categoryorder="total ascending",
            showgrid=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        bar_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    st.markdown("### SHAP dependence plot")

    col1, col2 = st.columns(2)

    default_x_index = (
        top_features.index("eval_before_move")
        if "eval_before_move" in top_features
        else 0
    )

    with col1:
        x_feature = st.selectbox(
            "Feature to examine",
            options=top_features,
            index=default_x_index,
            format_func=feature_label,
            help=(
                "Choose one of the most important features to examine "
                "how it influences the model's prediction."
            ),
        )

    color_options = [
        feature
        for feature in top_features
        if feature != x_feature
    ]

    if not color_options:
        color_options = top_features

    default_color_index = (
        color_options.index("material_imbalance")
        if "material_imbalance" in color_options
        else 0
    )

    with col2:
        color_feature = st.selectbox(
            "Colour points by",
            options=color_options,
            index=default_color_index,
            format_func=feature_label,
            help=(
                "A second feature can help reveal interactions or "
                "differences between positions."
            ),
        )

    shap_column = f"shap_{x_feature}"

    plot_data_filtered = plot_data.dropna(
        subset=[
            x_feature,
            shap_column,
            color_feature,
        ]
    ).copy()

    if plot_data_filtered.empty:
        st.warning(
            "No complete observations are available for these features."
        )
        return

    x_label = feature_label(x_feature)
    color_label = feature_label(color_feature)
    color_is_numeric = pd.api.types.is_numeric_dtype(
        plot_data_filtered[color_feature]
    )

    scatter_kwargs: dict[str, object] = {}

    if color_is_numeric:
        scatter_kwargs["color_continuous_scale"] = [
            "#2E7DFF",
            "#B8E35A",
            "#F4BE55",
        ]

        if color_feature == "material_imbalance":
            scatter_kwargs["range_color"] = (-10, 10)
        else:
            color_values = pd.to_numeric(
                plot_data_filtered[color_feature],
                errors="coerce",
            ).dropna()

            if not color_values.empty:
                lower = color_values.quantile(0.02)
                upper = color_values.quantile(0.98)

                if lower < upper:
                    scatter_kwargs["range_color"] = (
                        float(lower),
                        float(upper),
                    )
    else:
        scatter_kwargs["color_discrete_sequence"] = [
            "#2E7DFF",
            "#B8E35A",
            "#F4BE55",
            "#C084FC",
            "#F87171",
            "#38BDF8",
        ]

    fig = px.scatter(
        plot_data_filtered,
        x=x_feature,
        y=shap_column,
        color=color_feature,
        custom_data=[
            "game_id",
            "ply",
            color_feature,
        ],
        labels={
            x_feature: x_label,
            shap_column: f"SHAP contribution — {x_label}",
            color_feature: color_label,
        },
        opacity=0.75,
        **scatter_kwargs,
    )

    fig.update_traces(
        marker=dict(
            size=7,
            line=dict(
                width=0,
            ),
        ),
        hovertemplate=(
            "<b>Game:</b> %{customdata[0]}<br>"
            "<b>Half-move:</b> %{customdata[1]}<br>"
            f"<b>{x_label}:</b> %{{x}}<br>"
            "<b>SHAP contribution:</b> %{y:.4f}<br>"
            f"<b>{color_label}:</b> %{{customdata[2]}}"
            "<extra></extra>"
        ),
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1.5,
        line_color="#F4BE55",
        annotation_text="SHAP = 0",
        annotation_position="top left",
        annotation_font_color="#F4BE55",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT.to_plotly_json()
    )

    fig.update_layout(
        height=560,
        xaxis=dict(
            title=x_label,
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            title=f"SHAP contribution — {x_label}",
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False,
            showline=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#12181E",
            bordercolor="rgba(255,255,255,0.10)",
            font=dict(
                family="Inter",
                size=13,
            ),
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    st.caption(
        "Hover over a point to identify the game and half-move. "
        "The material imbalance colour scale is limited to the range "
        "−10 to +10, preventing extreme values from hiding differences "
        "between more typical positions."
    )

    st.markdown("### Calibration")

    calibration_path = shap_path.parent / "calibration.csv"
    if not calibration_path.exists():
        st.info("Calibration data are unavailable for this run; skipping the calibration plot.")
    else:
        plot_calibration(load_calibration(str(calibration_path)))
        st.caption(
            "Points above the diagonal indicate that the model under-predicts risk; "
            "points below it indicate that the model over-predicts risk."
        )

    st.markdown("### How to interpret this")

    st.markdown(
        """
        - **Each point represents one move.** Its horizontal position shows
          the feature value. Its vertical position shows that feature's
          contribution to the model's prediction.

        - **Positive SHAP:** the feature pushes the prediction towards a
          blunder. **Negative SHAP:** it pushes the prediction away from a
          blunder. For this binary LightGBM model, contributions are
          generally expressed in log-odds, not percentage-point changes
          in probability.

        - **Corrected reading rule — 2026-10-17:** on a floor-free model,
          the label alone predicts a blunder-rate curve peaking near 60%
          win probability and symmetric around 60% over the remaining
          domain [20, 100]. A 20-percentage-point win-probability drop
          corresponds to a centipawn gap that is smallest when the interval
          is centred on 50%, which occurs at a starting win probability of
          60%. This gap is symmetric for starting win probabilities
          \(w\) and \(120-w\). Therefore, a peak clearly away from 60% or
          asymmetry around 60% is a behavioural candidate, not proof of a
          behavioural effect. The centipawn-loss analysis remains the
          decisive data-only check.

        - **Look at the colours and point density.** They can reveal
          interactions with another feature and show whether a pattern
          is supported by many observations or only a few outliers.

        These rules are interpretation heuristics, not proof of causation.
        SHAP explains how the model uses its inputs; it does not prove
        that changing a feature would cause the same change in actual
        blunder risk.
        """
    )


if __name__ == "__main__":
    main()