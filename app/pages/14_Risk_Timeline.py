"""Per-game risk timeline (spec V2, "Per-game risk timeline", 2026-11-14).

One test-set game at a time: the model's predicted blunder risk on each of my
moves, my actual blunders, and my win% on the same move axis. Reads
test_predictions.parquet from the latest evaluation run that has one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.filters import (
    format_time_control,
    game_url,
    link_column,
    render_filters_not_applied,
    selected_points,
)
from app.model_labels import feature_label
from app.ui import load_ui

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVALUATION_DIR = PROJECT_ROOT / "model_evaluation"
PREDICTIONS_FILE = "test_predictions.parquet"
TOP_FACTORS = 3

RESULT_LABELS = {1.0: "Win", 0.5: "Draw", 0.0: "Loss"}
RISK_COLOR = "#2E7DFF"
BLUNDER_COLOR = "#F87171"
WIN_PCT_COLOR = "#B8E35A"
REFERENCE_COLOR = "#F4BE55"


def find_latest_predictions() -> Path | None:
    """Most recent model_evaluation/<timestamp>/test_predictions.parquet.

    Run folders are named YYYY-MM-DD_HH-MM-SS, so names sort in time order.
    """
    files = sorted(
        EVALUATION_DIR.glob(f"*/{PREDICTIONS_FILE}"),
        key=lambda path: path.parent.name,
    )
    return files[-1] if files else None


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_games() -> pd.DataFrame:
    return pd.read_parquet("data/processed/games.parquet")


@st.cache_data
def load_my_moves(game_ids: tuple[str, ...]) -> pd.DataFrame:
    """My moves in the test games, with my win% before each move."""
    moves = pd.read_parquet(
        "data/processed/moves.parquet",
        columns=["game_id", "ply", "move_number", "is_my_move", "san", "win_pct", "move_quality"],
        filters=[("game_id", "in", list(game_ids))],
    )
    # On my rows, win_pct is already MY win% before my move.
    return moves[moves["is_my_move"]].drop(columns="is_my_move")


def game_table(predictions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per test game: context, result, risk summary and blunders."""
    per_game = predictions.groupby("game_id").agg(
        max_risk=("predicted_risk", "max"),
        mean_risk=("predicted_risk", "mean"),
        n_blunders=("is_blunder", "sum"),
        n_model_moves=("predicted_risk", "size"),
    )
    table = games.set_index("game_id").loc[
        per_game.index,
        ["date_utc", "platform", "perf_type", "base_seconds", "increment_seconds",
         "my_color", "result_for_me", "my_rating", "opp_rating"],
    ].join(per_game)
    table["result"] = table["result_for_me"].map(RESULT_LABELS).fillna("Unknown")
    table["time_control"] = [
        format_time_control(base, inc)
        for base, inc in zip(table["base_seconds"], table["increment_seconds"])
    ]
    return table.reset_index()


def game_option_label(row: pd.Series) -> str:
    blunders = int(row["n_blunders"])
    return (
        f"{row['date_utc']:%Y-%m-%d} · {row['platform']} {row['perf_type']} "
        f"{row['time_control']} · {row['my_color']} · {row['result']} · "
        f"max risk {row['max_risk']:.0%} · {blunders} blunder{'s' if blunders != 1 else ''}"
    )


def factor_text(row: pd.Series) -> str:
    """Hover lines for the top SHAP factors of one move."""
    lines = []
    for rank in range(1, TOP_FACTORS + 1):
        feature = row.get(f"factor_{rank}")
        if not isinstance(feature, str):
            continue
        shap_value = row[f"factor_{rank}_shap"]
        direction = "raises" if shap_value > 0 else "lowers"
        lines.append(
            f"{feature_label(feature)} = {row[f'factor_{rank}_value']} "
            f"({direction} risk, {shap_value:+.2f})"
        )
    return "<br>".join(lines)


def build_timeline(game_moves: pd.DataFrame, base_rate: float) -> go.Figure:
    """Risk panel (top) and win% panel (bottom), shared move-number axis."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Predicted blunder risk", "My win% before the move"),
    )

    # NaN risk (outside the model population) leaves a gap in the line.
    fig.add_trace(
        go.Scatter(
            x=game_moves["move_number"],
            y=game_moves["predicted_risk"],
            mode="lines+markers",
            name="Predicted risk",
            line=dict(color=RISK_COLOR, width=2),
            marker=dict(size=6),
            connectgaps=False,
            customdata=np.stack(
                [game_moves["ply"], game_moves["san"], game_moves["hover_factors"]], axis=1
            ),
            hovertemplate=(
                "<b>Move %{x}: %{customdata[1]}</b><br>"
                "Predicted risk: %{y:.1%}<br>"
                "%{customdata[2]}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    blunders = game_moves[game_moves["move_quality"].eq("blunder")]
    if not blunders.empty:
        # Blunders outside the model population have no risk: draw them at 0.
        blunder_y = blunders["predicted_risk"].fillna(0)
        fig.add_trace(
            go.Scatter(
                x=blunders["move_number"],
                y=blunder_y,
                mode="markers",
                name="My blunder",
                marker=dict(symbol="x", size=13, color=BLUNDER_COLOR, line=dict(width=1)),
                customdata=np.stack([blunders["ply"], blunders["san"]], axis=1),
                hovertemplate=(
                    "<b>Blunder — move %{x}: %{customdata[1]}</b>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_hline(
        y=base_rate,
        line_dash="dash",
        line_color=REFERENCE_COLOR,
        annotation_text=f"Test-set average {base_rate:.1%}",
        annotation_position="top left",
        annotation_font_color=REFERENCE_COLOR,
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=game_moves["move_number"],
            y=game_moves["win_pct"] / 100,
            mode="lines",
            name="My win%",
            line=dict(color=WIN_PCT_COLOR, width=2),
            customdata=game_moves[["san"]].to_numpy(),
            hovertemplate="Move %{x}: %{customdata[0]}<br>My win%: %{y:.0%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hrect(
        y0=0.70, y1=0.80, fillcolor=REFERENCE_COLOR, opacity=0.12, line_width=0,
        annotation_text="70–80% zone", annotation_position="top left",
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter", size=14, color="#DCE3E9"),
        height=640,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
        margin=dict(l=40, r=30, t=70, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title="Move number", row=2, col=1)
    return fig


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def main() -> None:
    st.title("V2 — Risk timeline of one game")
    render_filters_not_applied(
        "only the model's chronological test games can be shown (honest, "
        "out-of-sample predictions); pick one with the selectors below."
    )
    st.markdown(
        """
        **Question:** over one game, where did the model see danger coming,
        and did my blunders happen there?

        Each point is one of my moves: the model's predicted probability that
        this move is a blunder, computed only from what was known before I
        played it. Hover a point to see the three factors that weighed most.
        """
    )

    path = find_latest_predictions()
    if path is None:
        st.warning(
            "No `test_predictions.parquet` found. Run `python -m src.model.evaluate_model` "
            "(after `python -m src.model.train`) to generate it."
        )
        return

    predictions = load_predictions(str(path))
    games = load_games()
    base_rate = float(predictions["is_blunder"].mean())
    table = game_table(predictions, games)

    st.caption(
        f"Evaluation run: {path.parent.name} · {len(table):,} test games · "
        f"{len(predictions):,} moves · test-set blunder rate {base_rate:.1%}"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        perf_types = st.multiselect(
            "Time control", sorted(table["perf_type"].unique()), placeholder="All"
        )
    with col2:
        results = st.multiselect("Result", ["Win", "Draw", "Loss"], placeholder="All")
    with col3:
        sort_by = st.radio(
            "Sort games by", ["Most recent", "Highest predicted risk"], horizontal=True
        )

    shown = table
    if perf_types:
        shown = shown[shown["perf_type"].isin(perf_types)]
    if results:
        shown = shown[shown["result"].isin(results)]
    if shown.empty:
        st.info("No test game matches these selectors.")
        return
    shown = shown.sort_values(
        "date_utc" if sort_by == "Most recent" else "max_risk", ascending=False
    )

    game_id = st.selectbox(
        f"Game ({len(shown):,} available)",
        shown["game_id"].tolist(),
        format_func=lambda gid: game_option_label(shown.set_index("game_id").loc[gid]),
    )
    game = shown.set_index("game_id").loc[game_id]

    my_moves = load_my_moves(tuple(table["game_id"]))
    game_moves = (
        my_moves[my_moves["game_id"].eq(game_id)]
        .merge(
            predictions[predictions["game_id"].eq(game_id)].drop(
                columns=["move_number", "san"]
            ),
            on=["game_id", "ply"],
            how="left",
        )
        .sort_values("ply")
        .reset_index(drop=True)
    )
    game_moves["hover_factors"] = [
        factor_text(row) if pd.notna(row["predicted_risk"]) else "Outside the model population"
        for _, row in game_moves.iterrows()
    ]

    url = game_url(game_id, game["my_color"])
    st.markdown(
        f"**{game['result']}** as {game['my_color']} · {game['platform']} "
        f"{game['perf_type']} {game['time_control']} · "
        f"{int(game['my_rating'])} vs {int(game['opp_rating'])} · [open the game]({url})"
    )

    event = st.plotly_chart(
        build_timeline(game_moves, base_rate),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        on_select="rerun",
        selection_mode="points",
        key="risk_timeline_chart",
    )
    points = selected_points(event)
    if points and points[0].get("customdata"):
        ply, san = points[0]["customdata"][:2]
        # Open the board BEFORE the move: the model predicts from that position.
        move_url = game_url(game_id, game["my_color"], int(ply) - 1)
        st.markdown(f"Selected move: [{san}, half-move {int(ply)}]({move_url})")

    st.caption(
        "Gaps in the risk line are moves outside the model population (my first "
        "5 moves, positions below 20% win%, where a blunder is impossible by "
        "definition, and the last move, which has no evaluation after it). A "
        "blunder drawn at 0% is one of those moves. Click a point "
        "for a link to the position before the move (Lichess only; chess.com opens "
        "at the start)."
    )

    st.markdown("### Riskiest moves of this game")
    riskiest = game_moves.dropna(subset=["predicted_risk"]).nlargest(5, "predicted_risk")
    st.dataframe(
        pd.DataFrame(
            {
                "Move": riskiest["move_number"],
                "Played": riskiest["san"],
                "Predicted risk": riskiest["predicted_risk"].map("{:.1%}".format),
                "Blunder?": riskiest["is_blunder"].map({True: "yes", False: "no"}),
                "Main factor": riskiest["factor_1"].map(feature_label),
                "Position": [
                    game_url(game_id, game["my_color"], int(ply) - 1)
                    for ply in riskiest["ply"]
                ],
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Position": link_column("Position")},
    )

    st.markdown("### How to read this")
    st.markdown(
        """
        - **The risk is a probability, not a verdict.** A 30% move that I played
          well is not a model error: 7 times out of 10, that is exactly what
          should happen. Judge the model on many games (page *Blunder Risk
          Model*: calibration, PR-AUC), not on one.
        - **Only test games are shown.** The model never saw them during
          training, so these predictions are honest. Training games would look
          better than the model really is.
        - **SHAP factors** explain the model's reasoning for that move, in
          log-odds: "raises risk" pushes the prediction up. They describe how
          the model uses its inputs, not what caused the blunder.
        - **No control group here:** this page is a view of the model, not a
          metric about my play.
        """
    )


if __name__ == "__main__":
    main()
