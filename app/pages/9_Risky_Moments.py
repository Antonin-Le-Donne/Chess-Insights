"""V2 risk page (spec V2, "Per-game risk timeline" + "Risk page layout",
revised 2026-11-15).

Question: does the model see my blunders coming, and what do those moments
look like? Test games only (out-of-sample predictions), read from
test_predictions.parquet of the latest evaluation run.

Layout, following the dashboard principles (spec section 6): the answer
first (blunder rate by risk band), then the riskiest moments, then the
selected moment on a board, then a replay of that game.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import chess.svg
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.filters import game_url, render_filters_not_applied
from app.risk_bands import add_risk_band, band_rates, find_latest_predictions
from app.ui import load_ui

TOP_FACTORS = 3
LIST_SIZE = 100

ACCENT = "#C7F36B"
INFO = "#7CC4FF"
DANGER = "#FF6B6B"
MUTED = "#8F9893"
TEXT = "#F4F2EA"

BOARD_COLORS = {
    "square light": "#E7E2D2",
    "square dark": "#7A8C6E",
    "square light lastmove": "#D9E89A",
    "square dark lastmove": "#9DB858",
    "margin": "#12171C",
    "coord": "#D2D6D0",
}

PLOT_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Inter", size=14, color=TEXT),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


# ---------- data ----------

@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    """Test predictions joined with game context, with a risk band per move."""
    predictions = pd.read_parquet(path)
    games = pd.read_parquet(
        "data/processed/games.parquet",
        columns=["game_id", "date_utc", "platform", "perf_type", "my_color", "base_seconds"],
    )
    predictions = predictions.merge(games, on="game_id", how="left")
    predictions["band"] = add_risk_band(predictions)
    return predictions


@st.cache_data
def load_test_game_moves(game_ids: tuple[str, ...]) -> pd.DataFrame:
    """Every ply of the test games, to replay them on a board."""
    return pd.read_parquet(
        "data/processed/moves.parquet",
        columns=["game_id", "ply", "move_number", "is_my_move", "san", "uci", "move_quality"],
        filters=[("game_id", "in", list(game_ids))],
    ).sort_values(["game_id", "ply"])


# ---------- plain-language factors ----------

def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def factor_sentence(feature: str, value: object) -> str:
    """One model factor as a short sentence a non-specialist can read."""
    v = _number(value)
    text = str(value)
    if feature == "eval_before_move" and v is not None:
        return f"I was at {v:.0f}% win probability"
    if feature == "eval_volatility" and v is not None:
        return f"the evaluation had been swinging (spread of {v:.0f} win% points over the last moves)"
    if feature == "clock_before_move_pct" and v is not None:
        return f"{v:.0%} of my starting time left"
    if feature == "clock_before_move" and v is not None:
        return f"{v:.0f} s left on my clock"
    if feature == "material_imbalance" and v is not None:
        return f"material balance {v:+.0f} (pawn = 1)"
    if feature == "material_count" and v is not None:
        return f"{v:.0f} points of material still on the board"
    if feature == "queens_on_board":
        return "queens still on the board" if text == "yes" else "queens already traded"
    if feature == "n_legal_moves" and v is not None:
        return f"{v:.0f} legal moves to choose from"
    if feature == "hanging_pieces" and v is not None:
        return f"{v:.0f} of my pieces attacked more than defended"
    if feature == "checks_available" and v is not None:
        return f"{v:.0f} checks available"
    if feature == "rolling_time_mean_3" and v is not None:
        return f"about {v:.0f} s per move over my last 3 moves"
    if feature == "previous_time_spent" and v is not None:
        return f"{v:.0f} s spent on my previous move"
    if feature == "previous_long_think":
        return "I had just made a very long think" if v == 1 else "no long think just before"
    if feature == "time_pressure_before_move":
        return "already in time trouble" if text == "yes" else "not in time trouble"
    if feature == "move_number" and v is not None:
        return f"move {v:.0f}"
    if feature == "has_increment":
        return "increment game" if text == "yes" else "no increment"
    if feature == "phase":
        return f"{text} phase"
    if feature == "perf_type":
        return f"{text} game"
    if feature == "opening_family":
        return f"{text} opening" if text != "n/a" else "opening outside the main families"
    if feature == "eval_state_before_move":
        return f"position {text} for me"
    return f"{feature.replace('_', ' ')} = {text}"


def factor_lines(row: pd.Series) -> list[tuple[str, bool]]:
    """Top factors of one move: (sentence, raises_risk)."""
    lines = []
    for rank in range(1, TOP_FACTORS + 1):
        feature = row.get(f"factor_{rank}")
        if isinstance(feature, str):
            lines.append(
                (factor_sentence(feature, row[f"factor_{rank}_value"]), row[f"factor_{rank}_shap"] > 0)
            )
    return lines


def main_reason(row: pd.Series) -> str:
    sentence, raises = factor_lines(row)[0]
    return ("↑ " if raises else "↓ ") + sentence


# ---------- board ----------

def board_after(ucis: list[str], n_plies: int) -> chess.Board:
    """Position after the first n_plies half-moves of the game."""
    board = chess.Board()
    for uci in ucis[:n_plies]:
        board.push(chess.Move.from_uci(uci))
    return board


def render_board(
    board: chess.Board,
    my_color: str,
    arrow_uci: str | None = None,
    arrow_is_blunder: bool = False,
) -> None:
    """Draw the board as an SVG image, oriented to my color."""
    arrows = []
    if arrow_uci:
        move = chess.Move.from_uci(arrow_uci)
        color = "#FF6B6Bcc" if arrow_is_blunder else "#C7F36Bcc"
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=color))
    svg = chess.svg.board(
        board,
        orientation=chess.WHITE if my_color == "white" else chess.BLACK,
        lastmove=board.peek() if board.move_stack and not arrow_uci else None,
        arrows=arrows,
        colors=BOARD_COLORS,
        size=440,
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{encoded}" '
        'style="width:100%;max-width:440px;border-radius:12px;" alt="Chess board">',
        unsafe_allow_html=True,
    )


def move_label(move_number: int, san: str, is_white: bool) -> str:
    return f"{move_number}. {san}" if is_white else f"{move_number}… {san}"


# ---------- sections ----------

def render_headline(predictions: pd.DataFrame) -> None:
    """The answer first: blunder rate per risk band."""
    by_band = band_rates(predictions)
    low, high = by_band.loc["Low risk", "rate"], by_band.loc["High risk", "rate"]
    share_caught = by_band.loc["High risk", "blunders"] / by_band["blunders"].sum()
    base_rate = predictions["is_blunder"].mean()

    st.markdown(
        f"#### When the model rates a move as **high risk**, I blunder **{high:.0%}** of the "
        f"time, against **{low:.1%}** on low-risk moves. **{share_caught:.0%}** of my "
        "blunders happened on the moves it flagged as high risk."
    )

    fig = go.Figure(
        go.Bar(
            x=["Low risk<br>(bottom half)", "Medium risk", "High risk<br>(top 10%)"],
            y=by_band["rate"],
            marker_color=[INFO, MUTED, DANGER],
            text=[f"{rate:.1%}" for rate in by_band["rate"]],
            textposition="outside",
            customdata=by_band["n"],
            hovertemplate="%{y:.1%} of %{customdata:,} moves were blunders<extra></extra>",
        )
    )
    fig.add_hline(
        y=base_rate,
        line_dash="dot",
        line_color=MUTED,
        annotation_text=f"average {base_rate:.1%}",
        annotation_position="top left",
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(tickformat=".0%", title="Blunder rate", gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_moments_list(predictions: pd.DataFrame) -> pd.Series | None:
    """The riskiest moments, filterable; returns the selected one."""
    col1, col2 = st.columns([1, 1])
    with col1:
        show = st.radio(
            "Show",
            ["All moments", "Blunders only", "No blunder"],
            horizontal=True,
            key="risk_show",
        )
    with col2:
        perf_types = st.multiselect(
            "Time control",
            sorted(predictions["perf_type"].dropna().unique()),
            placeholder="All",
            key="risk_perf_types",
        )

    shown = predictions
    if show == "Blunders only":
        shown = shown[shown["is_blunder"]]
    elif show == "No blunder":
        shown = shown[~shown["is_blunder"]]
    if perf_types:
        shown = shown[shown["perf_type"].isin(perf_types)]
    if shown.empty:
        st.info("No moment matches these filters.")
        return None
    shown = shown.nlargest(LIST_SIZE, "predicted_risk").reset_index(drop=True)

    table = pd.DataFrame(
        {
            "Date": shown["date_utc"].dt.strftime("%Y-%m-%d"),
            "Time control": shown["perf_type"],
            "Move": [
                move_label(n, san, color == "white")
                for n, san, color in zip(shown["move_number"], shown["san"], shown["my_color"])
            ],
            "Risk": shown["predicted_risk"].map("{:.0%}".format),
            "Blunder": shown["is_blunder"].map({True: "✗ blunder", False: "—"}),
            "Main reason": [main_reason(row) for _, row in shown.iterrows()],
        }
    )
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=280,
        on_select="rerun",
        selection_mode="single-row",
        key="risk_moments_table",
    )
    rows = event.selection.rows if event and event.selection else []
    st.caption(
        f"The {min(LIST_SIZE, len(shown))} riskiest of my moves in the test games. "
        "Click a row to see the position. ↑ raised the predicted risk, ↓ lowered it."
    )
    return shown.iloc[rows[0] if rows else 0]


def render_moment(moment: pd.Series, game_moves: pd.DataFrame) -> None:
    """The selected moment on a board, with the reasons in plain words."""
    ucis = game_moves["uci"].tolist()
    ply = int(moment["ply"])
    is_white = moment["my_color"] == "white"
    blunder = bool(moment["is_blunder"])

    col_board, col_text = st.columns([5, 6], gap="large")
    with col_board:
        render_board(
            board_after(ucis, ply - 1),
            moment["my_color"],
            arrow_uci=ucis[ply - 1],
            arrow_is_blunder=blunder,
        )
    with col_text:
        verdict = (
            f"<span style='color:{DANGER}'>blunder</span>"
            if blunder
            else f"<span style='color:{ACCENT}'>no blunder</span>"
        )
        st.markdown(
            f"### {move_label(int(moment['move_number']), moment['san'], is_white)} — {verdict}",
            unsafe_allow_html=True,
        )
        clock = moment["clock_before_move"]
        base = moment["base_seconds"]
        clock_text = (
            f"{clock:.0f} s left ({clock / base:.0%} of my time)"
            if pd.notna(clock) and pd.notna(base) and base > 0
            else "unknown"
        )
        st.markdown(
            f"- **Predicted risk:** {moment['predicted_risk']:.0%} "
            f"(average move: {st.session_state['risk_base_rate']:.1%})\n"
            f"- **My win probability before the move:** {moment['eval_before_move']:.0f}%\n"
            f"- **Clock:** {clock_text}"
        )
        st.markdown("**What made the model worried — or calm:**")
        st.markdown(
            "\n".join(
                f"- {'↑' if raises else '↓'} {sentence} "
                f"({'raises' if raises else 'lowers'} the risk)"
                for sentence, raises in factor_lines(moment)
            )
        )
        url = game_url(moment["game_id"], moment["my_color"], ply - 1)
        st.markdown(f"[Open this position in the game]({url}) · {moment['date_utc']:%Y-%m-%d}")


def _set_replay(value: int) -> None:
    st.session_state["replay_ply"] = value


def _jump_from_strip() -> None:
    """Click on the risk strip: show the position BEFORE that move."""
    state = st.session_state.get("risk_strip")
    try:
        points = state["selection"]["points"]
    except (KeyError, TypeError):
        return
    if points and points[0].get("customdata"):
        st.session_state["replay_ply"] = max(int(points[0]["customdata"][0]) - 1, 0)


def render_replay(moment: pd.Series, game_moves: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Replay the selected game, with a single risk strip under the board."""
    game_id = moment["game_id"]
    ucis = game_moves["uci"].tolist()
    n_plies = len(ucis)

    # A new moment resets the replay to the position before that move.
    moment_key = (game_id, int(moment["ply"]))
    if st.session_state.get("replay_moment") != moment_key:
        st.session_state["replay_moment"] = moment_key
        st.session_state["replay_ply"] = int(moment["ply"]) - 1
    st.session_state["replay_ply"] = min(max(st.session_state["replay_ply"], 0), n_plies)
    current = st.session_state["replay_ply"]

    b1, b2, b3, b4, _ = st.columns([1, 1, 1, 1, 6])
    b1.button("⏮", on_click=_set_replay, args=(0,), help="Start", width="stretch")
    b2.button("◀", on_click=_set_replay, args=(max(current - 1, 0),), help="Previous half-move", width="stretch")
    b3.button("▶", on_click=_set_replay, args=(min(current + 1, n_plies),), help="Next half-move", width="stretch")
    b4.button("⏭", on_click=_set_replay, args=(n_plies,), help="End", width="stretch")
    st.slider("Half-move", 0, n_plies, key="replay_ply", label_visibility="collapsed")
    current = st.session_state["replay_ply"]

    game_predictions = predictions[predictions["game_id"].eq(game_id)].set_index("ply")
    my_moves = game_moves[game_moves["is_my_move"]]

    col_board, col_strip = st.columns([5, 6], gap="large")
    with col_board:
        render_board(board_after(ucis, current), moment["my_color"])
    with col_strip:
        if current == 0:
            st.markdown("**Starting position.**")
        else:
            last = game_moves.iloc[current - 1]
            label = move_label(int(last["move_number"]), last["san"], int(last["ply"]) % 2 == 1)
            who = "my move" if last["is_my_move"] else "opponent's move"
            details = ""
            if last["is_my_move"] and int(last["ply"]) in game_predictions.index:
                details = f" · predicted risk {game_predictions.loc[int(last['ply']), 'predicted_risk']:.0%}"
            flag = ""
            if last["move_quality"] == "blunder":
                flag = " · **my blunder**" if last["is_my_move"] else " · opponent's blunder"
            st.markdown(f"**After {label}** ({who}){details}{flag}")

        risk = my_moves["ply"].map(game_predictions["predicted_risk"])
        is_blunder = my_moves["move_quality"].eq("blunder")
        colors = [DANGER if b else INFO for b in is_blunder]
        fig = go.Figure(
            go.Bar(
                x=my_moves["move_number"],
                y=risk.fillna(0),
                marker_color=colors,
                customdata=list(zip(my_moves["ply"], my_moves["san"])),
                hovertemplate=(
                    "Move %{x}: %{customdata[1]}<br>Predicted risk %{y:.0%}<extra></extra>"
                ),
            )
        )
        fig.add_hline(
            y=st.session_state["risk_base_rate"], line_dash="dot", line_color=MUTED,
        )
        if current > 0:
            fig.add_vline(
                x=int(game_moves.iloc[current - 1]["move_number"]),
                line_color=ACCENT,
                line_width=2,
            )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Move"),
            yaxis=dict(tickformat=".0%", title="Predicted risk", gridcolor="rgba(255,255,255,0.07)"),
            showlegend=False,
            bargap=0.15,
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False},
            on_select=_jump_from_strip,
            selection_mode="points",
            key="risk_strip",
        )
        st.caption(
            "One bar per move of mine: the model's predicted risk (red = an actual blunder; "
            "no bar = a move the model does not score: my first 5 moves, or a position "
            "already below 20% where a blunder is impossible by definition). "
            "Click a bar to see the position before that move."
        )


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def main() -> None:
    st.title("V2 — My risky moments")
    render_filters_not_applied(
        "only the model's chronological test games can be shown (honest, "
        "out-of-sample predictions)."
    )
    st.markdown("**Question:** does the model see my blunders coming, and what do those moments look like?")

    path = find_latest_predictions()
    if path is None:
        st.warning(
            "No `test_predictions.parquet` found. Run `python -m src.model.train`, then "
            "`python -m src.model.evaluate_model`."
        )
        return

    predictions = load_predictions(str(path))
    st.session_state["risk_base_rate"] = float(predictions["is_blunder"].mean())

    render_headline(predictions)

    st.markdown("### The riskiest moments")
    moment = render_moments_list(predictions)
    if moment is None:
        return

    game_moves_all = load_test_game_moves(tuple(predictions["game_id"].unique()))
    game_moves = game_moves_all[game_moves_all["game_id"].eq(moment["game_id"])].reset_index(drop=True)

    render_moment(moment, game_moves)

    st.markdown("### Replay this game")
    render_replay(moment, game_moves, predictions)

    with st.expander("How this is computed"):
        st.markdown(
            f"""
            - **Test games only** ({predictions['game_id'].nunique():,} games,
              {len(predictions):,} of my moves, evaluation run `{path.parent.name}`). The model
              never saw them while training, so these predictions are honest.
            - **Predicted risk** = the LightGBM model's probability that my move is a blunder
              (a drop of 20 win% points or more), computed only from what was known before
              I played it.
            - **Risk bands** are fixed on the predicted risk of all test moves: low = the bottom
              half, high = the top 10%, medium = in between.
            - **Reasons** are the three largest SHAP contributions of that move, translated
              into words: they explain how the model used its inputs, not what caused the
              blunder.
            - **A probability, not a verdict:** a 30% move played well is not a model error.
              Judge the model on many moves (page *Blunder Risk Model*), not on one.
            - **No control group here:** this page describes the model, not my play compared
              with my opponents.
            """
        )


if __name__ == "__main__":
    main()
