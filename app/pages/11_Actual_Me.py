"""Actual me — how much have I progressed, and do my findings still describe
me today? (redesigned 2026-11-15 as a progress page: rating tiles, rating
and quality-of-play curves, then full history vs current level.)
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.controls import BADGE_GREEN, GROUP_LABELS, render_badge
from app.filters import render_filters_not_applied
from app.ui import load_ui
from src.metrics import m2_time_to_blunder, m3_greed_conversion
from src.scope import (
    filter_current_regime,
    m1_peak_by_move,
    per_game_quality,
    performance_rating_estimate,
    quality_of_play_trajectory,
    rating_trajectory,
)

MIN_MOVES_PER_BUCKET = 100  # same reliability rule as the M1 page
MIN_GAMES_PER_MONTH = 20  # a month's mean rating needs this many games to count
MIN_GAMES_PER_TILE = 100  # time controls with fewer games get no tile
QUALITY_WINDOW_MONTHS = 6
WEAK_FIT_R2 = 0.10
PERF_COLORS = {"bullet": "#7CC4FF", "blitz": "#C7F36B", "rapid": "#F6C85F", "classical": "#A992FF"}
PLATFORM_LABELS = {"lichess": "Lichess", "chesscom": "Chess.com"}

PLOT_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Inter", size=14),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    with open("config.toml", "rb") as f:
        floors = tomllib.load(f)["current_regime"]
    return games, moves, floors


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


# ---------- progress ----------

def rating_progress(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Per (platform, perf_type): first and latest monthly mean rating, using
    only months with enough games."""
    rows = []
    for (platform, perf_type), group in trajectory.groupby(["platform", "perf_type"]):
        if group["n"].sum() < MIN_GAMES_PER_TILE:
            continue
        solid = group[group["n"] >= MIN_GAMES_PER_MONTH].sort_values("date_utc")
        if len(solid) < 2:
            continue
        first, last = solid.iloc[0], solid.iloc[-1]
        rows.append({
            "platform": platform,
            "perf_type": perf_type,
            "first_month": first["date_utc"],
            "first_rating": first["mean_rating"],
            "last_month": last["date_utc"],
            "last_rating": last["mean_rating"],
            "n_games": int(group["n"].sum()),
        })
    return pd.DataFrame(rows).sort_values("n_games", ascending=False) if rows else pd.DataFrame()


def quality_progress(quality: pd.DataFrame) -> tuple[float, float]:
    """Win% lost per move: first vs last QUALITY_WINDOW_MONTHS months (move-weighted)."""
    quality = quality[quality["n"] > 0].sort_values("date_utc")

    def weighted(table: pd.DataFrame) -> float:
        return float((table["mean_win_pct_lost"] * table["n"]).sum() / table["n"].sum())

    return weighted(quality.head(QUALITY_WINDOW_MONTHS)), weighted(quality.tail(QUALITY_WINDOW_MONTHS))


def render_answer(progress: pd.DataFrame, quality_start: float, quality_now: float) -> None:
    render_badge("My progress", BADGE_GREEN)
    if progress.empty:
        st.markdown("#### Not enough games per month to measure a rating trend.")
        return
    main = progress.iloc[0]
    improvement = 1 - quality_now / quality_start if quality_start else float("nan")
    st.markdown(
        f"#### In my most played time control ({PLATFORM_LABELS[main['platform']]} "
        f"{main['perf_type']}), my rating went from **{main['first_rating']:.0f}** "
        f"({main['first_month']:%B %Y}) to **{main['last_rating']:.0f}** "
        f"({main['last_month']:%B %Y}): **{main['last_rating'] - main['first_rating']:+.0f}**. "
        f"Per move, I now give away **{quality_now:.2f}%** of win probability, against "
        f"**{quality_start:.2f}%** in my first months: **{improvement:.0%} less**."
    )


def render_tiles(progress: pd.DataFrame, floors: dict) -> None:
    columns = st.columns(len(progress))
    for column, (_, row) in zip(columns, progress.iterrows()):
        floor = floors.get(f"{row['platform']}_{row['perf_type']}")
        help_text = (
            f"Monthly mean rating, {row['first_month']:%B %Y} → {row['last_month']:%B %Y} "
            f"(months with {MIN_GAMES_PER_MONTH}+ games). {row['n_games']:,} games."
            + (f" Current-level floor: {floor}." if floor else " No current-level floor: not stable yet.")
        )
        with column, st.container(border=True):
            st.metric(
                f"{PLATFORM_LABELS[row['platform']]} {row['perf_type']}",
                f"{row['last_rating']:.0f}",
                delta=f"{row['last_rating'] - row['first_rating']:+.0f} since {row['first_month']:%b %Y}",
                help=help_text,
            )


def render_rating_trajectory(trajectory: pd.DataFrame, floors: dict) -> None:
    platforms = [p for p in ("lichess", "chesscom") if trajectory["platform"].eq(p).any()]
    for tab, platform in zip(st.tabs([PLATFORM_LABELS[p] for p in platforms]), platforms):
        with tab:
            data = trajectory[trajectory["platform"].eq(platform)].sort_values("date_utc")
            fig = px.line(
                data, x="date_utc", y="mean_rating", color="perf_type",
                color_discrete_map=PERF_COLORS, custom_data=["n"],
                labels={"date_utc": "", "mean_rating": "Monthly mean rating", "perf_type": ""},
            )
            fig.update_traces(
                line=dict(width=3),
                hovertemplate="%{x|%B %Y}: %{y:.0f} (%{customdata[0]} games)<extra></extra>",
            )
            for key, floor in floors.items():
                floor_platform, perf_type = key.split("_", 1)
                if floor_platform == platform:
                    fig.add_hline(
                        y=floor, line_dash="dot", line_color=PERF_COLORS.get(perf_type, "#8F9893"),
                        opacity=0.5, annotation_text=f"{perf_type} current-level floor",
                        annotation_position="bottom right",
                    )
            fig.update_layout(
                **PLOT_LAYOUT, height=360, margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", y=1.08, x=0),
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_quality(quality: pd.DataFrame) -> None:
    fig = px.line(
        quality, x="date_utc", y="mean_win_pct_lost", custom_data=["n"],
        labels={"date_utc": "", "mean_win_pct_lost": "Win% given away per move"},
    )
    fig.update_traces(
        line=dict(width=3, color="#C7F36B"),
        hovertemplate="%{x|%B %Y}: %{y:.2f}% (%{customdata[0]:,} moves)<extra></extra>",
    )
    fig.update_layout(**PLOT_LAYOUT, height=300, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(ticksuffix="%", gridcolor="rgba(255,255,255,0.07)"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Lower is better. Unlike ratings, this is on the same scale for both platforms, so it "
        "is the one curve that shows my whole progress in a single line."
    )


# ---------- does it change my findings? ----------

def peak_rate(moves: pd.DataFrame, games: pd.DataFrame, group: str) -> float:
    """Highest M1 blunder rate among 5-move buckets with enough of my moves."""
    curve = m1_peak_by_move(moves, games).reset_index()
    mine = moves[moves["is_my_move"]]
    is_bullet = mine["game_id"].map(games.set_index("game_id")["perf_type"]).eq("bullet")
    counts = (
        mine.assign(
            perf_group=is_bullet.map({True: "bullet", False: "main"}),
            move_bucket=(mine["move_number"] // 5) * 5,
        )
        .groupby(["perf_group", "move_bucket"]).size().rename("n").reset_index()
    )
    curve = curve.merge(counts, on=["perf_group", "move_bucket"], how="left")
    reliable = curve[curve["perf_group"].eq(group) & curve["n"].ge(MIN_MOVES_PER_BUCKET)]
    return float(reliable["blunder_rate"].max()) if not reliable.empty else float("nan")


def conversion(m3: pd.DataFrame, group: str) -> float:
    rows = m3[m3["perf_type"].eq("bullet") == (group == "bullet")]
    return float(rows["did_convert"].mean()) if not rows.empty else float("nan")


@st.cache_data(show_spinner="Comparing my whole history with my current level…")
def comparison(_games: pd.DataFrame, _moves: pd.DataFrame, _current: tuple, floors_key: str) -> pd.DataFrame:
    """M1 peak, M2 multiplier and M3 conversion, full history vs current level."""
    games_current, moves_current = _current
    m2_full, m2_current = m2_time_to_blunder(_moves, _games), m2_time_to_blunder(moves_current, games_current)
    m3_full, m3_current = m3_greed_conversion(_moves, _games), m3_greed_conversion(moves_current, games_current)
    rows = []
    for group in ("main", "bullet"):
        if group not in m2_current or group not in m2_full:
            continue
        rows.append({
            "group": group,
            "peak_full": peak_rate(_moves, _games, group),
            "peak_current": peak_rate(moves_current, games_current, group),
            "m2_full": m2_full[group]["ratio"],
            "m2_current": m2_current[group]["ratio"],
            "m3_full": conversion(m3_full, group),
            "m3_current": conversion(m3_current, group),
        })
    return pd.DataFrame(rows)


def render_findings_check(table: pd.DataFrame, n_current: int, n_all: int) -> None:
    st.markdown("### Do my findings still describe me today?")
    group = st.radio(
        "Time control", table["group"].tolist(), format_func=lambda g: GROUP_LABELS.get(g, g),
        horizontal=True, key="actual_me_group", label_visibility="collapsed",
    )
    row = table.set_index("group").loc[group]
    st.dataframe(
        pd.DataFrame(
            [
                ("Peak blunder rate (M1)", f"{row['peak_full']:.1%}", f"{row['peak_current']:.1%}",
                 f"{100 * (row['peak_current'] - row['peak_full']):+.1f} pts"),
                ("Conversion from 80% (M3)", f"{row['m3_full']:.1%}", f"{row['m3_current']:.1%}",
                 f"{100 * (row['m3_current'] - row['m3_full']):+.1f} pts"),
                ("Blunder multiplier after a long think (M2)", f"{row['m2_full']:.2f}×",
                 f"{row['m2_current']:.2f}×", f"{row['m2_current'] - row['m2_full']:+.2f}"),
            ],
            columns=["", "Whole history", f"Current level ({n_current:,} games)", "Today − history"],
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Current level = games above my rating floors, {n_current:,} of my {n_all:,} games. "
        "Close values mean the findings computed on my whole history also describe me today, "
        "even though my rating has moved a lot."
    )


def render_performance_estimate(games: pd.DataFrame, moves: pd.DataFrame, current: tuple, floors: dict) -> None:
    quality_full = per_game_quality(moves, games)
    quality_current = per_game_quality(current[1], current[0])
    latest = rating_trajectory(games).sort_values("date_utc").groupby(["platform", "perf_type"]).last()
    rows = []
    for key in floors:
        platform, perf_type = key.split("_", 1)
        group = quality_current[quality_current["platform"].eq(platform) & quality_current["perf_type"].eq(perf_type)]
        if group.empty:
            continue
        level = group["quality_score"].mean()
        full = performance_rating_estimate(quality_full, platform, perf_type, level)
        cur = performance_rating_estimate(quality_current, platform, perf_type, level)
        actual = latest["mean_rating"].get((platform, perf_type), float("nan"))
        rows.append({
            "Group": f"{PLATFORM_LABELS.get(platform, platform)} {perf_type}",
            "Actual rating (last month)": f"{actual:.0f}",
            "Implied by my play — whole-history fit": f"{full['predicted_rating']:.0f} (R² {full['r2']:.2f})",
            "Implied by my play — current-level fit": f"{cur['predicted_rating']:.0f} (R² {cur['r2']:.2f})",
        })
    if not rows:
        st.info("No current-level game for any configured floor.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Fits my rating against my quality of play, per platform and time control, then reads "
        f"the rating my current quality implies. With R² below {WEAK_FIT_R2:.2f}, quality "
        "explains little of the rating: compare the numbers, do not treat them as estimates."
    )


def main() -> None:
    st.title("Actual me — how much have I progressed?")
    render_filters_not_applied("it follows my whole history and my current level, which has its own definition.")
    st.markdown("**Question:** how much have I progressed — and do my findings still describe me today?")

    games, moves, floors = load_data()
    current = filter_current_regime(games, moves, floors)
    trajectory = rating_trajectory(games)
    progress = rating_progress(trajectory)
    quality = quality_of_play_trajectory(moves, games)
    quality_start, quality_now = quality_progress(quality)

    render_answer(progress, quality_start, quality_now)
    if not progress.empty:
        render_tiles(progress, floors)
    render_rating_trajectory(trajectory, floors)
    st.markdown("### Quality of play, both platforms on one scale")
    render_quality(quality)

    table = comparison(games, moves, current, repr(sorted(floors.items())))
    if not table.empty:
        render_findings_check(table, len(current[0]), len(games))

    with st.expander("What rating does my play imply?"):
        render_performance_estimate(games, moves, current, floors)
    with st.expander("How this is computed"):
        st.markdown(
            f"""
            - **Rating progress:** monthly mean rating per platform and time control; the first
              and latest months count only with at least {MIN_GAMES_PER_MONTH} games, and a
              time control needs {MIN_GAMES_PER_TILE}+ games to get a tile. Lichess and
              chess.com ratings are not on the same scale, and chess.com records them after
              the game.
            - **Quality of play:** the win probability I give away per move on average,
              comparing my first and my last {QUALITY_WINDOW_MONTHS} months (weighted by moves).
            - **Current level:** games at or above a rating floor per platform and time control
              (`config.toml`, `[current_regime]`), where each rating curve flattens; also the V2
              model's training scope.
            - **Findings check:** the M1, M2 and M3 definitions, unchanged, computed on each
              population; the peak ignores 5-move groups with fewer than
              {MIN_MOVES_PER_BUCKET} of my moves (same rule as the M1 page).
            """
        )


if __name__ == "__main__":
    main()
