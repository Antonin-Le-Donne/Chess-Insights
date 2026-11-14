"""One-off check: the tracking indicators of the player profile (October 2026
baseline), recomputed before vs since the start of the training plan.

Definitions reproduce the profile's numbers exactly on the "before" period
(checked 2026-11-14: 70% / 54%, 17.4% / 17.5%, 72.6% / 64.8%, 7.2% / 6.3%):

- clock at move 40 in 10+0: median of clock_after / base at move_number 40,
  one value per game and side (games that reach move 40 only); also split
  by my win% before move 40, winning (>= 80) vs balanced (40-60), in
  minutes (coach's question, 2026-11-14);
- heavy error: centipawn loss >= 200 by the mover, evals clipped to +/-1000
  (metrics.mover_cp_loss), blitz + rapid, every ply;
  win% bucket = the mover's win_pct before the move, RIGHT-CLOSED as in the
  profile: (70, 80] and (40, 50]. Note: (40, 50] includes the positions at
  exactly 50% (eval 0.00, mostly theory), which the half-open [40, 50)
  would leave out;
- conversion from >= 80%: metrics.m3_control (same rule and bootstrap);
- line 1.e4 c5 2.Nf3 e6 3.d4 d5 as Black: mean (score - Elo expected
  score), in points. chess.com ratings are post-game (PROJECT_TRACKER
  Blockers), so this carries a small bias.

Intervals: Wilson for move-level rates (moves of one game are not
independent, so these are a bit optimistic), game bootstrap for the median
clock and M3, normal approximation for the opening line.

Usage (from the repo root):
    python one_use_scripts/check_training_indicators.py
    python one_use_scripts/check_training_indicators.py --since 2026-10-20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import elo_expected_score, m3_control, mover_cp_loss  # noqa: E402

LINE_AS_BLACK = ["e4", "c5", "Nf3", "e6", "d4", "d5"]
HEAVY_ERROR_CP = 200
Z = 1.96


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return centre - half, centre + half


def bootstrap_median(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(values, size=(1000, len(values))), axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def side_label(is_my_move: bool) -> str:
    return "me" if is_my_move else "opponent"


def clock_at_move_40(moves: pd.DataFrame, games: pd.DataFrame) -> list[dict]:
    rapid_ids = games.loc[
        games["base_seconds"].eq(600) & games["increment_seconds"].eq(0), "game_id"
    ]
    at_40 = moves[moves["game_id"].isin(rapid_ids) & moves["move_number"].eq(40)]
    rows = []
    for is_mine, side_rows in at_40.groupby("is_my_move"):
        values = (side_rows["clock_after"] / 600).dropna().to_numpy()
        lower, upper = bootstrap_median(values)
        rows.append({
            "indicator": "Clock left at move 40, 10+0 (median % of base)",
            "side": side_label(is_mine),
            "value": 100 * np.median(values) if len(values) else np.nan,
            "ci_lower": 100 * lower,
            "ci_upper": 100 * upper,
            "n": len(values),
            "unit": "games",
        })
    return rows


def clock_at_move_40_by_state(moves: pd.DataFrame, games: pd.DataFrame) -> list[dict]:
    """My clock after move 40 in 10+0, split by my win% before that move:
    winning (>= 80) vs balanced (40-60). Minutes left, median per game."""
    rapid_ids = games.loc[
        games["base_seconds"].eq(600) & games["increment_seconds"].eq(0), "game_id"
    ]
    mine = moves[
        moves["game_id"].isin(rapid_ids)
        & moves["move_number"].eq(40)
        & moves["is_my_move"]
        & moves["clock_after"].notna()
    ]
    states = {
        "winning (>= 80%)": mine["win_pct"].ge(80),
        "balanced (40-60%)": mine["win_pct"].between(40, 60),
    }
    rows = []
    for state, mask in states.items():
        values = (mine.loc[mask, "clock_after"] / 60).to_numpy()
        lower, upper = bootstrap_median(values)
        rows.append({
            "indicator": "My clock left at move 40, 10+0, by position (median minutes)",
            "side": state,
            "value": np.median(values) if len(values) else np.nan,
            "ci_lower": lower,
            "ci_upper": upper,
            "n": len(values),
            "unit": "games",
        })
    return rows


def heavy_error_rate(moves: pd.DataFrame, low: float, high: float) -> list[dict]:
    bucket = moves[
        moves["win_pct"].gt(low) & moves["win_pct"].le(high) & moves["cp_loss"].notna()
    ]
    rows = []
    for is_mine, side_rows in bucket.groupby("is_my_move"):
        n = len(side_rows)
        k = int(side_rows["cp_loss"].ge(HEAVY_ERROR_CP).sum())
        lower, upper = wilson(k, n)
        rows.append({
            "indicator": f"Heavy errors at ({low:g}, {high:g}] win%, blitz+rapid",
            "side": side_label(is_mine),
            "value": 100 * k / n if n else np.nan,
            "ci_lower": 100 * lower,
            "ci_upper": 100 * upper,
            "n": n,
            "unit": "moves",
        })
    return rows


def conversion(moves: pd.DataFrame, games: pd.DataFrame) -> list[dict]:
    control = m3_control(moves, games, winning_pct=80)
    main = control[control["perf_group"].eq("main")]
    return [
        {
            "indicator": "Conversion from >= 80% win%, blitz+rapid",
            "side": row["side"],
            "value": 100 * row["conversion"],
            "ci_lower": 100 * row["ci_lower"],
            "ci_upper": 100 * row["ci_upper"],
            "n": int(row["n_games"]),
            "unit": "games",
        }
        for _, row in main.iterrows()
    ]


def opening_line(moves: pd.DataFrame, games: pd.DataFrame) -> list[dict]:
    first_plies = (
        moves[moves["ply"].le(len(LINE_AS_BLACK))]
        .sort_values(["game_id", "ply"])
        .groupby("game_id")["san"]
        .agg(list)
    )
    in_line = first_plies[first_plies.apply(lambda sans: sans == LINE_AS_BLACK)].index
    line_games = games[games["game_id"].isin(in_line) & games["my_color"].eq("black")]
    gap = 100 * (
        line_games["result_for_me"]
        - elo_expected_score(line_games["my_rating"] - line_games["opp_rating"])
    )
    n = len(gap)
    half = Z * gap.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return [{
        "indicator": "1.e4 c5 2.Nf3 e6 3.d4 d5 as Black: score - Elo expected (pts)",
        "side": "me",
        "value": gap.mean() if n else np.nan,
        "ci_lower": gap.mean() - half if n else np.nan,
        "ci_upper": gap.mean() + half if n else np.nan,
        "n": n,
        "unit": "games",
    }]


def indicators(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    main_ids = games.loc[games["perf_type"].ne("bullet"), "game_id"]
    main_moves = moves[moves["game_id"].isin(main_ids)]
    rows = (
        clock_at_move_40(moves, games)
        + clock_at_move_40_by_state(moves, games)
        + heavy_error_rate(main_moves, 70, 80)
        + conversion(moves, games)
        + heavy_error_rate(main_moves, 40, 50)
        + opening_line(moves, games)
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile indicators, before vs since.")
    parser.add_argument("--since", default="2026-10-17", help="first day of the new period")
    args = parser.parse_args()

    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    games["date_utc"] = pd.to_datetime(games["date_utc"])
    moves["cp_loss"] = mover_cp_loss(moves)

    since = pd.Timestamp(args.since)
    periods = {
        f"before {args.since}": games[games["date_utc"] < since],
        f"since {args.since}": games[games["date_utc"] >= since],
    }

    tables = []
    for period, period_games in periods.items():
        period_moves = moves[moves["game_id"].isin(period_games["game_id"])]
        table = indicators(period_moves, period_games)
        table.insert(0, "period", period)
        tables.append(table)
        print(f"{period}: {len(period_games):,} games")

    result = pd.concat(tables, ignore_index=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    for indicator, rows in result.groupby("indicator", sort=False):
        print(f"\n{indicator}")
        print(
            rows.drop(columns="indicator").to_string(
                index=False, float_format=lambda x: f"{x:.1f}"
            )
        )


if __name__ == "__main__":
    main()
