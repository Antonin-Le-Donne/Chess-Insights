"""One-off check: what does my clock look like, per time control?

Input for the time-pressure threshold decision (PROJECT_TRACKER Blockers,
2026-11-14): spec says 20% of base time, config.toml says 10%, and the M4
onset in metrics.py uses a hard-coded 36 s on a median that mixes base times.

Prints, per time control (base + increment, platform-independent):
  1. clock_after percentiles at a few move numbers, with the number of games
     still running at that move (survivorship: long games only);
  2. the clock that 10% and 20% of base time correspond to, and the share of
     moves below each one, next to fixed absolute thresholds;
  3. per game: the share of games that ever go below each threshold, and the
     median move number at which they first do.

Descriptive only: no definition is changed here.

Usage (from the repo root):
    python one_use_scripts/check_clock_distribution.py
    python one_use_scripts/check_clock_distribution.py --side opponent --min-games 50
"""

from __future__ import annotations

import argparse

import pandas as pd

MOVE_NUMBERS = [20, 40, 60, 80]
PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
PCT_THRESHOLDS = [0.10, 0.20]
ABSOLUTE_THRESHOLDS = [10, 20, 30]


def load_moves(side: str) -> pd.DataFrame:
    games = pd.read_parquet(
        "data/processed/games.parquet",
        columns=["game_id", "base_seconds", "increment_seconds", "perf_type"],
    )
    moves = pd.read_parquet(
        "data/processed/moves.parquet",
        columns=["game_id", "ply", "is_my_move", "clock_after"],
    )
    if side == "me":
        moves = moves[moves["is_my_move"]]
    elif side == "opponent":
        moves = moves[~moves["is_my_move"]]

    moves = moves.merge(games, on="game_id", how="inner")
    moves["move_number"] = (moves["ply"] + 1) // 2
    # chess.com writes "180", Lichess "180+0": rebuild one label from the numbers.
    moves["tc"] = (
        moves["base_seconds"].astype(int).astype(str)
        + "+"
        + moves["increment_seconds"].astype(int).astype(str)
    )
    return moves.dropna(subset=["clock_after"])


def frequent_time_controls(moves: pd.DataFrame, min_games: int) -> list[str]:
    n_games = moves.groupby("tc")["game_id"].nunique().sort_values(ascending=False)
    return n_games[n_games >= min_games].index.tolist()


def percentiles_by_move(moves: pd.DataFrame, tcs: list[str]) -> pd.DataFrame:
    rows = []
    for tc in tcs:
        tc_moves = moves[moves["tc"] == tc]
        for move_number in MOVE_NUMBERS:
            at_move = tc_moves.loc[tc_moves["move_number"] == move_number, "clock_after"]
            if at_move.empty:
                continue
            row = {"tc": tc, "move": move_number, "games_still_running": len(at_move)}
            row.update(
                {f"p{int(q * 100)}": at_move.quantile(q) for q in PERCENTILES}
            )
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_columns(base_seconds: pd.Series) -> dict[str, pd.Series]:
    thresholds = {f"{int(p * 100)}% base": p * base_seconds for p in PCT_THRESHOLDS}
    for seconds in ABSOLUTE_THRESHOLDS:
        thresholds[f"{seconds} s"] = pd.Series(float(seconds), index=base_seconds.index)
    return thresholds


def share_of_moves_below(moves: pd.DataFrame, tcs: list[str]) -> pd.DataFrame:
    rows = []
    for tc in tcs:
        tc_moves = moves[moves["tc"] == tc]
        base = tc_moves["base_seconds"].iloc[0]
        row = {
            "tc": tc,
            "10% base = s": 0.10 * base,
            "20% base = s": 0.20 * base,
            "n_moves": len(tc_moves),
        }
        for name, threshold in threshold_columns(tc_moves["base_seconds"]).items():
            row[f"below {name}"] = (tc_moves["clock_after"] < threshold).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def per_game_onset(moves: pd.DataFrame, tcs: list[str]) -> pd.DataFrame:
    rows = []
    for tc in tcs:
        tc_moves = moves[moves["tc"] == tc]
        n_games = tc_moves["game_id"].nunique()
        row = {"tc": tc, "n_games": n_games}
        for name, threshold in threshold_columns(tc_moves["base_seconds"]).items():
            below = tc_moves[tc_moves["clock_after"] < threshold]
            first_move = below.groupby("game_id")["move_number"].min()
            row[f"games ever below {name}"] = len(first_move) / n_games
            row[f"median onset move ({name})"] = (
                first_move.median() if not first_move.empty else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=["me", "opponent", "both"], default="me")
    parser.add_argument("--min-games", type=int, default=100)
    args = parser.parse_args()

    moves = load_moves(args.side)
    tcs = frequent_time_controls(moves, args.min_games)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    print(f"Side: {args.side} | time controls with >= {args.min_games} games: {tcs}\n")

    print("1. clock_after percentiles (seconds) at given move numbers")
    print("   (games_still_running shrinks with move number: later rows = long games only)")
    print(percentiles_by_move(moves, tcs).to_string(index=False))

    print("\n2. Share of moves below each threshold")
    print(share_of_moves_below(moves, tcs).to_string(index=False))

    onset = per_game_onset(moves, tcs)
    ever_cols = ["tc", "n_games"] + [c for c in onset.columns if c.startswith("games ever")]
    onset_cols = ["tc"] + [c for c in onset.columns if c.startswith("median onset")]
    print("\n3a. Per game: share of games that ever go below each threshold")
    print(onset[ever_cols].to_string(index=False))
    print("\n3b. Per game: median move number of the first move below it")
    print("   (among the games that do go below; see 3a for how many)")
    print(onset[onset_cols].to_string(index=False))


if __name__ == "__main__":
    main()
