"""One-off cleanup: drop the chess.com games that parse.py should never have
kept (variants, unrated games, time classes outside config.perf_types; spec
section 1, "rated ... at parse time") from games.parquet and moves.parquet.

parse.py now skips them (parse.chesscom_skip_reason), but it is incremental
and never rewrites existing rows, so the ones parsed before the fix stay
until removed here. Every other row, Stockfish evals included, is kept.

Dry run by default: prints what would be removed, writes nothing.
--apply: copies both parquets to data/processed/backup_<timestamp>/ first,
then rewrites them (temporary file + replace).

After --apply, re-run `python -m src.features` (derived columns are
per-game, but the label guardrail and moves.parquet consistency get
re-checked there).

Usage (from the repo root):
    python one_use_scripts/remove_out_of_scope_chesscom.py
    python one_use_scripts/remove_out_of_scope_chesscom.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fetch import load_config  # noqa: E402
from src.parse import chesscom_skip_reason  # noqa: E402

PROCESSED = Path("data/processed")


def out_of_scope_ids(perf_types: list[str]) -> dict[str, str]:
    """game_id -> skip reason, for every out-of-scope chess.com archive entry."""
    reasons: dict[str, str] = {}
    for json_path in sorted(Path("data/raw/chesscom").glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            for game_obj in json.load(f)["games"]:
                reason = chesscom_skip_reason(game_obj, perf_types)
                if reason is not None:
                    external_id = game_obj["url"].rstrip("/").split("/")[-1]
                    reasons[f"chesscom_{external_id}"] = reason
    return reasons


def write_atomically(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove out-of-scope chess.com games.")
    parser.add_argument("--apply", action="store_true", help="write the cleaned parquets")
    args = parser.parse_args()

    config = load_config()
    games = pd.read_parquet(PROCESSED / "games.parquet")
    moves = pd.read_parquet(PROCESSED / "moves.parquet")

    reasons = out_of_scope_ids(config.perf_types)
    to_drop = games.loc[games["game_id"].isin(reasons), ["game_id", "perf_type", "date_utc"]]
    drop_ids = set(to_drop["game_id"])
    n_moves = int(moves["game_id"].isin(drop_ids).sum())

    by_reason = Counter(reasons[game_id] for game_id in to_drop["game_id"])
    print(f"games.parquet: {len(games):,} games, {len(moves):,} moves")
    print(f"Out of scope and present: {len(drop_ids):,} games, {n_moves:,} moves")
    for reason, n in by_reason.most_common():
        print(f"  {reason}: {n}")
    print("By perf_type:", to_drop["perf_type"].value_counts().to_dict())
    print(f"Remaining after removal: {len(games) - len(drop_ids):,} games")

    if not args.apply:
        print("\nDry run: nothing written. Re-run with --apply to remove them.")
        return
    if not drop_ids:
        print("Nothing to remove.")
        return

    backup = PROCESSED / f"backup_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    backup.mkdir(parents=True)
    shutil.copy2(PROCESSED / "games.parquet", backup / "games.parquet")
    shutil.copy2(PROCESSED / "moves.parquet", backup / "moves.parquet")
    print(f"\nBackup: {backup}")

    kept_games = games[~games["game_id"].isin(drop_ids)]
    kept_moves = moves[~moves["game_id"].isin(drop_ids)]

    # Same consistency rule as parse.py: every move belongs to a kept game.
    if not kept_moves["game_id"].isin(kept_games["game_id"]).all():
        raise SystemExit("Inconsistent result (moves without a game): nothing written.")

    write_atomically(kept_moves, PROCESSED / "moves.parquet")
    write_atomically(kept_games, PROCESSED / "games.parquet")
    print(
        f"Written: {len(kept_games):,} games, {len(kept_moves):,} moves. "
        "Next: python -m src.features"
    )


if __name__ == "__main__":
    main()
