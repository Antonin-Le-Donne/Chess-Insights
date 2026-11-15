"""One-off: the statistics behind a player profile for a coach (2026-11-15),
for any processed dataset (e.g. a friend's clone). Read-only.

Every comparison is me vs opponents in the same games, blitz/rapid ("main")
and bullet separately. Definitions reuse the project's: blunder = move_quality,
heavy error = mover_cp_loss >= 200 (evals clipped at +/-1000), win% of the mover
before the move, time pressure = clock_after < 10% of base.

Usage (from the repo root):
    python one_use_scripts/player_profile_stats.py --data-dir "C:/path/to/clone/data/processed"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import classify_termination, elo_expected_score, mover_cp_loss  # noqa: E402

HEAVY_CP = 200


def group_of(perf_type: pd.Series) -> pd.Series:
    return perf_type.where(perf_type.eq("bullet"), "main")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def ratings(games: pd.DataFrame) -> None:
    section("Ratings per platform / time control (monthly mean, first and last month with 20+ games, peak game rating)")
    games = games.assign(month=games["date_utc"].dt.to_period("M"))
    for (platform, perf), g in games.groupby(["platform", "perf_type"]):
        monthly = g.groupby("month").agg(r=("my_rating", "mean"), n=("my_rating", "size"))
        solid = monthly[monthly["n"] >= 20]
        if solid.empty:
            continue
        print(
            f"{platform} {perf}: {len(g):,} games | {solid.index[0]} {solid['r'].iloc[0]:.0f} -> "
            f"{solid.index[-1]} {solid['r'].iloc[-1]:.0f} | peak {g['my_rating'].max():.0f}"
        )
    tc = games.assign(tc=games["base_seconds"].astype(int).astype(str) + "+" + games["increment_seconds"].astype(int).astype(str))
    print("Most played time controls:", tc["tc"].value_counts().head(6).to_dict())
    recent = games[games["date_utc"] >= games["date_utc"].max() - pd.Timedelta(days=90)]
    print("Last 90 days by perf:", recent["perf_type"].value_counts().to_dict())


def results(games: pd.DataFrame) -> None:
    section("Results and score vs Elo expectation (chess.com ratings are post-game: small bias)")
    expected = elo_expected_score(games["my_rating"] - games["opp_rating"])
    for group, g in games.groupby(group_of(games["perf_type"])):
        e = expected.loc[g.index]
        print(
            f"{group}: W/D/L {g['result_for_me'].eq(1).mean():.1%} / {g['result_for_me'].eq(0.5).mean():.1%} / "
            f"{g['result_for_me'].eq(0).mean():.1%} | score {g['result_for_me'].mean():.1%} | "
            f"vs Elo expectation {100 * (g['result_for_me'] - e).mean():+.1f} pts (n={len(g):,})"
        )
    cats = classify_termination(games["termination"])
    for group, g in games.groupby(group_of(games["perf_type"])):
        lost = g["result_for_me"].eq(0)
        print(f"{group} — how I lose:", (cats[g.index][lost].value_counts(normalize=True).round(3).head(5)).to_dict())
        print(f"{group} — how I win:", (cats[g.index][g['result_for_me'].eq(1)].value_counts(normalize=True).round(3).head(5)).to_dict())


def errors(moves: pd.DataFrame, games: pd.DataFrame) -> None:
    section("Blunder rate per move, me vs opponents, by phase")
    rated = moves[moves["move_quality"].notna()]
    rated = rated.assign(group=group_of(rated["game_id"].map(games.set_index("game_id")["perf_type"])))
    table = rated.groupby(["group", "phase", "is_my_move"])["move_quality"].apply(lambda s: s.eq("blunder").mean()).unstack()
    table.columns = ["opponent", "me"]
    print((100 * table[["me", "opponent"]]).round(2).to_string())
    overall = rated.groupby(["group", "is_my_move"])["move_quality"].apply(lambda s: s.eq("blunder").mean()).unstack()
    print("overall (%):", (100 * overall).round(2).to_dict())

    section("Heavy errors (>= 200 cp) by the mover's win% before the move, me vs opponents (%), buckets (lo, hi]")
    cp = moves.assign(cp_loss=mover_cp_loss(moves), group=group_of(moves["game_id"].map(games.set_index("game_id")["perf_type"])))
    cp = cp[cp["cp_loss"].notna() & cp["win_pct"].notna()]
    cp["bucket"] = pd.cut(cp["win_pct"], bins=range(0, 101, 10), include_lowest=True)
    heavy = cp.groupby(["group", "bucket", "is_my_move"], observed=True)["cp_loss"].apply(lambda s: s.ge(HEAVY_CP).mean()).unstack()
    heavy.columns = ["opponent", "me"]
    counts = cp.groupby(["group", "bucket"], observed=True).size().rename("n")
    print((100 * heavy[["me", "opponent"]]).round(1).join(counts).to_string())


def hanging(moves: pd.DataFrame, games: pd.DataFrame, sample: int = 40000) -> None:
    section("Loose pieces: my blunder rate when >= 1 of my pieces is attacked more than defended (before my move)")
    mine = moves[moves["is_my_move"] & moves["move_quality"].notna() & moves["ply"].gt(10)]
    mine = mine.sample(min(sample, len(mine)), random_state=42)
    color = mine["game_id"].map(games.set_index("game_id")["my_color"])
    counts = []
    for fen, c in zip(mine["fen_before"], color):
        board = chess.Board(fen)
        me, opp = (chess.WHITE, chess.BLACK) if c == "white" else (chess.BLACK, chess.WHITE)
        counts.append(sum(
            1 for sq, p in board.piece_map().items()
            if p.color == me and p.piece_type != chess.KING
            and len(board.attackers(opp, sq)) > len(board.attackers(me, sq))
        ))
    mine = mine.assign(loose=np.array(counts) > 0)
    stats = mine.groupby("loose")["move_quality"].agg(rate=lambda s: s.eq("blunder").mean(), n="size")
    print(stats.to_string())
    print(f"share of my moves with a loose piece: {mine['loose'].mean():.1%}")


def clock(moves: pd.DataFrame, games: pd.DataFrame) -> None:
    section("Clock left (% of base) at moves 10/20/30/40, median, me vs opponents, main time controls")
    info = games.set_index("game_id")
    m = moves.assign(
        tc=moves["game_id"].map(info["base_seconds"].astype(int).astype(str) + "+" + info["increment_seconds"].astype(int).astype(str)),
        base=moves["game_id"].map(info["base_seconds"]),
    )
    top = m.groupby("tc")["game_id"].nunique().sort_values(ascending=False).head(4).index
    for tc in top:
        g = m[m["tc"].eq(tc) & m["base"].gt(0)]
        row = []
        for move in (10, 20, 30, 40):
            at = g[g["move_number"].eq(move)]
            mine = (at.loc[at["is_my_move"], "clock_after"] / at.loc[at["is_my_move"], "base"]).median()
            opp = (at.loc[~at["is_my_move"], "clock_after"] / at.loc[~at["is_my_move"], "base"]).median()
            row.append(f"m{move}: {mine:.0%} vs {opp:.0%}")
        print(f"{tc} ({g['game_id'].nunique():,} games): " + " | ".join(row))

    section("Losses on time")
    cats = classify_termination(games["termination"])
    lost_on_time = games[games["result_for_me"].eq(0) & cats.eq("time")]
    last_mine = (
        moves[moves["is_my_move"] & moves["game_id"].isin(lost_on_time["game_id"])]
        .sort_values("ply").groupby("game_id")["win_pct"].last()
    )
    for group, g in lost_on_time.groupby(group_of(lost_on_time["perf_type"])):
        total_losses = games[group_of(games["perf_type"]).eq(group) & games["result_for_me"].eq(0)]
        winning = last_mine.reindex(g["game_id"]).ge(80).sum()
        print(
            f"{group}: {len(g):,} losses on time = {len(g) / len(total_losses):.0%} of losses; "
            f"{winning} of them while >= 80% before my last move"
        )


def endgames(moves: pd.DataFrame, games: pd.DataFrame) -> None:
    section("Entering the endgame balanced (my win% 40-60 at the first endgame position): score vs implied")
    my_pct = moves["win_pct"].where(moves["is_my_move"], 100 - moves["win_pct"])
    first_end = moves.assign(my_pct=my_pct)[moves["phase"].eq("endgame")].sort_values("ply").groupby("game_id").first()
    balanced = first_end[first_end["my_pct"].between(40, 60)]
    g = games.set_index("game_id").loc[balanced.index]
    for group, rows in g.groupby(group_of(g["perf_type"])):
        print(
            f"{group}: n={len(rows):,} | score {rows['result_for_me'].mean():.1%} vs implied "
            f"{balanced.loc[rows.index, 'my_pct'].mean() / 100:.1%}"
        )


def openings(moves: pd.DataFrame, games: pd.DataFrame, min_n: int = 30) -> None:
    section(f"Openings by move sequence: score and score vs Elo expectation (pts), n >= {min_n}")
    first = moves[moves["ply"].le(6)].sort_values(["game_id", "ply"]).groupby("game_id")["san"].agg(list)
    g = games.set_index("game_id").loc[first.index].assign(seq=first)
    g["gap"] = 100 * (g["result_for_me"] - elo_expected_score(g["my_rating"] - g["opp_rating"]))

    def report(label: str, rows: pd.DataFrame, depth: int) -> None:
        key = rows["seq"].apply(lambda s: " ".join(s[:depth]))
        stats = rows.groupby(key).agg(n=("gap", "size"), score=("result_for_me", "mean"), gap=("gap", "mean"))
        stats = stats[stats["n"] >= min_n].sort_values("n", ascending=False).head(12)
        print(f"-- {label}")
        print(stats.round({"score": 3, "gap": 1}).to_string())

    white, black = g[g["my_color"].eq("white")], g[g["my_color"].eq("black")]
    report("As White, my first move", white, 1)
    report("As White, first 3 plies", white, 3)
    report("As Black, opponent's first move + my reply", black, 2)
    report("As Black, first 4 plies", black, 4)


def sessions(games: pd.DataFrame) -> None:
    section("Volume: games per active day, hours")
    days = games["date_utc"].dt.date
    print(f"active days: {days.nunique():,} | median games per active day: {games.groupby(days).size().median():.0f}")
    print("local hour (Europe/Paris) distribution, top:", games["date_utc"].dt.tz_localize("UTC").dt.tz_convert("Europe/Paris").dt.hour.value_counts().head(6).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    data = Path(args.data_dir)
    games = pd.read_parquet(data / "games.parquet")
    moves = pd.read_parquet(data / "moves.parquet")
    games["date_utc"] = pd.to_datetime(games["date_utc"])
    print(f"{len(games):,} games, {games['date_utc'].min():%Y-%m-%d} -> {games['date_utc'].max():%Y-%m-%d}")
    ratings(games)
    results(games)
    errors(moves, games)
    hanging(moves, games)
    clock(moves, games)
    endgames(moves, games)
    openings(moves, games)
    sessions(games)


if __name__ == "__main__":
    main()
