"""Throwaway check (no model): is the blunder-rate shape across win% buckets
about MY play, or about the win%-drop label? Delete after use.

If my error size did not depend on the position, the rate of large
centipawn losses would be roughly flat across win% buckets, while the
win%-based blunder rate could peak near 60 by construction.

Use the same current-regime population as the model: my moves, ply > 10,
non-null move_quality, and win_pct before the move >= the configured
blunder threshold.

Convention: eval_cp and win_pct are evaluations of fen_before for that
ply. On my moves, win_pct is already MY win percentage before my move.
"""

from pathlib import Path
import tomllib

import pandas as pd

from src.scope import filter_current_regime


# 200 cp represents a two-pawn evaluation swing: a substantial loss
# without restricting the diagnostic to only extreme cases.
CP_LOSS_THRESHOLD = 200

# Clip mate scores and extreme evaluations before calculating cp_loss.
CP_CLIP = 1000


def main() -> None:
    project_root = Path(__file__).resolve().parent
    processed_dir = project_root / "data" / "processed"

    with open(project_root / "config.toml", "rb") as f:
        config = tomllib.load(f)

    blunder_threshold = config["thresholds"]["blunder"]

    games = pd.read_parquet(
        processed_dir / "games.parquet"
    )
    moves = pd.read_parquet(
        processed_dir / "moves.parquet"
    )

    games_current, moves_current = filter_current_regime(
        games,
        moves,
        config["current_regime"],
    )

    moves_current = moves_current.sort_values(
        ["game_id", "ply"]
    ).copy()

    # The next ply's eval_cp is the position after my move:
    # eval_cp is the evaluation of fen_before, from White's perspective.
    moves_current["next_eval_cp"] = (
        moves_current.groupby("game_id")["eval_cp"].shift(-1)
    )

    # On my moves, win_pct already represents MY win percentage before
    # the move. Do not shift it or invert it.
    moves_current["win_pct_before_move"] = moves_current["win_pct"]

    game_context = games_current[
        ["game_id", "my_color"]
    ]

    moves_current = moves_current.merge(
        game_context,
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    # Convert both evaluations from White's perspective to MY perspective.
    color_sign = moves_current["my_color"].map({
        "white": 1,
        "black": -1,
    })

    moves_current["my_eval_before_cp"] = (
        moves_current["eval_cp"] * color_sign
    ).clip(
        lower=-CP_CLIP,
        upper=CP_CLIP,
    )

    moves_current["my_eval_after_cp"] = (
        moves_current["next_eval_cp"] * color_sign
    ).clip(
        lower=-CP_CLIP,
        upper=CP_CLIP,
    )

    # Positive cp_loss means I lost evaluation after my move.
    moves_current["cp_loss"] = (
        moves_current["my_eval_before_cp"]
        - moves_current["my_eval_after_cp"]
    )

    # Same move filters as the model. move_quality.notna() means the
    # move has a label; it does not mean the game was rated.
    model_population = moves_current[
        moves_current["is_my_move"]
        & (moves_current["ply"] > 10)
        & moves_current["move_quality"].notna()
        & (
            moves_current["win_pct_before_move"]
            >= blunder_threshold
        )
    ].copy()

    # Use the same denominator for both rates: moves with valid evaluations
    # before and after the move. Otherwise, the rate comparison is biased
    # by moves for which cp_loss cannot be calculated.
    population = model_population.dropna(
        subset=["cp_loss"]
    ).copy()

    if population.empty:
        raise ValueError(
            "No eligible moves have valid evaluations both before "
            "and after the move."
        )

    # Build buckets starting at the configured blunder threshold.
    bucket_edges = [float(blunder_threshold)]
    bucket_edges.extend(
        float(edge)
        for edge in range(10, 100, 10)
        if edge > blunder_threshold
    )
    bucket_edges.append(100.000001)

    bucket_labels = [
        f"{lower:g}-{min(upper, 100.0):g}"
        for lower, upper in zip(
            bucket_edges[:-1],
            bucket_edges[1:],
        )
    ]

    population["win_pct_bucket"] = pd.cut(
        population["win_pct_before_move"],
        bins=bucket_edges,
        labels=bucket_labels,
        right=False,
        include_lowest=True,
    )

    summary = (
        population.groupby(
            "win_pct_bucket",
            observed=False,
        )
        .agg(
            n=("cp_loss", "size"),
            blunder_rate_win_pct_label=(
                "move_quality",
                lambda values: 100 * values.eq("blunder").mean(),
            ),
            cp_loss_rate=(
                "cp_loss",
                lambda values: (
                    100 * values.ge(CP_LOSS_THRESHOLD).mean()
                ),
            ),
            median_cp_loss=("cp_loss", "median"),
        )
    )

    summary.index.name = "win_pct_before_move"

    print(f"Current-regime games: {len(games_current):,}")
    print(
        "Eligible model moves before requiring valid CP evaluations: "
        f"{len(model_population):,}"
    )
    print(
        "Moves with valid CP loss (table denominator): "
        f"{len(population):,}"
    )
    print(f"Centipawn-loss threshold: {CP_LOSS_THRESHOLD} cp")
    print(
        "\nRates are percentages. Compare the SHAPES of the two rate "
        "columns, not their absolute levels.\n"
    )
    print(
        summary.to_string(
            float_format=lambda value: f"{value:.1f}"
        )
    )

        # Split the same eligible population by game phase.
    # Intervals are [6, 16), [16, 31), and [31, +inf).
    population["move_number_band"] = pd.cut(
        population["move_number"],
        bins=[6, 16, 31, float("inf")],
        labels=["6-15", "16-30", "31+"],
        right=False,
        include_lowest=True,
    )

    population["win_pct_bucket"] = pd.cut(
        population["win_pct_before_move"],
        bins=bucket_edges,
        labels=bucket_labels,
        right=False,
        include_lowest=True,
    )

    # population already excludes moves without a valid cp_loss, so both
    # rates use the same denominator within every move-number / win% cell.
    summary = (
        population.groupby(
            ["move_number_band", "win_pct_bucket"],
            observed=False,
        )
        .agg(
            n=("cp_loss", "size"),
            blunder_rate_win_pct_label=(
                "move_quality",
                lambda values: 100 * values.eq("blunder").mean(),
            ),
            cp_loss_rate=(
                "cp_loss",
                lambda values: (
                    100 * values.ge(CP_LOSS_THRESHOLD).mean()
                ),
            ),
            median_cp_loss=("cp_loss", "median"),
        )
    )

    summary.index.names = [
        "move_number_band",
        "win_pct_before_move",
    ]

    print(f"Current-regime games: {len(games_current):,}")
    print(
        "Eligible model moves before requiring valid CP evaluations: "
        f"{len(model_population):,}"
    )
    print(
        "Moves with valid CP loss (table denominator): "
        f"{len(population):,}"
    )
    print(f"Centipawn-loss threshold: {CP_LOSS_THRESHOLD} cp")
    print(
        "\nRates are percentages. Compare the SHAPES of the two rate "
        "columns within each move-number band, not their absolute levels. "
        "Check n before interpreting any cell.\n"
    )
    print(
        summary.to_string(
            float_format=lambda value: f"{value:.1f}"
        )
    )


if __name__ == "__main__":
    main()