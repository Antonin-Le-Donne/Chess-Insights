"""Readable labels for the V2 model features, shared by the model pages."""

from __future__ import annotations

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


def feature_label(feature: str) -> str:
    """Return a readable label for a model feature."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return feature.replace("_", " ").capitalize()
