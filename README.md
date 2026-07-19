# Chess Insights

Data pipeline + ML model that analyzes my own chess games to find when, where,
and why I blunder. Lichess/chess.com API → Stockfish → pandas → Streamlit → LightGBM.

Not a Lichess Insights clone: every metric tests a specific hypothesis about my
own failure patterns — mental fatigue curve, time-management leaks, conversion
greed in winning positions, and opening mismatches.

## Status

Work in progress — v1 (data pipeline + analytics dashboard) under construction.
See [`chess_insights_spec.md`](chess_insights_spec.md) for the full specification
and [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) for decisions and findings.

## Planned

- **v1** — Fetch my games (Lichess + chess.com), evaluate with local Stockfish,
  compute per-move metrics, explore them in a Streamlit dashboard.
- **v2** — Supervised model (LightGBM) predicting my personal blunder-risk
  contexts, with SHAP analysis of what actually drives my mistakes.

## Findings

_To be filled as milestones complete — the findings are the deliverable._