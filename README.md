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

1. **Blunder rate peaks mid-game, not a simple fatigue ramp.** Survives a
   survivorship-bias control (restricted to games with n_plies >= 40, so
   short games ending early can't be driving it): blunder rate rises from
   ~0.5% at the start to a peak of ~5.5-6.5% around move 30-40, then
   declines through the endgame (~2% by move 65, though the tail past
   move 80 has too few games to trust). Not a continuously draining
   battery. Hypothesis, not yet tested: middlegame complexity (more
   pieces, more candidate moves) drives the peak rather than pure
   fatigue — directly testable against a complexity proxy (legal move
   count, hanging pieces, both already planned as v2 features)
   independent of move number; tracked as a follow-up, not run yet.
   Caveat on the decline itself: win% saturates near 0/100 in decisive
   endgame positions, so a fixed 20-point-drop blunder definition gets
   mechanically less sensitive there — part of the apparent decline is
   likely measurement scale, not purely easier calculation.

2. **Time→blunder signature confirmed.** After a "long think" (>8x the
   game's median thinking time), the probability of blundering within my
   next 3 moves is 14.6%, vs a 3.9% baseline (my overall blunder rate
   across all moves, all phases) — a 3.8x ratio, on a sample of 1,240
   long-think instances (181 followed by a blunder within 3 moves).

3. **Conversion leak quantified.** Of 1,155 games where I reached a ≥80%
   win probability, only 74.5% converted to a win — roughly 1 in 4
   "should-win" positions gets thrown away. Median plies to finish: 27
   overall, 24 for converted wins (thrown games drag on longer before
   collapsing).

4. **Time pressure and the mid-game blunder peak look like separate
   failure modes.** Genuine time pressure (clock below 20% of base time)
   only sets in around move 56 in blitz — well after the move 30-40
   blunder-rate peak from finding #1. Calculation load peaks in the
   middlegame independently of the clock actually running low.

5. **Opening mismatch hypothesis reversed.** I score 58.0% playing the
   Sicilian as Black (n=182) — above average, not a weakness. The real
   leak is facing the Sicilian as White (47.6%, n=185). Checked against
   an opponent-strength confound (are stronger players choosing the
   Sicilian against me?): opponents playing the Sicilian are on average
   *weaker* than my overall pool (rating_diff ≈ -2.9), not stronger — the
   confound doesn't explain the number away, if anything it makes it more
   notable. Facing the Italian Game as Black (38.3%, n=47) shows the same
   pattern (opponents ≈ -6.2 weaker on average, so not an Elo artifact
   either), but n=47 still gives a ±14-point 95% CI — not distinguishable
   from 50% on sample size alone. Kept as a hypothesis to watch, not a
   finding, pending more games.