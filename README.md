# Chess Insights

Data pipeline + ML model that analyzes my own chess games to find when, where,
and why I blunder. Lichess/chess.com API → Stockfish → pandas → Streamlit → LightGBM.

Not a Lichess Insights clone: every metric tests a specific hypothesis about my
own failure patterns — mental fatigue curve, time-management leaks, conversion
greed in winning positions, and opening mismatches.

## Status

Work in progress — v1 (data pipeline + analytics dashboard) under construction.
Data pipeline now covers Lichess + chess.com combined, blitz/rapid/bullet,
~11,251 games total (~9,200 blitz+rapid, ~2,500+ bullet after M3 filtering).
See [`chess_insights_spec.md`](chess_insights_spec.md) for the full specification
and [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) for decisions and findings.

## Planned

- **v1** — Fetch my games (Lichess + chess.com), evaluate with local Stockfish,
  compute per-move metrics, explore them in a Streamlit dashboard.
- **v2** — Supervised model (LightGBM) predicting my personal blunder-risk
  contexts, with SHAP analysis of what actually drives my mistakes.

## Findings

_Blitz + rapid ("main" in `metrics.py`), Lichess + chess.com combined,
~9,200 qualifying games. Bullet is analyzed separately below — the two are
kept structurally apart throughout `metrics.py`, not blended into one
per-time-control split._

1. **Blunder rate peaks mid-game, not a simple fatigue ramp.** Survives a
   survivorship-bias control (restricted to games with n_plies >= 40, so
   short games ending early can't be driving it): blunder rate rises from
   ~0.3% at the start to a peak of ~6.4-6.8% around **move 15-20**, then
   declines through the endgame (~1-2% by move 65-80, though the tail past
   move 90 has too few games to trust). Revised from an earlier ~move 30-40
   estimate on a much smaller sample (1,909 games) — the fuller dataset
   places the peak notably earlier, confirmed by both the filtered and
   unfiltered curve agreeing on the same location this time. Not a
   continuously draining battery. Hypothesis, not yet tested: middlegame
   complexity (more pieces, more candidate moves) drives the peak rather
   than pure fatigue — directly testable against a complexity proxy (legal
   move count, hanging pieces, both already planned as v2 features)
   independent of move number; tracked as a follow-up, not run yet. Caveat
   on the decline itself: win% saturates near 0/100 in decisive endgame
   positions, so a fixed 20-point-drop blunder definition gets mechanically
   less sensitive there — part of the apparent decline is likely
   measurement scale, not purely easier calculation.

2. **Time→blunder signature confirmed, stable under a much larger
   sample.** After a "long think" (>8x the game's median thinking time),
   the probability of blundering within my next 3 moves is 17.2%, vs a
   4.6% baseline (my overall blunder rate across all moves, all phases) —
   a **3.75x ratio**, on a sample of 4,414 long-think instances (760
   followed by a blunder within 3 moves). Essentially unchanged from the
   original 3.77x on a sample 3.5x smaller. (Rechecked in October 2026
   after fixing an off-by-one in how Lichess server evaluations were
   imported — it was 3.72x before; that bug only touched ~3.5% of the
   history, so every finding on this page held.)

3. **Conversion leak quantified — bigger than first estimated.** Of 4,510
   games (blitz+rapid) where I reached a ≥80% win probability, only
   **72.6%** converted to a win — more than 1 in 4 "should-win" positions
   get thrown away. Median plies to finish: 30 overall, 24 for converted
   wins. Revised down from an initial 74.5%/1,155-game estimate — the
   larger, more representative sample shows a bigger leak, not a smaller
   one.

4. **Time pressure and the mid-game blunder peak look like separate
   failure modes — the gap is wider than first thought.** Genuine time
   pressure (clock below 20% of base time) only sets in around **move 67**
   — well after the move 15-20 blunder-rate peak from finding #1.
   Calculation load peaks in the middlegame independently of the clock
   actually running low, and the two events are now further apart than
   the original estimate (peak at 30-40, pressure at 56) suggested.

5. **Opening mismatch hypothesis reversed — and the reversal itself
   reversed again with more data.** Sicilian as Black: 55.9% (n=1,395) —
   still not a weakness. Facing the Sicilian as White, the number that
   originally looked like a leak (47.6%, n=185) is now **51.8%** (n=613,
   3.3x the sample) — essentially break-even, no longer a finding at all.
   Meanwhile facing the Italian Game as Black, previously flagged as
   "too small to trust" (38.3%, n=47), now sits at **45.0%** on n=200 (4.3x
   the sample) — still below 50%, with a much tighter interval (~±6.9
   points at 95% CI, vs ±14 before). Checked against the opponent-strength
   confound again: opponents are *weaker* than my pool on both the
   Sicilian (rating_diff ≈ -8.1) and especially the Italian
   (rating_diff ≈ -15.2) — if anything this makes the Italian number more
   notable, not less. **The Italian Game as Black is now the better-
   supported finding of the two**, exactly reversing which opening looked
   like the real leak on the first, smaller pass. Lesson for the project
   as much as for the chess: an opening-mismatch finding built on n<50
   should be labeled a hypothesis, not a result, until the sample says
   otherwise.

## Bullet findings

_Same ~9,200-game vintage, `perf_group == "bullet"` in `metrics.py`,
2,512 qualifying games for M3. M5 (opening surgery) is not split by time
control — a deliberate choice, see `PROJECT_TRACKER.md` — so bullet-specific
opening numbers don't exist yet._

- **Sharper, earlier blunder peak than blitz/rapid.** ~11.7% blunder rate
  around move 15 (vs ~6.8% for main at the same point) — despite bullet
  being my strongest time control by peak rating (2,200+, against a ~2,050
  ceiling in blitz), the raw blunder rate at the critical early-middlegame
  point runs noticeably higher here. Noisier than the main-line curve
  (smaller per-bucket samples within bullet's own duration terciles), so
  the magnitude is a rougher estimate than finding #1's.
- **Time→blunder ratio holds up even under bullet's compressed clock.**
  3.84x (vs 3.75x for main) — the "recalculate, then blunder" signature
  isn't specific to having minutes to spare; it shows up even when the
  whole game lasts under a minute.
- **Conversion is worse in bullet, not better — a real tension with the
  strength gap.** 67.3% (n=2,512) vs 72.6% for main — despite the higher
  peak rating in bullet, winning positions get thrown away more often
  here, not less, a 5.3-point gap. Median plies to finish: 32 (28 for
  converted wins), both a touch longer than main's 30/24.
- **Time pressure sets in far earlier, as expected from the clock alone.**
  Move 54, vs move 67 for main — a shorter base time (60s typical) makes
  this mechanical rather than surprising, included mainly as a sanity
  check that M4 behaves as it should.