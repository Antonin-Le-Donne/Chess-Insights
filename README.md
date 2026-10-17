# Chess Insights

Data pipeline + ML model that analyzes my own chess games to find when, where,
and why I blunder. Lichess/chess.com API → Stockfish → pandas → Streamlit → LightGBM.

Not a Lichess Insights clone: every metric tests a specific hypothesis about my
own failure patterns — mental fatigue curve, time-management leaks, conversion
greed in winning positions, and opening mismatches.

## Status

- **v1 — pipeline + analytics dashboard: working, not complete.** Lichess +
  chess.com, blitz/rapid/bullet, ~11,251 games (~9,200 blitz+rapid, ~2,500+
  bullet after M3 filtering), evaluated with local Stockfish or Lichess
  server evals. Metrics M1–M5 and their Streamlit pages are live; findings
  below were re-checked in October 2026 after an eval-import bug fix.
- **v2 — personal blunder-risk model: trained and evaluated.** LightGBM on
  the current rating regime, reproducible train/evaluate scripts, PR-AUC
  with bootstrap CIs, SHAP analysis, and a dashboard page — see
  [What actually predicts my blunders](#what-actually-predicts-my-blunders--v2-model).

See [`chess_insights_spec.md`](chess_insights_spec.md) for the full specification
and [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) for decisions and findings.

## Planned

- **v2** — calibration plot on the model page; optional per-game risk timeline.
- **v1** — M7 (time of day / session position), M6 (swindle profile),
  sidebar filters, hover-to-game links, month-over-month M3 tracking on the
  Overview page.
- **Pipeline** — incremental parsing (append new games without rewriting
  existing Stockfish evaluations), then a refresh with the latest games.

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

## What Actually Predicts My Blunders? — V2 Model

The LightGBM model outperforms the logistic-regression baseline, and the strongest signal identified so far is an elevated error rate when my win probability is around 70–80%. Centipawn diagnostics and a SHAP dependence plot support this pattern. These findings are observational and do not establish causality.

### Scope

The model predicts whether one of my moves will be labelled a blunder, using only information available before that move.

The dataset includes games above my rating thresholds: **Lichess blitz (1700), Lichess bullet (1950), and Chess.com bullet (1650)**. Rapid is excluded because performance has not yet reached a stable plateau in that format.

The evaluation uses my own moves after ply 10, restricted to positions where my win probability is at least 20%. Below 20%, a drop of 20 percentage points is impossible by definition, so these moves cannot be labelled as blunders under the chosen threshold. The test set consists of my **419 most recent games**, held out chronologically at game level.

### Model performance

The temporal test set contains **10,094 moves**, with a **6.4% blunder rate**.

- **LightGBM PR-AUC:** 0.191 (95% CI: 0.163–0.225), compared with 0.064 for a random-ranking reference.
- **Logistic-regression baseline PR-AUC:** 0.172.
- **Paired game-level bootstrap:** LightGBM's PR-AUC improvement over logistic regression is **+0.019** (95% CI: +0.003 to +0.038); LightGBM performs better in **98.8%** of resamples.
- **Calibration:** mean predicted risk is 7.4%, versus 6.4% observed.
- **Risk ranking:** the top risk decile has a 21.8% blunder rate (about **3.4×** the test-set base rate), while the bottom decile has a 0.2% rate.

The model is useful for ranking positions by risk, with a measurable improvement over the logistic-regression baseline.

### Main finding: the 70–80% win-probability danger zone

The strongest finding so far is that positions where I am clearly better but have not yet converted the advantage—roughly **70–80% win probability**—are unusually error-prone.

A separate centipawn diagnostic defines a large error as a loss of at least **200 centipawns** between the pre-move and post-move evaluations. In the 70–80% win-probability bucket, the large-error rate is **18.9%**, compared with roughly 10% across much of the surrounding range.

The pattern remains after stratifying by move number:

| Move-number band | Large-error rate at 70–80% win probability | Sample size |
|---|---:|---:|
| 6–15 | 17.3% | 1,614 |
| 16–30 | 19.4% | 2,411 |
| 31+ | 20.3% | 893 |

This consistency across move-number bands makes the pattern less likely to be explained by a single phase of the game.

The strongest model-based evidence comes from the **SHAP dependence plot for `eval_before_move`**. Its SHAP contribution forms a positive bump around the 70–80% win-probability zone, indicating that the model associates this specific zone with increased predicted blunder risk. This pattern appears while the model accounts for other features, including clock pressure, material imbalance, and evaluation volatility. Unlike a global feature-importance ranking, the dependence plot directly supports the specific risk pattern observed in the centipawn analysis.

The global SHAP ranking also identifies evaluation volatility (0.49), `eval_before_move` (0.29), `clock_before_move_pct` (0.22), and material imbalance (0.16) as the four leading features.

The finding is also relevant to conversion: the M3 game-level analysis found a **72.6% conversion rate** from positions at win% ≥ 80 (**n = 4,510** main games, blitz and rapid combined), with a median of 30 plies to finish overall and 24 plies among converted wins. This is related evidence, not the same population or exact win-probability bucket as the 70–80% move-level analysis.

### Important caveats

- **Win-percentage labels flatten near the extremes.** In lost or nearly won positions, a large centipawn loss can still produce a relatively small change in win probability. The centipawn diagnostic therefore finds substantial error rates in extreme buckets even when the win%-based blunder label marks fewer moves as blunders.
- **Training-seed variability:** the PR-AUC of the same model varies by approximately 0.01 across random seeds. This is a meaningful source of uncertainty when interpreting a performance difference of +0.019 over the baseline.
- **The test set is limited to 419 recent games.** Although it contains 10,094 moves, these moves are clustered within games and are not independent observations. Results may also reflect the characteristics of this recent period.
- **The 50–60% bucket has a low large-error rate.** This may reflect quieter positions, including opening theory or dead-drawn endgames, rather than a general psychological effect. It is a candidate pattern, not a proven explanation.
- **The late-game 40–50% bucket may be riskier.** In moves 31+, its large-error rate is 18.9% (n = 1,587; win%-label blunder rate 17.1%). This is an observational subgroup result and should be treated cautiously.
- **Uncertainty matters.** For a rate near 19% with a sample of about 900 moves, a simple binomial 95% interval is roughly ±2.6 percentage points. This ignores within-game correlation, so it may understate uncertainty.
- **The evidence is observational.** These analyses identify associations and predictive patterns; they do not establish why these errors occur.

### Current conclusion

The V2 model has measurable ranking value over the logistic-regression baseline. The most actionable finding is the elevated rate of large errors when I am at **70–80% win probability**: it persists across move-number bands, appears in the centipawn diagnostic, and is supported by a positive SHAP dependence pattern while accounting for clock pressure, material imbalance, and evaluation volatility.

This is the strongest converging signal identified so far, but it remains an observational result rather than proof of a causal mechanism.
