# Chess Insights — Findings in detail

The full version of the results summarised in the [README](../README.md): exact
definitions, sample sizes, confidence intervals, definition history and caveats.
Metric definitions live in [`chess_insights_spec.md`](../chess_insights_spec.md);
the reasoning behind each decision is in [`PROJECT_TRACKER.md`](../PROJECT_TRACKER.md).

Data snapshot: games up to **2026-10-16** (11,482 games). Newer games are imported
but not yet reflected in these numbers.

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

2. **Long think → blunder: a real association, but not a personal
   signature.** A "long think" is a move taking more than 8x *that
   player's own* median time in the game. After one, the probability of
   blundering within my next 3 moves is 17.5%, vs a 4.6% baseline — a
   **3.83x ratio** (95% CI 3.61–4.05; 6,362 long thinks, 1,116 followed
   by a blunder). The control group decides what that means: the same rule
   applied to my opponents' moves gives **3.97x** (3.72–4.21, n = 4,927),
   and the paired game-level bootstrap puts the difference at **−0.14
   (−0.44 to +0.19)** — no detectable difference. So "recalculate, then
   blunder" is what hard positions do to players at my level, not a habit
   of mine. The reading rule was written in the spec before the intervals
   were computed (the point estimates had already been seen). Definition
   history: earlier versions of this finding (3.75x) measured long thinks
   against the median of *both* players, which flags the slower player more
   often — on the same games it gave my opponents 6,359 long thinks to my
   4,590; measured against each player's own pace, the counts flip to
   4,927 vs 6,362.

3. **Conversion: 1 in 4 winning positions not won — yet better than my
   opponents from the same point.** Of 4,658 blitz/rapid games where I
   reached a ≥80% win probability, I won **72.6%** (95% CI 71.4–73.8);
   median plies to finish 30 overall, 24 for converted wins. Read alone,
   that looks like a leak, and earlier versions of this README called it
   one. The control says otherwise: from the mirrored threshold in the same
   games, my opponents convert only **64.8%** (63.5–66.0, n = 4,829), a
   difference of **+7.9 points (+5.7 to +10.0)** by paired bootstrap. Not a
   relative weakness. The limit: my rate depends on my conversion and on my
   opponents' defence, theirs on their conversion and on *my* defence — and
   finding #6 shows my defence is strong — so the gap mixes both skills and
   does not prove a better conversion technique. In absolute terms, the 27%
   of unconverted games remain the largest pool of points I can still win.

4. **Time pressure comes too late to explain the mid-game blunder peak —
   and I reach it less often than my opponents.** _(Redefined and re-run on
   2026-11-15, on 11,620 eligible games after the out-of-scope cleanup;
   protocol in the spec, "M4 — Time-pressure onset".)_ A player is in time
   pressure once their clock after a move falls below **10% of the base
   time**; the onset is the first such move, per player and per game.
   - **Confirmatory (me vs opponents, same games):** the clock goes below
     10% in **14.4%** of my blitz/rapid games (95% CI 13.6–15.3) vs
     **21.1%** for my opponents (20.1–22.1): difference **−6.7 points
     (−7.6 to −5.6)** by paired game-level bootstrap. Fewer of my games
     reach time pressure than my opponents' — a side-specific result.
   - **Against the blunder peak:** the reference peak, derived from the M1
     curve itself (games of 40+ plies, buckets of ≥ 100 of my moves), is the
     15–19 bucket, i.e. **move 19**. When I do reach time pressure, the
     median onset is **move 42** (41–42), far after it — so the clock
     cannot be what drives the mid-game peak; calculation load peaks first.
   - **Exploratory:** the direction holds at every sensitivity threshold
     (20% of base; 10, 20 and 30 seconds) and in every time control with at
     least 100 games. In 10+0 I go below 10% in only **7 of 1,327 games**.
     When time pressure does happen, it starts at the same move for both
     sides (median difference about one move, CI 0 to 2): the gap is how
     often, not how early. Of my blunders, 5.0% are played with the clock
     already below the threshold after the move (opponents 7.3%) —
     descriptive, and it includes blunders whose own long think crossed
     the threshold.
   - **Limits:** reaching low time less often does not prove better clock
     management: a faster general pace, or opponents burning time while
     trying to convert against my defence (finding #6: half my saves end on
     time), would produce the same gap. Observational.
   - **Definition history:** earlier versions of this finding quoted "move
     67", the move at which the *median clock of the games still running*
     fell below a hard-coded 36 s — a different quantity, dominated by long
     games — and the spec, the config and the code disagreed on the
     threshold (20% / 10% / 36 s). Fixed on 2026-11-14.

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

6. **Swindle profile: a real edge from lost positions — and half of it is
   the clock.** A game counts as "lost" once my win probability stays at or
   below 20% for 3 full moves (6 consecutive plies); 2,938 games qualify.
   From there I score **24.4%**, against an expected **14.0%** (the average
   win probability at the start of the lost stretch) — **+10.4 points**.
   That gap only means something next to a control group, so the exact same
   rule was applied to my opponents' lost stretches: they beat the same
   formula by just **+2.3 points** (n=2,722). The formula's calibration
   (it is fitted on Lichess players overall, not on my rating) explains a
   couple of points at most; the remaining **~8 points are mine**, against
   rating-matched opponents. How the saves happen: a genuine turning point
   (an opponent move lifting me from ≤20% to ≥40%) appears in only 25.4% of
   lost games, and when it does I save **66.4%** of them; without one I
   still save 14.9% — and **64% of those end on time**. Across all saves
   from lost positions, about **half (48%) end on the clock** (38% even
   when a turning point happened). The identity stat, quantified: I stay in
   lost games long and fast enough for winning opponents to crack or flag.
   Measured in win% swings and game results, not with the blunder label —
   below 20% a "blunder" is impossible by definition, so the spec's
   original label-based version would have read zero by construction.

7. **Time of day and session position: no fatigue, no "tilt queue" —
   just a little momentum after wins.** All 11,482 games, local time
   (Europe/Paris), a new session after a 15-minute break measured from the
   estimated end of the previous game. Every group is read as score minus
   the Elo-expected score from the rating gap, so a stronger late-night pool
   cannot pass for fatigue. Reference point: I beat the Elo expectation by
   **+2.2 points overall** (ratings lag an improving player), so +2 is
   "normal", not 0. **Hour of day: no detectable effect** — hourly gaps
   spread from about −3.5 to +7.5 points, but each hour holds only ~100–450
   games (a ±4–8 point 95% interval), and the blunder rate stays flat at
   ~4–5.5%. **Fatigue: none** — game 1 through game 10+ of a session moves
   between −1.5 and +4.4 points with no trend, blunder rate 4.6–5.0%
   throughout. **Previous result: a small effect.** After a win +3.6 points
   (n=4,858), first game of a session +1.2 (n=1,854), after a loss +0.8
   (n=4,277); blunder rate 4.7% after a win vs 5.0% after a loss. The
   win-vs-loss gap (~2.8 points) is about 2.8 standard errors, so likely
   real but small — and since "after a loss" sits at the same level as a
   fresh session start, the honest reading is *I play slightly better after
   a win*, not *I collapse after a loss*. Observational: a loss often
   follows a hard game, and a win raises my rating and therefore the
   opponent pool.

## Bullet findings

_Same ~9,200-game vintage, `perf_group == "bullet"` in `metrics.py`,
2,512 qualifying games for M3. M5 (opening surgery) is not split by time
control — a deliberate choice, see `PROJECT_TRACKER.md` — so bullet-specific
opening numbers don't exist yet._

- **Sharper, earlier blunder peak than blitz/rapid.** ~11.7% blunder rate
  around move 15 (vs ~6.8% for main at the same point) — despite bullet
  being my strongest time control by peak rating on Lichess (2,170, against
  a 1,953 peak in blitz), the raw blunder rate at the critical early-middlegame
  point runs noticeably higher here. Noisier than the main-line curve
  (smaller per-bucket samples within bullet's own duration terciles), so
  the magnitude is a rougher estimate than finding #1's.
- **The long-think association holds under bullet's compressed clock —
  and is not mine there either.** 3.48x (95% CI 2.81–4.21, n = 804 long
  thinks, own-pace definition; 3.84x under the old shared-median one) vs
  2.99x for my opponents (2.54–3.47); difference +0.50 (−0.30 to +1.33),
  no detectable difference, same reading as finding #2.
- **Conversion is worse in bullet than in blitz/rapid, but still better
  than my bullet opponents.** 67.3% (n = 2,518) vs 72.6% for main —
  despite the higher peak rating in bullet, winning positions are thrown
  away more often here. From the same mirrored threshold, my bullet
  opponents convert 58.0% (56.2–59.8): **+9.3 points (+6.4 to +12.3)** in
  my favour, with the same conversion/defence limit as finding #3. Median
  plies to finish: 32 (28 for converted wins), both a touch longer than
  main's 30/24.
- **Time pressure: far more frequent than in blitz/rapid, and still less
  frequent for me than for my opponents** _(2026-11-15 run, M4 protocol)_.
  The clock goes below 10% of the base in **37.8%** of my bullet games
  (36.4–39.2) vs **48.1%** for my opponents (46.7–49.5): **−10.3 points
  (−12.0 to −8.6)**. Median onset **move 37** (36–37), against an M1
  reference peak at move 19 (15–19 bucket) — same reading as finding #4.
  The earlier "move 54" figure came from the old median-clock method.

## What Actually Predicts My Blunders? — V2 Model

The LightGBM model outperforms the logistic-regression baseline, and the strongest signal identified so far is an elevated error rate when my win probability is around 70–80%. Centipawn diagnostics and a SHAP dependence plot support this pattern. These findings are observational and do not establish causality.

### Scope

The model predicts whether one of my moves will be labelled a blunder, using only information available before that move.

The dataset includes games above my rating thresholds: **Lichess blitz (1700), Lichess bullet (1950), and Chess.com bullet (1650)**. Rapid is excluded because performance has not yet reached a stable plateau in that format.

The model does not use player ratings. `my_rating` and `rating_diff` were removed in November 2026: chess.com stores post-game ratings (Lichess stores pre-game ones), so on chess.com games these two features carried the result of the game being predicted. A pre-registered ablation over five training seeds measured the cost of removing them at −0.0006 PR-AUC on average, far inside the ~0.01 seed-to-seed noise.

The evaluation uses my own moves after ply 10, restricted to positions where my win probability is at least 20%. Below 20%, a drop of 20 percentage points is impossible by definition, so these moves cannot be labelled as blunders under the chosen threshold. The test set consists of my **426 most recent games** (data up to 2026-10-16), held out chronologically at game level.

### Model performance

The temporal test set contains **10,270 moves**, with a **6.5% blunder rate** (evaluation run `2026-11-14_17-03-16`).

- **LightGBM PR-AUC:** 0.199 (95% CI: 0.167–0.238), compared with 0.065 for a random-ranking reference.
- **Logistic-regression baseline PR-AUC:** 0.169.
- **Paired game-level bootstrap:** LightGBM's PR-AUC improvement over logistic regression is **+0.030** (95% CI: +0.012 to +0.049); LightGBM performs better in **all 1,000** resamples.
- **Calibration:** mean predicted risk is 7.4%, versus 6.5% observed.
- **Risk ranking:** the top risk decile has a 22.3% blunder rate (about **3.4×** the test-set base rate), while the bottom decile has a 0.5% rate.

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

The global SHAP ranking also identifies evaluation volatility (0.49), `eval_before_move` (0.29), `clock_before_move_pct` (0.21), and material imbalance (0.21) as the four leading features.

The finding is also relevant to conversion: the M3 game-level analysis found a **72.6% conversion rate** from positions at win% ≥ 80 (**n = 4,658** main games, blitz and rapid combined) — 7.9 points better than my opponents from the same threshold (finding #3). This is related evidence, not the same population or exact win-probability bucket as the 70–80% move-level analysis.

### Important caveats

- **Win-percentage labels flatten near the extremes.** In lost or nearly won positions, a large centipawn loss can still produce a relatively small change in win probability. The centipawn diagnostic therefore finds substantial error rates in extreme buckets even when the win%-based blunder label marks fewer moves as blunders.
- **Training-seed variability:** the PR-AUC of the same model varies by approximately 0.01 across random seeds. This is a meaningful source of uncertainty when interpreting a performance difference of +0.030 over the baseline (every shift since October — new games, then the removal of the rating features — stayed within this noise).
- **The test set is limited to 426 recent games.** Although it contains 10,270 moves, these moves are clustered within games and are not independent observations. Results may also reflect the characteristics of this recent period.
- **The 50–60% bucket has a low large-error rate.** This may reflect quieter positions, including opening theory or dead-drawn endgames, rather than a general psychological effect. It is a candidate pattern, not a proven explanation.
- **The late-game 40–50% bucket may be riskier.** In moves 31+, its large-error rate is 18.9% (n = 1,587; win%-label blunder rate 17.1%). This is an observational subgroup result and should be treated cautiously.
- **Uncertainty matters.** For a rate near 19% with a sample of about 900 moves, a simple binomial 95% interval is roughly ±2.6 percentage points. This ignores within-game correlation, so it may understate uncertainty.
- **Not yet controlled against my opponents.** This zone is measured on my moves only. A one-off check outside the pipeline found my opponents' large-error rate in their own 70–80% zone to be about the same as mine (17.5% vs 17.4%, all history, blitz/rapid). If the pipeline confirms it, the zone describes where blunders concentrate at my level rather than a personal weakness — the same reading findings #2 and #3 received once controlled.
- **The evidence is observational.** These analyses identify associations and predictive patterns; they do not establish why these errors occur.

### Current conclusion

The V2 model has measurable ranking value over the logistic-regression baseline. The most actionable finding is the elevated rate of large errors when I am at **70–80% win probability**: it persists across move-number bands, appears in the centipawn diagnostic, and is supported by a positive SHAP dependence pattern while accounting for clock pressure, material imbalance, and evaluation volatility.

This is the strongest converging signal identified so far, but it remains an observational result rather than proof of a causal mechanism.
