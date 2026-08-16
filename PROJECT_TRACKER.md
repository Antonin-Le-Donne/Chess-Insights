# Chess Insights — Project Log

Owner: Antonin. Lives in the repo. Execution checklist (milestones/issues) is
tracked separately in my personal notes; this file only records judgment:
decisions, blockers, findings, and results over time.
Reference for all definitions: `chess_insights_spec.md`.

---

## Decisions log

One row per non-obvious technical choice. If I can't fill "Why", the decision
isn't made yet.

| Date | Decision | Why |
|---|---|---|
| 2026-08-03 | Stockfish limit: `depth=20`, `time=5.0` safety cap | Benchmarked 14/16/18/20 on real positions; depth=20 with 5 parallel workers estimated ~4.4h, acceptable overnight. `time=5.0` added after one position hung during the real run (Ctrl+C'd, safety net prevents recurrence). |
| 2026-08-03 | Blunder threshold kept at default (20 win% drop) | Matches spec default; `mistake=10`, `inaccuracy=5` also kept, all in `config.toml [thresholds]`. |
| 2026-08-04 | Long-think definition kept at default (`> 8 × game median`) | `long_think_factor=8` in `config.toml`, not hardcoded. |
| 2026-08-04 | `time_pressure_pct` set to 0.10, not the spec's literal 20% | 20% felt too generous on blitz (36s on 180+0 didn't feel like real pressure); 10% chosen as a more honest threshold, tunable in config if wrong. |
| 2026-08-04 | `move_quality` computed for ALL moves, not just `is_my_move` | M6 (swindle profile) needs opponent move quality as a proxy for "opponent blunder rate" per spec's own parenthetical — masking it would make M6 uncomputable. Filtering by `is_my_move` happens at the metrics layer, not the features layer. |
| 2026-08-04 | `has_server_evals` = "ANY position has a server eval", not "ALL" | Discovered after 82 positions (1 per game, 82 games) stayed unevaluated post-`evaluate.py` run: those games had partial Lichess coverage (missing eval on their last ply, often right before a forced mate) but were skipped entirely since the game-level flag was `True`. Fixed with a targeted single-position Stockfish backfill script, not a full `evaluate.py` rerun. |
| 2026-08-04 | ECO family ranges for `ECO_FAMILY_MAP`: Sicilian B20-B99, French C00-C19, Dutch A80-A99, Italian C50-C59, Catalan E00-E09, Queen's Gambit D06-D69 | Cross-validated against own data (`opening_name` containing the family name for games in each ECO range: Sicilian 99.7%, French/Dutch/Italian 100%) and confirmed against an external source (Exeter Chess Club ECO list). |
| 2026-08-15 | Add bullet to `config.toml`'s `perf_types`, but keep it structurally separate from blitz/rapid everywhere in `metrics.py` (a `"main"` vs `"bullet"` split per metric, not a `"bullet"` value mixed into a generic per-perf_type split) | 2,229 bullet games exist on Lichess alone (more than blitz+rapid combined, 1,909), with more still to come from chess.com. Not a minor comparison set — it's the majority of the full game history once combined. Spec's own reasoning for excluding bullet ("instinct-only games pollute the fatigue analysis") still applies regardless of population size, and the player is reportedly stronger at bullet than blitz/rapid — a genuinely different regime worth its own dedicated M1-M5 pass rather than folding it into the existing blitz/rapid numbers. |

## Blockers

One row per blocker, updated in place when resolved. A resolved blocker with a
filled "How" column is a draft README finding — promote it.

| Date | Blocker | Resolved on | How |
|---|---|---|---|
| 2026-08-03 | Stockfish hung on one position during the parallel overnight run (0% CPU, no progress) | 2026-08-03 | Ctrl+C (safe, resumable design preserved prior progress), added `time=5.0` cap to `LIMIT`, relaunched — never recurred. |
| 2026-08-04 | 82 positions missing `eval_cp` despite `evaluate.py` reporting 100% done | 2026-08-04 | Root cause: `has_server_evals` is an "any", not "all", check — see Decisions log. Fixed via a standalone script evaluating just the 82 missing positions directly with Stockfish, no pipeline rerun. |
| 2026-08-04 | `perf_type` had inconsistent casing (`blitz`/`Blitz`) and a bogus `SuperBlitz` value | 2026-08-04 | Root cause: `parse.py` extracted `perf_type` from `Event` header text (`.split()[1]`), which has a different format for arena/tournament games (`"Daily SuperBlitz Arena"`) than regular rated games (`"rated blitz game"`). Patched `games.parquet["perf_type"]` directly by deriving it from `base_seconds`/`increment_seconds` instead (robust to header format). **`parse.py` itself still has the underlying bug** — will resurface on any future refetch; not yet patched in the source. |
| 2026-08-04 | M1 fatigue curve: possible survivorship bias (games ending early are, by construction, games where someone already blundered) | 2026-08-04 | Retested restricted to `n_plies >= 40` games, full curve to move 95. Not a survivorship artifact, but not simple fatigue either: blunder rate rises to a peak ~5.5-6.5% around move 30-40, then declines through the endgame (~2% by move 65). Reframed as middlegame complexity, not battery drain. Tail past move 80 too sparse to trust (small n). |
| 2026-08-04 | M5 opening surgery: possible Elo confound (stronger opponents may play the Sicilian more often, so the 47.6% "facing Sicilian as White" score may reflect opponent strength, not the opening) | 2026-08-04 | Checked mean `opp_rating - my_rating` per opening family. Sicilian ≈ -2.9, Italian ≈ -6.2 — opponents are *weaker* on average when playing these against me, not stronger. Confound doesn't explain the low scores away. Italian's n=47 remains too small regardless (±14pt CI), unrelated to this check. |

## Findings

Numbered, with numbers. These feed the README directly. (Full text in `README.md`.)

1. Fatigue curve: blunder rate ~0.4-0.5% (moves 0-4) → ~7% (moves 15-19, short games) — strong signal, survivorship bias not yet ruled out (see Blockers).
2. Time→blunder signature confirmed: 3.8x ratio (14.6% vs 3.9% baseline), n=1,240 long-think instances.
3. Conversion leak quantified: 74.5% conversion rate from win% ≥ 80 (n=1,155 games) — 1 in 4 "should-win" positions thrown. Median 27 plies to finish (24 for converted wins).
4. Time pressure (move ~56 onset) and the fatigue inflection (move 10-15) are far apart — likely two separate failure modes, pending #1's robustness check.
5. Opening mismatch hypothesis reversed, provisionally: Sicilian as Black scores 58.0% (n=182, good, not a weakness). Facing Sicilian as White scores 47.6% (n=185, not yet Elo-controlled). Facing Italian as Black scored 38.3% but n=47 gives a ±14pt 95% CI — downgraded to a hypothesis to watch, not a finding.

## Results over time (chess-side tracking)

The point of the tool. Fill monthly once M1–M3 are live. Candidate for promotion
to a README "Results" section once there are 2+ months of data.

| Month | Median plies_to_finish (M3) | Conversion rate from win% ≥ 80 | P(blunder \| long_think) / baseline (M2) | Fatigue inflection move (M1) |
|---|---|---|---|---|
| baseline (2026-08) | 27 | 74.5% | 3.77 | ~10-15 (tentative, pending survivorship retest) |

## Session log

One line per session, filled at the end of the session, not during. Answers
"where was I?" after a break.

| Date | Did | Next step |
|---|---|---|
| 2026-08-03 | Built `fetch.py`, `parse.py`, `evaluate.py` (resumable + parallel Stockfish pass), full overnight eval run on 1,909 games | Backfill 82 games with partial `eval_cp` gaps (`has_server_evals` false-positive), then `features.py` |
| 2026-08-04 | Fixed the 82-position eval gap, built `features.py` (win_pct, move_quality, phase, time_pressure, long_think, eval_state — with M6 unmasking fix), fixed `perf_type` casing/SuperBlitz bug, built `metrics.py` M1-M5, wrote README findings v1 | Commit everything (metrics.py + README.md + this file), close issues M1-M5, then M6/M7 + dashboard.py (Milestone 4) |