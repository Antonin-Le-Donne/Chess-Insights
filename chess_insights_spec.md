# Chess Insights — Personal Blunder & Time Analysis Tool

Project specification for a two-version build. Owner: Antonin. All code, comments, and docs in English.

**Goal.** Build a data pipeline + dashboard that tests specific hypotheses about my own chess failures (mental fatigue curve, time-management leak, greed in winning positions, Sicilian mismatch), then a supervised ML model that predicts my personal blunder-risk contexts. This is NOT a Lichess Insights clone: every metric below answers a question Lichess cannot answer because it doesn't know my hypotheses.

**Why this project (portfolio framing).** "I built a tool that tests hypotheses about my own cognitive failure modes at chess, then trained a contextual risk model on the results." Full pipeline: API ingestion → parsing → feature engineering → statistics → supervised ML → dashboard.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | python-chess, pandas, scikit-learn ecosystem; no TS equivalent |
| Chess parsing / engine | `python-chess` | PGN parsing, board manipulation, UCI engine control |
| Engine | Stockfish (local binary, free) | Evaluate games that lack server analysis |
| Data | `pandas`, Parquet files (no DB needed for v1) | Simple, fast, versionable |
| Dashboard | `streamlit` + `plotly` | Interactive dashboard in ~100 lines, free, local |
| ML (v2) | `scikit-learn` (baseline), `lightgbm` (main model) | Standard supervised stack |
| Config | `.env` for Lichess token, `config.toml` for thresholds | Keep thresholds tunable |

## Repo structure

```
chess-insights/
├── config.toml               # thresholds, username, time controls to include
├── .env                      # LICHESS_TOKEN (optional but raises rate limits)
├── data/
│   ├── raw/                  # downloaded PGN/NDJSON, never modified
│   ├── interim/              # parsed per-move records before eval completion
│   └── processed/            # final moves.parquet + games.parquet
├── src/
│   ├── fetch.py              # Lichess API download
│   ├── parse.py              # PGN -> per-move records
│   ├── evaluate.py           # local Stockfish pass for missing evals
│   ├── features.py           # derived columns (phase, time features, win%)
│   ├── metrics.py            # all v1 analytics functions (pure, testable)
│   └── model/                # v2 only
│       ├── dataset.py        # build X, y with game-level split
│       ├── train.py
│       └── evaluate_model.py
├── app/
│   └── dashboard.py          # Streamlit entry point
├── tests/
└── README.md
```

---

# V1 — Data pipeline + hypothesis-driven analytics

## 1. Data ingestion (`fetch.py`)

Two sources, ~11.7k games total (~5k Lichess, ~6.7k chess.com before bullet
filtering). Implement Lichess first, validate the full pipeline end-to-end on it,
then add the chess.com fetcher as a second pass — the schema supports both from
day one via the `platform` column.

### Source A — Lichess

**Endpoint:** `GET https://lichess.org/api/games/user/{username}`

Key query parameters:
- `clocks=true` — emit clock comments per move (`[%clk 0:04:12]`). **Required**, this is the core of the time analysis.
- `evals=true` — emit eval comments per move (`[%eval 0.52]`), **only present for games already analyzed by Lichess server**. Expect a minority of games to have them.
- `opening=true` — ECO code + opening name in headers.
- `perfType=blitz,rapid,classical` — exclude bullet (instinct-only games pollute the fatigue analysis; can be re-added later as a comparison set).
- `rated=true`
- `since` / `until` — millisecond timestamps, useful for incremental fetches.
- Accept header `application/x-chess-pgn` (PGN stream) is simplest with python-chess.

Notes:
- Streamed response; write to `data/raw/lichess/games_{date}.pgn`. Do not parse during download.
- With an API token: 60 games/s download rate; without: 20 games/s. Token is a free personal token, no OAuth app needed.
- Idempotent incremental fetch: store last fetch timestamp, use `since` on next run.

### Source B — chess.com (add AFTER the Lichess pipeline works end-to-end)

**Endpoint:** `GET https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}`
(monthly archives; list available months first via `.../games/archives`).

- Free public API, no token, no paid account needed.
- Returns JSON; each game object contains a `pgn` field with `[%clk]` tags →
  the whole time analysis works identically.
- **No evals ever** (chess.com website analysis is not exposed through this API),
  so 100% of these games go through the local Stockfish pass.
- Etiquette: identifiable User-Agent header, sequential requests (no parallel
  hammering), cache each month to `data/raw/chesscom/{YYYY}_{MM}.json` and never
  re-fetch a completed past month (immutable).
- Filter to rated blitz/rapid/classical at parse time (`time_class` field);
  bullet excluded, same rule as Lichess.

### Cross-platform rules

- `platform` column ("lichess" / "chesscom") in BOTH parquet files, from day one.
- Ratings are NOT comparable across platforms (~1850 chess.com ≈ ~2100 Lichess).
  Any rating-based stat, filter, or model feature must be platform-scoped or use
  a per-platform normalization. Move-quality metrics (M1–M6) merge freely — a
  blunder is a blunder.
- Free extra analysis once both sources are in: do my failure patterns differ by
  platform? (dashboard filter makes this a zero-cost question)

## 2. Parsing (`parse.py`)

Use `chess.pgn.read_game()` in a loop. For each game, walk the mainline and emit one record per ply.

**games.parquet** (one row per game):

| column | type | source |
|---|---|---|
| platform | str | "lichess" / "chesscom" |
| game_id | str | Site/Link header (URL suffix); prefix with platform to guarantee uniqueness |
| date_utc | datetime | UTCDate + UTCTime |
| time_control | str | e.g. "300+0" |
| base_seconds / increment_seconds | int | parsed from TimeControl |
| perf_type | str | blitz / rapid / classical |
| my_color | str | white/black (match username) |
| my_rating / opp_rating | int | Elo headers |
| eco / opening_name | str | headers |
| result_for_me | float | 1 / 0.5 / 0 |
| termination | str | Normal / Time forfeit / ... |
| n_plies | int | |
| has_server_evals | bool | any `[%eval]` found |

**moves.parquet** (one row per ply — but analysis focuses on MY moves):

| column | type | notes |
|---|---|---|
| platform | str | "lichess" / "chesscom" |
| game_id, ply, move_number | | ply = half-move index from 1 |
| is_my_move | bool | |
| san / uci | str | |
| clock_after | float | seconds, from `[%clk]` |
| time_spent | float | derived, see below |
| eval_cp | float | centipawns from White POV; mate mapped to ±10000 |
| eval_source | str | "lichess" or "stockfish" or null |
| fen_before | str | needed for v2 features and re-evaluation |

**time_spent derivation** — the subtle part:
```
time_spent(ply n) = clock_after(ply n-2, same side) - clock_after(ply n) + increment
```
First move of each side has no previous clock: treat as `base - clock_after + increment`. Clamp negatives to 0 (berserk/edge cases). Validate on a handful of games by hand before trusting it.

## 3. Evaluation completion (`evaluate.py`)

For games where `has_server_evals == false`, run local Stockfish via `chess.engine.SimpleEngine`:
- Fixed limit per position, e.g. `depth=14` or `time=0.05` (choose ONE and keep it constant across the whole dataset — mixed depths bias eval_delta; log the choice in the tracker Decisions).
- Evaluate every position in the game once, store into `eval_cp` with `eval_source="stockfish"`.

**Compute budget (the real one):** ~3.4k unanalyzed Lichess games + ~6.7k chess.com
games ≈ 10k games ≈ 700k positions. At 0.1s/position single-threaded that is ~20 hours,
not 3. Mitigations, all of them:

1. **Parallelize**: N worker processes, each owning its own Stockfish instance
   (engines are single-session; never share one engine across workers), splitting
   the game list. 4–6 workers on a modern Ryzen ≈ ÷4–6 wall time → one long night.
2. **Prioritize**: evaluate rated blitz+rapid from the last ~2 years first; older
   games and classical in later background runs. Dashboard runs on partial
   coverage from night one — the Overview page displays eval coverage % so partial
   data is never mistaken for complete data.
3. **Resumable is mandatory, not nice-to-have**: track evaluated game_ids
   (e.g., a done-list file or an `evaluated` flag), skip on restart, safe to
   Ctrl+C at any time.

## 4. Derived features (`features.py`)

Applied to moves.parquet:

- **win_pct** — convert centipawns to win probability with Lichess's formula:
  `Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)`
  where cp is from the mover's perspective. All quality metrics use win% deltas, not raw centipawns (a 100cp drop at +8 is noise; at +1 it's a game-changer).
- **win_pct_delta** — win% after my move minus win% before, from my perspective. Negative = my move lost ground.
- **move_quality** — categorical from win_pct_delta:
  - blunder: drop ≥ 20 win% points
  - mistake: drop ≥ 10
  - inaccuracy: drop ≥ 5
  (thresholds in config.toml; these mirror Lichess conventions, tune later)
- **phase** — opening: ply ≤ 20 or until first non-book-like criterion; middlegame; endgame: ≤ 6 non-pawn pieces total (simple material rule, keep it deterministic).
- **time_pressure** — clock_after < 20% of base time.
- **long_think** — time_spent ≥ threshold (default 40s blitz-adjusted: use `time_spent > 8 × median_time_spent(game)` so it scales across time controls).
- **eval_state** — from my perspective: winning (win% ≥ 80), better (65–80), equal (35–65), worse (20–35), lost (< 20).

## 5. Core metrics (`metrics.py`) — the calculations that ARE the project

Each function: pure, takes DataFrames, returns a DataFrame/dict. One dashboard view per metric.

### M1 — Fatigue curve (hypothesis: mental battery drains → accuracy drops with game depth)
- Mean win_pct_delta (and blunder rate) of MY moves, bucketed by move_number (bins of 5), split by perf_type.
- Overlay: same curve split by game duration terciles.
- **What to look for:** an inflection point. If blunder rate doubles after move 30, that's the measurable version of "je claque".

### M2 — Time→blunder signature (hypothesis: verbal recalculation burns clock, then errors follow)
- P(blunder within next 3 of my moves | long_think on current move) vs baseline P(blunder).
- Report as a ratio with counts (not just percentages — small-n cells lie).
- Secondary: scatter time_spent vs win_pct_delta on the same move.
- **What to look for:** ratio meaningfully > 1 confirms the "forgot my lines, recalculated, tilted" signature. This is the direct measurement of the Clock Time Usage 1/7 problem.

### M3 — Greed / conversion metric (hypothesis: winning position → soul mode → throw)
- For each game where my win% first reaches ≥ 80 ("winning point"):
  - plies_to_finish = n_plies − winning_point_ply
  - did_convert = result_for_me == 1
  - max_drawdown = lowest win% reached after the winning point
- Outputs: conversion rate; distribution of plies_to_finish (converted vs thrown games); count of games that went winning → equal-or-worse and back.
- **The tracking stat:** median plies_to_finish per month. This is the number to drive down.

### M4 — Clock burn map
- Median + p90 time_spent by move_number; median clock_after by move_number, split by perf_type.
- Flag pattern: "slow start → time pressure ≥ move X" vs "even burn".
- Cross with M1: is the fatigue inflection co-located with time pressure onset, or independent? (This distinguishes "tired brain" from "rushed clock" — different fixes.)

### M5 — Opening surgery table (hypothesis: Sicilian mismatch)
- Group by opening family (ECO letter + first 2 chars, plus a manual mapping for Sicilian/French/Dutch/Italian/Catalan/QG), split by color.
- Per group: n_games, score %, mean win% at ply 20 (did I leave the opening worse?), blunder rate in the first 20 plies.
- **What to look for:** Sicilian rows red on both colors (confirming the insight screenshot with per-move evidence), Dutch green, and specifically WHERE the Sicilian damage happens — out of the opening already worse, or fine at ply 20 then collapsing (different problems: theory gap vs middlegame plan gap).

### M6 — Swindle profile (the strength, quantified)
- Score % from lost positions (win% ≤ 20 at any point): win/draw/loss split.
- Opponent blunder rate when I'm lost vs when I'm winning (are opponents cracking, or am I setting traps? proxy: my move quality while lost).
- Keep this view: it's the identity stat, and watching it stay high while M3 improves proves improvement isn't costing the strength.

### M7 — Time-of-day / session position (optional, cheap)
- Score and blunder rate by hour of day and by game index within a session (games < 15 min apart = same session). Detects "tilt queue" behavior.

## 6. Dashboard (`app/dashboard.py`)

Streamlit, one page per metric M1–M7 plus an Overview page with: games count, date range, eval coverage %, current month's M3 tracking stat vs last month. Filters in sidebar: perf_type, date range, color, opening family. Charts with plotly (hover = drill down to game URL: `https://lichess.org/{game_id}`).

## V1 milestones

1. **Week 1:** fetch + parse → moves.parquet with clocks. Sanity-check time_spent on 5 games manually.
2. **Week 2:** Stockfish evaluation pass (overnight runs) + features.py. Eval coverage report.
3. **Week 3:** metrics M1–M5 + minimal dashboard.
4. **Week 4:** M6–M7, polish, README with findings written up (the findings ARE the deliverable).

---

# V2 — Personal blunder-risk model

**Reframe (critical):** the model does not predict chess moves and does not evaluate positions — Stockfish already did that in v1. It predicts MY probability of blundering given the *context*. It models my failure conditions, not chess. This is why ~100–150k of my own moves are enough data, where a move-playing model would need millions of games.

## Target

Binary classification, one sample per MY move (excluding opening book plies ≤ 10):
- `y = 1` if move_quality == blunder on this move (start simple: this move only; "within next 3 moves" variant as a second experiment — it leaks less actionable signal but matches the M2 framing).

Expect heavy class imbalance (~2–5% positives). Do not use accuracy as a metric, ever.

## Features (all available BEFORE the move is played — no leakage)

Contextual:
- move_number, ply, phase (one-hot)
- clock_after previous move (absolute + % of base), increment flag, time_pressure flag
- time_spent on my previous move, rolling mean of my last 3 time_spent, long_think flag on previous move
- eval_cp before move (my perspective), win_pct, eval_state (one-hot)
- eval volatility: std of win% over last 6 plies (sharp game proxy)
- material count, material imbalance, queens on board flag
- opening family (target-encoded or one-hot top-10 + other)
- my rating, rating difference, perf_type
- session features from M7 if implemented: game index in session

Position complexity proxies (cheap, from python-chess):
- number of legal moves
- number of my hanging pieces / attacked pieces (attackers > defenders count)
- checks available (to me / to opponent)

Explicitly EXCLUDED: anything computed from the move itself or after it (time_spent on the current move is borderline — it's known only after moving; exclude it from the main model, test it in a variant).

## Dataset construction (`dataset.py`)

- **Split by game_id, never by row** — moves within a game are correlated; row-level split = leakage and inflated scores.
- Temporal split preferred: train on oldest 70%, validate 15%, test on most recent 15%. This also answers "does my past predict my present?"

## Models

1. **Baseline:** logistic regression on scaled features. If LightGBM can't beat it clearly, the features are the problem, not the model.
2. **Main:** LightGBM (`class_weight` or `scale_pos_weight` for imbalance).
3. No deep learning here — tabular, small data, gradient boosting wins. (If the itch for a neural net remains, the separate Maia fine-tune project is the outlet.)

## Evaluation (`evaluate_model.py`)

- ROC-AUC and PR-AUC (PR-AUC is the honest one under imbalance).
- Calibration curve — the output should be usable as a real probability.
- **Feature importance + SHAP values: this is the actual payoff.** The ranking of what drives my blunder risk (time pressure? move depth? sharp Sicilians?) is the insight the whole project exists for, and the chart to show in an interview.
- Slice analysis: AUC per phase, per perf_type, per eval_state.

## V2 milestones

1. dataset.py + baseline logistic regression + honest metrics.
2. LightGBM + tuning (few iterations, no grid-search rabbit hole).
3. SHAP analysis + writeup of findings.
4. Dashboard page: "risk profile" — top risk factors + calibration plot; optional: per-game risk timeline overlay on the eval graph.

---

# Definition of done

- v1: `python -m src.fetch && python -m src.parse && python -m src.evaluate && streamlit run app/dashboard.py` works end-to-end on a fresh clone; README contains 5 written findings with numbers.
- v2: reproducible train script, test-set PR-AUC reported, SHAP summary chart, one paragraph in README: "what actually predicts my blunders".
