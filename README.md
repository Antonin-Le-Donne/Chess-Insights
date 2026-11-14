# Chess Insights

I analyse my own online chess games (about 12,000 of them) to find out **when, where
and why I make big mistakes**. For each pattern, I also check whether the mistake
is really mine or just what happens to every player at my level.

A data pipeline downloads my games and a chess engine grades every move. A
dashboard and a machine-learning model then turn those grades into answers.

**Stack:** Python · pandas · python-chess · Stockfish · Streamlit · Plotly ·
scikit-learn · LightGBM · SHAP

## How it works

```
Lichess + chess.com APIs → parse games → Stockfish grades every move → metrics → dashboard
                                                                     ↘ blunder-risk model
```

1. **Download** all my games from both sites. After the first run, only new
   games are fetched.
2. **Grade** every position with the Stockfish engine and turn each grade into
   a win probability.
3. **Measure** seven metrics, each testing one idea about my mistakes.
4. **Compare** with a control group: the same rule, applied to my opponents in
   the same games.
5. **Show** everything in a Streamlit dashboard, with filters and links that
   open each game.
6. **Predict** the risk of a blunder before each move, with a LightGBM model.

## Key terms

- **Win probability:** my chances of winning a position, from 0 to 100%,
  computed from the engine's evaluation.
- **Blunder:** a move that lowers my win probability by **20 points or more**.
- **Control group:** my opponents in the same games, measured with exactly the
  same rule. If they show the same pattern as me, the pattern is not personal.
- **95% confidence interval (CI):** the range the true value very likely falls
  in, given how much data there is. If the interval for a difference contains
  0, there is no detectable difference.
- **Main vs bullet:** blitz and rapid are analysed together ("main"). Bullet,
  the fastest games, is always analysed separately.

## Main findings

_Blitz + rapid games, data up to 2026-10-16. The full numbers, methods and
caveats are in [docs/findings_details.md](docs/findings_details.md)._

| # | Question | Short answer | Compared with my opponents? |
|---|---|---|---|
| 1 | When do I blunder most? | Around moves 15–20, not at the end of the game | Not yet |
| 2 | Do long thinks lead to blunders? | Yes, about 4× more often, but the same is true for my opponents | Yes: shared, not personal |
| 3 | Do I convert winning positions? | 73% of the time, more often than my opponents (65%) | Yes: better than them |
| 4 | Is the clock behind the mistakes? | No. My clock only runs really low around move 67 | Not yet |
| 5 | Which opening hurts me? | The Italian Game as Black | No |
| 6 | Do I save lost games? | Yes, clearly more than my opponents do | Yes: a personal strength |
| 7 | Do time of day or tiredness matter? | No. I play slightly better after a win | Against the expected score from ratings |

**1. Blunders peak in the middle of the game.** My blunder rate climbs from
about 0.3% of moves at the start to about 6.5% around moves 15–20, then falls
to 1–2% in long endgames. So it is not tiredness building up over the game. My
guess is that the middlegame is simply the most complex phase, but I have not
tested this yet. Part of the late decline is a measurement effect: near a won
or lost position, win probability barely moves, so blunders are harder to
detect.

**2. Long thinks come before blunders, for everyone.** A "long think" is a move
that takes more than 8 times that player's usual time in the game. After one, I
blunder within my next 3 moves 17.5% of the time, against 4.6% normally: a
**3.8× ratio**. My opponents show the same thing (4.0×), and the gap between us
is not detectable. So this is what hard positions do to players at my level,
not a habit of mine.

**3. I win 73% of my winning positions, which is better than my opponents.**
Once I reach an 80% win probability, I go on to win 72.6% of those games. On
its own, that looks like a leak. But my opponents, from the same point, only
win 64.8%: **+7.9 points** for me (CI +5.7 to +10.0). One limit: their rate
also depends on my defence, which is strong (finding 6). The 27% of games I
don't convert are still the biggest pool of points I can win back.

**4. Time trouble and blunders are two separate problems.** My clock only runs
really low around move 67, long after the blunder peak of finding 1. My
mid-game mistakes are not caused by a lack of time.

**5. The Italian Game as Black is my weakest opening.** I score 45.0% there over
200 games, where the score counts 1 for a win and ½ for a draw. My opponents in
those games are slightly weaker than usual, which makes the number more
notable, not less. The Sicilian, which first looked like the problem, is fine:
55.9% as Black and 51.8% when I face it as White. Lesson learned: with fewer
than 50 games, an opening result is a hypothesis, not a finding.

**6. I save lost games, and about half of those saves come from the clock.** A
game counts as lost once my win probability stays at or below 20% for 3 full
moves. From there, I score 24.4%, while the engine-based expectation was 14.0%.
My opponents beat the same expectation by only 2.3 points, so about **8 points
are a personal edge**. Almost half of these saves end with the opponent running
out of time.

**7. No time-of-day or tiredness effect, and a little momentum after wins.**
Each result is measured against the score my rating predicts. The hour of the
day and the number of games played in a row make no detectable difference. I do
slightly better after a win than after a loss (about 3 points), which is real
but small.

### Bullet

- The blunder peak is earlier and higher: about 11.7% of moves around move 15.
- Long thinks: 3.5× for me against 3.0× for my opponents, with no detectable
  difference.
- Conversion: 67.3% for me against 58.0% for my opponents (+9.3 points).
- The clock runs low from move 54, as expected with a one-minute game.

## Predicting my blunders: the V2 model

**Question:** before I play a move, can a model tell how likely I am to
blunder? It only uses information available before the move: position, clock,
recent eval swings, my recent pace, and so on.

It is trained on games played at my current level (Lichess blitz ≥ 1700, Lichess
bullet ≥ 1950, chess.com bullet ≥ 1650). It is tested on my **426 most recent
games** (10,270 moves), which it never saw during training.

- **It ranks risk well.** Among the 10% of moves it flags as riskiest, 22.3% are
  blunders, 3.4× the average rate of 6.5%. Among the 10% it rates safest, only
  0.5% are.
- **It beats a simpler model.** It scores a PR-AUC of 0.199, against 0.169 for a
  logistic regression and 0.065 for random guessing. PR-AUC measures how well a
  model puts the rare blunders at the top of its list. The gain is +0.030 (CI
  +0.012 to +0.049): real, but modest.
- **Strongest signal: the 70–80% zone.** When I am clearly better but have not
  won yet, I make large errors (about 2 pawns or more) 18.9% of the time,
  against about 10% in neighbouring positions. This holds at every stage of the
  game. The model's SHAP explanations also raise the risk in that zone on their
  own. Not yet checked inside the pipeline: a quick one-off test found the same
  rate for my opponents (about 17.5%). If confirmed, this zone is where everyone
  at my level slips, not a weakness of mine.
- **Player ratings are not used.** chess.com records ratings after the game, so
  they would leak the result. Removing them cost nothing measurable.

## Honest limits

- These results show associations, not causes.
- The "blunder" label is less sensitive near won or lost positions, where win
  probability barely moves.
- The model's results move by about 0.01 PR-AUC from one training run to the
  next, so its +0.030 gain should not be read as a precise figure.
- Findings 1, 4 and 5 are not yet compared with my opponents.

## Status

**Working, not complete.** The pipeline, the dashboard (seven metrics, filters,
links to games) and the V2 model work. Games are imported up to 2026-11-14
(12,042 games); the numbers above still come from the 2026-10-16 snapshot.

Next steps:

- an opponent control for the 70–80% zone, inside the pipeline;
- one final refresh of every number on the full data, including retraining the
  model;
- an end-to-end run from a fresh clone (Definition of Done v1);
- optional: a per-game risk timeline.

Full specification: [chess_insights_spec.md](chess_insights_spec.md). Decisions,
bugs and their fixes: [PROJECT_TRACKER.md](PROJECT_TRACKER.md).

## Setup

Requires **Python 3.11+** (`tomllib`). Every command runs from the repository root:
the scripts and the dashboard use paths relative to it.

1. **Install the dependencies** (versions are pinned):

   ```
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install Stockfish.** The engine is not in the repository (`stockfish/` is
   git-ignored). Download it from [stockfishchess.org](https://stockfishchess.org/download/)
   and place the binary at `stockfish/stockfish.exe`. On macOS/Linux, point
   `STOCKFISH_PATH` in `src/evaluate.py` to your binary instead.

3. **Set the usernames** in `config.toml`: `username` (Lichess) and
   `chesscom_username`.

4. **Optional: a Lichess API token.** Without one the download still works, only
   slower (20 games/s instead of 60). Create a free personal token on Lichess
   and put it in a `.env` file at the root, which is git-ignored:

   ```
   LICHESS_TOKEN=your_token_here
   ```

## Run

```
python -m src.fetch            # Lichess; incremental after the first run (--full for everything)
python -m src.fetch_chesscom   # chess.com monthly archives, cached in data/raw/chesscom/
python -m src.parse            # appends new games only; never rewrites existing rows
python -m src.evaluate         # local Stockfish on games without Lichess server evals
python -m src.features         # win%, move quality, phases; stops on a broken label
streamlit run app/dashboard.py
```

- `evaluate` is the slow step: depth 20 with 5 parallel workers took about
  2 hours per 1,000 games on my machine. It is resumable — stopping it with
  Ctrl+C keeps the games already analysed, and the next run skips them.
- Afterwards, the same chain imports new games: each step only processes what
  is new.
- **V2 model:** `python -m src.model.train`, then `python -m src.model.evaluate_model`
  (results in `model_evaluation/<timestamp>/`, read by the dashboard's model page).
