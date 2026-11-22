# Chess AI

A chess game written in Python (pygame) with a fast C++ alpha-beta engine behind it, an optional
neural-network evaluator trained on Lichess data, and tools to measure how strong it is.

- **Play** against the AI, against a friend, or against Stockfish.
- **C++ engine** (`engine/cpp/`): millions of nodes per second, searches 10+ plies in well under a second.
- **Thinks on your time:** in the strong setting the engine keeps searching while you choose your move (pondering).
- **Multi-threaded search and an opening book:** in the strong setting the engine searches with several threads
  (a quarter of your logical CPUs by default) and plays its first moves from an opening book.
- **Strength:** roughly **2600-2650** against Stockfish limited to a given Elo, at 0.1 s per move
  (benchmark games against Stockfish, see [Strength](#strength)). Search improvements since the first
  measurements gained much more in head-to-head matches (see [Comparing engine versions](#comparing-engine-versions))
  than they show against Stockfish.
- **Pure-Python fallback**: if the C++ engine has not been built, the game still works (just weaker and slower).
- **NNUE-style network** (`train_nnue.py`): trained from the Lichess evaluation database; optional.
- **Evaluation tools**: `evaluate_engine.py` (vs Stockfish), `match_engines.py` (one engine build or option set vs
  another, for testing changes), `tune_eval.py` (fits the evaluation weights) and `train_nnue.py match` (network vs hand-written eval).

## Quick start

```bash
pip install chess pygame-ce numpy
python main.py
```

This runs with the pure-Python engine. For the full-strength engine, build the C++ one (next section).

## Build the C++ engine

Needs **CMake 3.13+** and a C++17 compiler (Visual Studio 2022 with "Desktop development with C++",
MSYS2/MinGW `g++`, or `gcc`/`clang` on Linux/macOS). From the project root:

```bash
cmake -B build
cmake --build build --config Release
```

The executable is written to `engine/bin/chess_engine` (`.exe` on Windows), which is where the game
looks for it. When it is found, the terminal prints `AI: C++ engine ...` at startup.

- `--config Release` matters with the Visual Studio generator: without it you get a slow **Debug** build.
- After the first `cmake -B build` you only need `cmake --build build --config Release`. Re-run
  `cmake -B build` only when you change an option, the generator or the compiler.

| Option (add to `cmake -B build`) | Effect |
|---|---|
| `-DCMAKE_BUILD_TYPE=Debug` | `Release` (default), `Debug`, `RelWithDebInfo`, `MinSizeRel` (single-config generators) |
| `-DUSE_NATIVE=OFF` | portable binary instead of one tuned for this CPU (default ON) |
| `-DSTATIC_LINK=OFF` | MinGW only: don't link the C++ runtime statically (default ON) |
| `-DWARNINGS=ON` | extra compiler warnings |
| `-DENGINE_OUTPUT_DIR=<folder>` | write the executable somewhere else |
| `-G "MinGW Makefiles"`, `-G Ninja`, `-G "Visual Studio 17 2022"` | choose the toolchain |

Speed check (the last line shows nodes per second; a Release build reaches millions):

```powershell
"uci","bench","quit" | .\engine\bin\chess_engine.exe
```

## Menu

| Option | Description |
|--------|-------------|
| 1 | Human vs Human |
| 2 | Play vs AI (strong setting, skill level 4); you first choose White, Black or Random |
| 3 | Train a new model (legacy, see below) |
| 4 | Play vs Stockfish (needs `engine/stockfish.exe`); you first choose your side |
| 5 | Estimate the AI's Elo (plays in parallel; `evaluate_engine.py` is more reliable) |
| 6 | Exit |

Options 2 and 4 open a small screen first: **W** to play White, **B** to play Black, **R** for a random side,
**ESC** to go back. If you play Black the board is flipped so your pieces are at the bottom, and the AI makes
the first move.

Option 2 needs no model file. The weaker skill levels (1-3) are still available in code through
`ChessAI(skill_level=...)` but are not offered in the menu.

## Controls

| Key | Action |
|-----|--------|
| Click / Drag | Move pieces |
| A | Toggle AI on/off |
| S | Switch sides with the AI (ignored while the AI is thinking) |
| R | Reset board |
| Z | Undo move |
| ESC | Return to menu |
| F11 | Toggle fullscreen |

The window opens maximized and the board scales to fit; you can also resize it by hand. To start in a normal
window instead, run `python main.py --windowed`.

The board shows file letters (a-h) and rank numbers (1-8) along its edges; they flip with the board when you
play Black. Moves for the AI's side are ignored while it is that side's turn.

## Skill levels

`ChessAI(skill_level=1..5)`. With the C++ engine:

| Level | Behaviour |
|---|---|
| 1 | depth 1, deliberate mistakes (noise 220 cp) |
| 2 | depth 2, noise 90 cp |
| 3 | depth 4, noise 40 cp |
| 4 | full strength, about 0.6 s per move |
| 5 | full strength, about 2.5 s per move |

Levels 1-3 score every move and add random noise, so they make human-like mistakes instead of random
ones. Only level 4 has been benchmarked (see Strength); the others are not calibrated to an Elo. In the
first three moves of a game the engine picks slightly noisy moves so games do not repeat.

The evaluation printed in the terminal during a game (e.g. `AI plays: Nf3 (eval: +0.35)`) is the
engine's own score in pawns from White's point of view (`#3` = forced mate). It is not a Stockfish evaluation.

## How the engine works

- **Search:** iterative deepening, principal-variation search, transposition table, null-move pruning,
  reverse futility pruning, futility and late-move pruning of quiet moves, late-move reductions, check
  extensions, killer/history move ordering (the history table also penalises quiet moves that were tried
  and failed to cut), static exchange evaluation (SEE) to order captures and to skip losing captures in
  quiescence search, quiescence search.
- **Evaluation:** tapered middlegame/endgame piece-square tables (PeSTO) plus bishop pair, passed,
  isolated and doubled pawns, rook files and a king pawn shield.
- **Optional neural evaluation:** a king-bucketed network seen from both players' sides (see Training).
- **Time management:** a move gets its full time budget. The engine stops starting a new search depth after
  75% of the move time (UCI option `SoftTime`, 20-100), and a depth that is cut off still counts if it already
  found a better move.
- **Pondering:** after the strong AI moves, it searches the position in the background while you think
  (up to 30 s), so its next search starts with deep results for the likely replies. In a test against Stockfish
  limited to 1800 (0.1 s per move) it guessed the reply 44% of the time; when it did, the next search was about
  1 ply deeper, otherwise no gain, so about half a ply on average. It keeps one CPU core busy while you think.
  Turn it off with `ChessAI(..., ponder=False)` (set in `engine/game/chess_game.py`).
- **Board and move generation:** the third-party `chess.hpp` library (see Credits). The engine speaks
  UCI, and `engine/ai/chess_ai.py` talks to it through `python-chess`.

### UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 64 | transposition table size in MB (1-4096, rounded down to a power of two) |
| `UseNNUE`, `EvalFile` | on / empty | use the neural evaluation from this weights file (the game turns it off by default) |
| `Noise` | 0 | random noise in centipawns (used by the weak skill levels) |
| `SoftTime` | 75 | don't start a new depth after this % of the move time |
| `RFP` | 80 | reverse futility pruning margin in cp per remaining depth, 0 = off |
| `FP` | 120 | futility pruning margin for quiet moves at depth 3 or less, 0 = off |
| `LMP` | 1 | late-move pruning of quiet moves at depth 3 or less, 0 = off |
| `HistMalus` | 1 | bounded history with a penalty for quiet moves that failed to cut, 0 = old additive history |
| `NullEval` | 0 | only try the null move when the static eval is at or above beta |
| `NullDepth` | 0 | null-move reduction 3 + depth/4 instead of 3 (4 above depth 6) |
| `EvalParams` | empty | path of a weights file made by `tune_eval.py`; empty = the built-in weights (the tuned ones if `bake` was used, else the original hand-written ones); `original` = always the original hand-written weights |
| `Aspiration` | 25 | root aspiration window half-width in cp from depth 5, 0 = full window |
| `TimeManage` | 1 | with a game clock (`wtime`/`btime`): spend less when the best move is stable, more when it just changed or the score is falling; 0 = a fixed share of the clock |
| `TimeMax` | 300 | hard limit for one move as a % of the planned time (time management only) |
| `MoveOverhead` | 30 | milliseconds kept in reserve per move for communication delay |
| `Threads` | 1 | search threads (Lazy SMP); the game sets this itself for the strong setting |
| `SEE` | 2 | static exchange evaluation: 1 = order captures by it (losing captures last), 2 = also skip losing captures in quiescence, 0 = off |
| `LMR` | 0 | log-based late-move reduction table, 0 = the older fixed 1-2 ply reductions |
| `HistLMR` | 0 | reduce moves with a good history score less (only has an effect with `LMR=1`) |

Every search change is behind an option, so a build can be compared against itself with the option switched
off (see [Comparing engine versions](#comparing-engine-versions)). With `RFP=0 FP=0 LMP=0 HistMalus=0 SEE=0` the
search is identical to the version before these changes. The `bench` command prints a node count that can be
used to check that.

## Strength

**Current estimate: roughly 2600-2650 against Stockfish limited to a given Elo** (single-threaded), with the engine at
skill level 4 and both sides playing 0.1 s per move. This comes from benchmark games against Stockfish
(`evaluate_engine.py`), not from a rating list or games against people, so use it to compare versions
of this engine rather than as an official rating.

Benchmark results (colours alternate from random 4-ply openings; Stockfish uses `UCI_LimitStrength`
with the Elo shown):

| Stockfish Elo setting | Games | W / D / L | Score | Performance estimate |
|---|---|---|---|---|
| 1800 | 10 | 10 / 0 / 0 | 100% | at least ~2310 (a clean sweep only gives a bound) |
| 2200 | 10 | 9 / 0 / 1 | 90% | ~2580 |
| 2600 | 60 (10 + 50) | 21 / 13 / 26 | 46% | ~2570 (the 50-game run alone: 15 / 12 / 23, 42%, ~2544) |
| 2800 | 30 | 1 / 7 / 22 | 15% | ~2500 |

Performance = Stockfish's Elo setting + 400 * log10(score / (1 - score)). The levels where the result was
not a clean sweep agree to within about 100 Elo, which is how the 2500-2600 range was chosen.

Later runs with more games (2600 and 2800 levels, 100 games each; the number is the average of the two
levels' performance estimates; before the pruning and history changes):

| Games played at once | Estimate | Time |
|---|---|---|
| 2 | ~2521 | 17 min |
| 3 | ~2508 | 11 min |
| 4 | ~2471 | 9 min |

After the pruning, history-penalty and SEE changes (100 games per level, 3 games at once):

| Levels | Result | Estimate |
|---|---|---|
| 2600 and 2800 | overall estimate | ~2544 |
| 2800 | 7W 17D 76L (16%) | ~2505 |
| 2800 and 3000 | overall estimate | ~2586 |
| 2800 and 3000, after the tuned evaluation | overall estimate | ~2657 |
| 3000 only, 4 search threads (50 games) | overall estimate | ~2670 |
| 3000 | 0W 17D 83L (8%) | ~2587 |

The same build therefore reads about 2500-2590 depending on the level and the run. The head-to-head tests show
the changes are worth far more than this scale suggests (SEE alone +139 against the previous build), because
games against a copy of the same engine exaggerate gains, and Stockfish's limited-strength mode compresses
differences near the top of its range. Use the Stockfish figure as a slowly moving yardstick and
`match_engines.py` to judge changes.

Repeated 30-game-per-level runs of the same build scattered between about 2440 and 2600, which is what the
sample size allows. The drift down with more parallel games is within the noise but fits what a laptop does
when all cores are busy (lower clocks, so the engines search a little less in the same time).

For comparison, the earlier pure-Python engine held draws against the 1900 setting in a small early test
(about 6 games, longer thinking time), so the C++ engine is a large step up.

How to read it:

- Stockfish's Elo limiter is calibrated for long time controls; at 0.1 s per move Stockfish plays below its
  label, which flatters the engine's number somewhat. It is not comparable to FIDE or Lichess ratings.
- With 10-60 games per level the margin is roughly +/-100 Elo, with 100 games about +/-40. Differences of a
  few dozen points between two runs are noise; to compare two versions of the engine use `match_engines.py`.
- The engine searches by time, so results depend on the CPU it runs on.
- The first table (2 / 3 / 4 games at once) was measured before the pruning, history-penalty and SEE changes.
  None of the Stockfish runs used pondering. A score close to 0% or 100% at one level makes that level's
  estimate unreliable, so prefer levels where the score is between about 15% and 85%.

Reproduce it (needs `engine/stockfish.exe`, or pass `--stockfish PATH`):

```powershell
python evaluate_engine.py --movetime 0.1 --games 20
python evaluate_engine.py --movetime 0.1 --games 100 --elos 2600 2800 --concurrency 3
```

The script prints win/draw/loss counts, a performance estimate per level, and the total time taken.
`--concurrency N` plays N games at the same time (each game runs two engines, so keep N at or below half your
physical cores; on an 8-core CPU use 3-4, and compare runs only at the same setting). Games that are clearly
decided (Stockfish's own score beyond +-10 pawns for 4 moves in a row) are stopped early; `--no-resign` plays
every game to the end. With `--tc BASE+INC` both sides play with a real game clock instead of a fixed time per move, for example
`python evaluate_engine.py --tc 10+0.1 --games 100 --elos 2800 3000 --concurrency 3`; the engine then uses its
own time management (no opening book or pondering) and a side whose clock runs out loses.
Other flags: `--level`, `--sf-time`, `--nnue`, `--ponder` (let the AI think on
Stockfish's time; needs spare CPU cores), `--max-plies`. Pick Elo levels close to the engine's strength,
because a 100% or 0% score only gives a bound.

## Time management

When the engine is given a game clock (UCI `wtime`/`btime`/`winc`/`binc`/`movestogo`) it plans the time for
this move as about 1/30 of the remaining time plus most of the increment, capped at 40% of what is left, after
keeping `MoveOverhead` in reserve, so it can no longer plan more time than it has. With `TimeManage=1` that plan
is a target that is scaled after each finished depth: shorter when the best move has stayed the same for
several depths (down to about 55%), longer when the best move just changed (25% more) or the score is falling
(up to 60% more), and a depth that has started may run on up to `TimeMax` percent of the plan (never more than
half of the remaining clock). A fixed time per move (`go movetime`, used by the tests and `bench`) is unchanged.

In the game the strong setting gives the engine a fresh virtual clock for every move, so the per-level time
(about 0.6 s at level 4) is the target and the real time per move varies around it.

Fixed-time tests cannot judge this, so `match_engines.py` can play with a clock:

```powershell
python match_engines.py --a engine\bin\chess_engine.exe --b engine\bin\chess_engine.exe --a-opt TimeManage=1 --b-opt TimeManage=0 --tc 10+0.1 --games 300 --concurrency 4
```

`--tc BASE+INC` is the clock of each side in seconds; a side that runs out of time loses, and the summary shows
the average time per move of both engines and the number of games lost on time.

## Multi-threaded search

With `Threads` above 1 the engine runs Lazy SMP: helper threads search the same position with different depth
patterns and share one lock-free transposition table, and the main thread picks the move. The game uses
`default_threads()` in `engine/ai/chess_ai.py` (a quarter of the logical CPUs, at most 8; 4 on an 8-core /
16-thread laptop). Set the environment variable `CHESS_THREADS` to change it, for example
`$env:CHESS_THREADS = 6` in PowerShell before `python main.py`. The evaluation scripts stay single-threaded
unless asked: `evaluate_engine.py --threads 4` (use a low `--concurrency`, since every game then runs several
threads).

Test it against the single-threaded engine; each game now uses `Threads + 1` threads, so keep `--concurrency`
at about a quarter of your logical CPUs:

```powershell
python match_engines.py --a engine\bin\chess_engine.exe --b engine\bin\chess_engine.exe --a-opt Threads=4 --games 200 --movetime 0.1 --concurrency 2
```

## Opening book

In the strong setting the first 12 moves come from `engine/data/book.bin` when the position is in it (the
terminal prints `AI plays: e4 (eval: book)` for these moves), picked at random in proportion to how often each move occurs in the book. The book is a
standard Polyglot file. `engine/data/opening_lines.txt` lists about 50 well-known lines in SAN; add your own and
rebuild:

```powershell
python build_book.py                                   # opening_lines.txt -> engine/data/book.bin
python build_book.py --pgn games.pgn --min-elo 2300    # also learn from strong games (PGN)
```

Every move is checked for legality while the book is built. Any other Polyglot `.bin` book can be used by
copying it to `engine/data/book.bin`. The evaluation scripts and `match_engines.py` do not use the book.

## Tuning the evaluation

`tune_eval.py` fits the weights of the hand-written evaluation (piece values and piece-square tables, passed,
isolated and doubled pawns, rook files, bishop pair, king shield, tempo) to the Stockfish evaluations in the
prepared data (`engine/data/processed/nnue` and `nnue2`, each with `bitboards.npy` and `labels.npy`). It also
has two new terms that are off by default: piece **mobility** and **king-zone attacks** (squares near the enemy
king attacked by knights, bishops, rooks and queens). The score is linear in the weights, so the fit is exact
and fast and needs only NumPy. The fit starts from the engine's current weights, which it reads from the
engine itself (`evalparams` command), and writes a text file that the engine loads through the `EvalParams`
option. Nothing about the engine changes until you load a file.

```powershell
python tune_eval.py check                      # first: do the tuner's features match the engine's evaluation?
python tune_eval.py tune --max-positions 3000000 --epochs 60
python tune_eval.py tune --no-new-terms        # re-fit only the existing terms
```

It reports the loss and typical error (centipawns) on held-out positions before and after, and writes
`engine/models/saved/eval_params.txt` only if the held-out loss improved. Test the result like any other
change, then decide:

```powershell
python match_engines.py --a engine\bin\chess_engine.exe --b engine\bin\chess_engine.exe --a-opt EvalParams=engine\models\saved\eval_params.txt --games 300 --movetime 0.1 --concurrency 4
```

Mobility and king-zone terms cost search speed (the evaluation then needs attack maps); compare the `bench`
nodes per second with and without the file. A better held-out loss does not guarantee a stronger engine, so the
match result decides.

Weights fitted without a limit made the engine clearly weaker (-43 and -101 Elo), even though the held-out
error went down: large weight changes no longer fit the search's fixed margins and capture values, and the
target (Stockfish's deep evaluations) is not what this search needs. Keeping every weight within 10 cp of
the current one (`--max-change 10`) made the fit safe, and with mobility and king-zone terms enabled it gave
a small gain (see the table below).

### Making a tuned file the default

```powershell
python tune_eval.py bake --params engine\models\saved\p4.txt      # writes engine/cpp/eval_default.inc
cmake --build build --config Release
python tune_eval.py verify --params engine\models\saved\p4.txt    # engine's built-in weights == that file?
```

The weights are compiled into the engine, so the game and the evaluation scripts all use them without a path.
The original hand-written weights stay available as `EvalParams=original`, which makes `match_engines.py`
comparisons against the old evaluation easy (`--b-opt EvalParams=original`). To go back for good, empty the text
in `engine/cpp/eval_default.inc` (or restore the placeholder from git) and rebuild. A later `tune` run starts
from whatever the engine currently uses.

## Comparing engine versions

Stockfish games are too noisy to tell whether a change helped. `match_engines.py` plays two engine builds, or
the same build with different UCI options, against each other. Each random opening is played twice with the
colours swapped, decided games are stopped early (`--no-resign` to disable), and the result is printed as
W/D/L, score, Elo difference with a 95% interval and the likelihood that A is stronger.

```powershell
# keep the previous build, rebuild, then compare
Copy-Item engine\bin\chess_engine.exe engine\bin\chess_engine_old.exe
cmake --build build --config Release
python match_engines.py --a engine\bin\chess_engine.exe --b engine\bin\chess_engine_old.exe --games 300 --movetime 0.1 --concurrency 4

# test an option on one build
python match_engines.py --a engine\bin\chess_engine.exe --b engine\bin\chess_engine.exe --a-opt HistMalus=0 --games 300 --movetime 0.1 --concurrency 4
```

Options: `--a`, `--b`, `--a-opt Name=Value`, `--b-opt Name=Value` (repeatable), `--games`, `--movetime`, `--tc`,
`--concurrency` (at most half the physical cores), `--hash`, `--opening-plies`, `--max-plies`, `--seed`.
It checks that both engines know the requested options, so an old build that lacks one fails immediately.

Reading the result: if the interval still contains 0 the change is not proven. About 300 games reveal a
difference of 30-50 Elo, and 1000+ games are needed for 10-20 Elo. Compare only runs made with the same
`--movetime`.

Results so far (300 games at 0.1 s per move unless noted; positive = the first option is stronger):

| Test | Elo difference | 95% interval | Decision |
|---|---|---|---|
| `SoftTime=100` vs 75 (200 games) | -17 | [-59, +23] | keep 75 |
| `SoftTime=50` vs 75 (200 games) | -17 | [-61, +25] | keep 75 |
| pruning (`RFP`, `FP`, `LMP`) vs none | +24 | [-9, +58] | on by default (a 120-game test at 0.05 s gave +29) |
| `HistMalus=1` vs 0 | +42 | [+9, +76] | on by default |
| default build vs the pruning-only build | +23 | [-11, +58] | consistent with the line above |
| `SEE=2` vs 0 | +139 | [+103, +178] | on by default (a 120-game test at 0.05 s gave +101) |
| `NullEval=1` vs 0 (400 games) | +9 | [-21, +38] | off (no evidence of a gain) |
| `NullDepth=1` vs 0 (400 games) | -5 | [-35, +25] | off (no evidence of a gain) |
| `TimeManage=1` vs 0, clock 10+0.1 | +72 | [+38, +107] | on by default (a 100-game test at 4+0.04 gave +49); average time per move was equal |
| `EvalParams`, unbounded fit, old and new terms | -43 | [-82, -6] | rejected |
| `EvalParams`, unbounded fit, old terms only | -101 | [-142, -63] | rejected |
| `EvalParams`, fit within 10 cp, old terms only | +7 | [-28, +42] | no evidence of a gain |
| `EvalParams`, fit within 10 cp, with mobility and king zone (700 games pooled from 2 runs) | +18 | [-5, +42] | adopted as the default |
| new build (lockless table, options off) vs the previous build | +23 | [-20, +66] | no change expected; the search is node-for-node identical, so this is noise |
| `Aspiration=25` vs 0 | +33 | [+4, +63] | on by default (a 120-game test at 0.05 s gave +32) |
| `Threads=4` vs 1 (200 games, 2 games at once) | +106 | [+62, +153] | the game uses several threads in the strong setting |
| `LMR=1` alone vs the older build | -7 | [-42, +28] | off |
| `LMR=1 HistMalus=1 HistLMR=1` vs `HistMalus=0` | +19 | [-15, +53] | worse than `HistMalus` alone, so `LMR` and `HistLMR` stay off |

Ideas not done yet: tuning the search margins after the evaluation change, endgame tablebases, a smaller or
incrementally updated neural network.

## Training the neural network

The network is trained on the Lichess evaluation database (positions with deep Stockfish evaluations),
so no Stockfish run is needed.

```bash
pip install zstandard torch
```

1. Download `lichess_db_eval.jsonl.zst` from https://database.lichess.org/ (the "Evaluations" section)
   into `engine/data/raw/`. It is large; a partial download of a few GB is enough, e.g.
   `curl.exe -L -r 0-1999999999 -o engine\data\raw\lichess_db_eval.jsonl.zst <link>`.
2. Build the training arrays (quiet positions only, compact bitboard format):
   ```bash
   python train_nnue.py prepare --src engine/data/raw/lichess_db_eval.jsonl.zst --max-positions 5000000
   # for more variety, a second sample spread across the file:
   python train_nnue.py prepare --src engine/data/raw/lichess_db_eval.jsonl.zst --keep-prob 0.15 --max-positions 5000000 --out engine/data/processed/nnue2
   ```
3. Train (uses the GPU if available) and export `engine/models/saved/nnue.npz`:
   ```bash
   python train_nnue.py train --epochs 40
   python train_nnue.py train --data engine/data/processed/nnue engine/data/processed/nnue2 --hidden 128
   ```
4. Check it:
   ```bash
   python train_nnue.py compare --data engine/data/processed/nnue engine/data/processed/nnue2
   python train_nnue.py match --games 40 --movetime 0.2
   python train_nnue.py match --games 40 --movetime 0.2 --nn-time-factor 2.0
   ```
   `compare` scores the hand-written eval and the network on the same held-out positions; `match` plays
   the network against the hand-written eval; `--nn-time-factor` gives the network extra time.

**Current status:** the network is a more accurate evaluator than the hand-written one on held-out
positions (loss about 0.0062 vs 0.0135, typical error 92 cp vs 144 cp), but it is much slower per node in
the C++ engine, so the hand-written evaluation plays stronger at equal thinking time. The network is
therefore **off by default**, and the strength figures above are for the hand-written evaluation. To try
the network, create the AI with `ChessAI(skill_level=4, use_nnue=True)`; the C++ engine reads a `nnue.bin`
that is generated next to `nnue.npz` on first use. Making it competitive needs a smaller hidden size or
incremental updates inside the C++ engine.

## Legacy training (menu option 3)

Menu option 3 trains the original convolutional network from PGN games placed in `engine/data/raw/`
(`engine/data/pgn_processor.py`, `engine/ai/chess_trainer.py`, `engine/models/chess_net.py`). The AI no
longer uses that network, so use `train_nnue.py` instead. These files are kept only so the menu still starts.

## Project layout

```
Chess/
├── main.py                     entry point
├── CMakeLists.txt              builds the C++ engine
├── evaluate_engine.py          AI vs Stockfish evaluation
├── match_engines.py            engine build/option set vs another (A/B testing)
├── tune_eval.py                fits the evaluation weights to the prepared Lichess positions
├── build_book.py               builds the opening book (Polyglot) from opening lines or PGN games
├── train_nnue.py               prepare data / train / compare / match
├── controllers/                menu and game controllers
├── ui/                         pygame menu and board drawing
├── utils/config.py             settings
└── engine/
    ├── ai/chess_ai.py          ChessAI: C++ backend over UCI, pure-Python fallback, book and thread support
    ├── ai/opening_book.py      Polyglot opening book reader
    ├── ai/chess_trainer.py     legacy trainer
    ├── cpp/                    engine.cpp, chess.hpp, eval_default.inc (baked evaluation weights)
    ├── bin/                    build output (not committed)
    ├── data/                   book.bin and opening_lines.txt; raw/ and processed/ training data (not committed)
    ├── game/                   board, rules and game state
    └── models/                 nnue.py, chess_net.py (legacy), saved/ (trained networks)
```

## Troubleshooting

- **Terminal says `AI: Python engine`:** `engine/bin/chess_engine(.exe)` was not found; build it.
- **The engine feels slow or weak:** make sure it was built with `--config Release`, not Debug.
- **`could not load net` or the network is ignored:** only v2 networks (made by the current
  `train_nnue.py`) work with the C++ engine; retrain with `train_nnue.py train`.
- **A new UCI option is missing after rebuilding:** stop any running `chess_engine` process
  (`Get-Process chess_engine | Stop-Process`) and rebuild with `cmake --build build --config Release --clean-first`;
  check with `Select-String -Path engine\bin\chess_engine.exe -Pattern "HistMalus" -Encoding ascii`.
- **Stockfish not found:** download it from https://stockfishchess.org/download/ and place it at
  `engine/stockfish.exe`.

## Credits and licenses

- **[chess-library](https://github.com/Disservin/chess-library)** by Disservin (MIT) - board, move
  generation and draw detection, used unmodified as `engine/cpp/chess.hpp`. Its license notice must stay
  at the top of that file.
- **PeSTO evaluation tables** by Ronald Friederich - the piece-square tables used by the hand-written eval.
- **[Lichess evaluation database](https://database.lichess.org/)** (CC0) - training data for the network.
- **[python-chess](https://python-chess.readthedocs.io/)** - used by the Python side for boards and the UCI link.
- **[Stockfish](https://stockfishchess.org/)** - used only as an opponent for measuring strength; not bundled.