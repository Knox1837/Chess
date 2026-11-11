# Chess AI

A chess game written in Python (pygame) with a fast C++ alpha-beta engine behind it, an optional
neural-network evaluator trained on Lichess data, and tools to measure how strong it is.

- **Play** against the AI, against a friend, or against Stockfish.
- **C++ engine** (`engine/cpp/`): millions of nodes per second, searches 10+ plies in well under a second.
- **Thinks on your time:** in the strong setting the engine keeps searching while you choose your move (pondering).
- **Strength:** roughly **2500-2550** against Stockfish limited to a given Elo, at 0.1 s per move
  (benchmark games against Stockfish, see [Strength](#strength)). Later search improvements gained roughly
  +20-30 Elo each in head-to-head matches and have not been re-measured against Stockfish yet.
- **Pure-Python fallback**: if the C++ engine has not been built, the game still works (just weaker and slower).
- **NNUE-style network** (`train_nnue.py`): trained from the Lichess evaluation database; optional.
- **Evaluation tools**: `evaluate_engine.py` (vs Stockfish), `match_engines.py` (one engine build or option set vs
  another, for testing changes) and `train_nnue.py match` (network vs hand-written eval).

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
  and failed to cut), quiescence search.
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
| `LMR` | 0 | log-based late-move reduction table, 0 = the older fixed 1-2 ply reductions |
| `HistLMR` | 0 | reduce moves with a good history score less (only has an effect with `LMR=1`) |

Every search change is behind an option, so a build can be compared against itself with the option switched
off (see [Comparing engine versions](#comparing-engine-versions)). With `RFP=0 FP=0 LMP=0 HistMalus=0` the
search is identical to the version before these changes. The `bench` command prints a node count that can be
used to check that.

## Strength

**Current estimate: roughly 2500-2550 against Stockfish limited to a given Elo**, with the engine at
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
- The Stockfish runs were made before the pruning and history-penalty changes and without pondering; re-run
  `evaluate_engine.py` to refresh them.

Reproduce it (needs `engine/stockfish.exe`, or pass `--stockfish PATH`):

```powershell
python evaluate_engine.py --movetime 0.1 --games 20
python evaluate_engine.py --movetime 0.1 --games 100 --elos 2600 2800 --concurrency 3
```

The script prints win/draw/loss counts, a performance estimate per level, and the total time taken.
`--concurrency N` plays N games at the same time (each game runs two engines, so keep N at or below half your
physical cores; on an 8-core CPU use 3-4, and compare runs only at the same setting). Games that are clearly
decided (Stockfish's own score beyond +-10 pawns for 4 moves in a row) are stopped early; `--no-resign` plays
every game to the end. Other flags: `--level`, `--sf-time`, `--nnue`, `--ponder` (let the AI think on
Stockfish's time; needs spare CPU cores), `--max-plies`. Pick Elo levels close to the engine's strength,
because a 100% or 0% score only gives a bound.

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

Options: `--a`, `--b`, `--a-opt Name=Value`, `--b-opt Name=Value` (repeatable), `--games`, `--movetime`,
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
| `LMR=1` alone vs the older build | -7 | [-42, +28] | off |
| `LMR=1 HistMalus=1 HistLMR=1` vs `HistMalus=0` | +19 | [-15, +53] | worse than `HistMalus` alone, so `LMR` and `HistLMR` stay off |

Ideas not done yet: static exchange evaluation for captures, aspiration windows, tuning the evaluation
weights, multi-threaded search, an opening book, a smaller or incrementally updated neural network.

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
├── train_nnue.py               prepare data / train / compare / match
├── controllers/                menu and game controllers
├── ui/                         pygame menu and board drawing
├── utils/config.py             settings
└── engine/
    ├── ai/chess_ai.py          ChessAI: C++ backend over UCI, pure-Python fallback
    ├── ai/chess_trainer.py     legacy trainer
    ├── cpp/                    engine.cpp, chess.hpp
    ├── bin/                    build output (not committed)
    ├── data/                   raw/ and processed/ training data (not committed)
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