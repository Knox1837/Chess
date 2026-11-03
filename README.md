# Chess AI

A chess game written in Python (pygame) with a fast C++ alpha-beta engine behind it, an optional
neural-network evaluator trained on Lichess data, and tools to measure how strong it is.

- **Play** against the AI, against a friend, or against Stockfish.
- **C++ engine** (`engine/cpp/`): about 5 million nodes/sec, searches 10+ plies in well under a second.
- **Strength:** roughly **2500-2600** against Stockfish limited to a given Elo, at 0.1 s per move
  (benchmark games against Stockfish, see [Strength](#strength)).
- **Pure-Python fallback**: if the C++ engine has not been built, the game still works (just weaker and slower).
- **NNUE-style network** (`train_nnue.py`): trained from the Lichess evaluation database; optional.
- **Evaluation tools**: `evaluate_engine.py` (vs Stockfish) and `train_nnue.py match` (network vs hand-written eval).

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
| 2 | Play vs AI (strong setting, skill level 4) |
| 3 | Play vs Basic AI (weak setting, skill level 2) |
| 4 | Train a new model (legacy, see below) |
| 5 | Play vs Stockfish (needs `engine/stockfish.exe`) |
| 6 | Estimate the AI's Elo (plays in parallel; `evaluate_engine.py` is more reliable) |
| 7 | Exit |

Options 2 and 3 only differ in skill level; neither needs a model file.

## Controls

| Key | Action |
|-----|--------|
| Click / Drag | Move pieces |
| A | Toggle AI on/off |
| S | Switch AI side |
| R | Reset board |
| Z | Undo move |
| ESC | Return to menu |

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
  late-move reductions, check extensions, killer/history move ordering, quiescence search.
- **Evaluation:** tapered middlegame/endgame piece-square tables (PeSTO) plus bishop pair, passed,
  isolated and doubled pawns, rook files and a king pawn shield.
- **Optional neural evaluation:** a king-bucketed network seen from both players' sides (see Training).
- **Board and move generation:** the third-party `chess.hpp` library (see Credits). The engine speaks
  UCI, and `engine/ai/chess_ai.py` talks to it through `python-chess`.

## Strength

**Current estimate: roughly 2500-2600 against Stockfish limited to a given Elo**, with the engine at
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

For comparison, the earlier pure-Python engine held draws against the 1900 setting in a small early test
(about 6 games, longer thinking time), so the C++ engine is a large step up.

How to read it:

- Stockfish's Elo limiter is calibrated for long time controls; at 0.1 s per move Stockfish plays below its
  label, which flatters the engine's number somewhat. It is not comparable to FIDE or Lichess ratings.
- With 10-60 games per level the margin is roughly +/-100 Elo. Differences of a few dozen points between
  two versions are noise; changes of a few hundred points are real.
- The engine searches by time, so results depend on the CPU it runs on.

Reproduce it (needs `engine/stockfish.exe`, or pass `--stockfish PATH`):

```powershell
python evaluate_engine.py --movetime 0.1 --games 20
python evaluate_engine.py --movetime 0.1 --games 50 --elos 2400 2600 2800
```

The script plays the AI against Stockfish one game at a time (no parallel workers, so the time-limited
search is not starved of CPU) and prints win/draw/loss counts, a performance estimate per level, and the
total time taken. Other flags: `--level`, `--sf-time`, `--nnue`, `--max-plies`. Pick Elo levels close to
the engine's strength, because a 100% or 0% score only gives a bound.

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

## Legacy training (menu option 4)

Menu option 4 trains the original convolutional network from PGN games placed in `engine/data/raw/`
(`engine/data/pgn_processor.py`, `engine/ai/chess_trainer.py`, `engine/models/chess_net.py`). The AI no
longer uses that network, so use `train_nnue.py` instead. These files are kept only so the menu still starts.

## Project layout

```
Chess/
├── main.py                     entry point
├── CMakeLists.txt              builds the C++ engine
├── evaluate_engine.py          AI vs Stockfish evaluation
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