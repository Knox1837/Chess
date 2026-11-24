# Changelog

## v1.0.0

A chess game (Python and pygame) with a fast C++ engine behind it, and the tools used to measure and improve it.

### Game
- Play against the AI as White, Black or a random side, against a friend, or against Stockfish.
- Move list next to the board in standard notation. Step through earlier positions with the arrow keys, the
  buttons or by clicking a move, and the board shows that position.
- Ends in checkmate or a draw: stalemate, insufficient material, threefold repetition or the fifty-move rule.
- File letters and rank numbers on the board, undo, reset and a switch-sides key.
- The strong AI plays its first moves from an opening book, thinks on your time (pondering) and uses several
  search threads.

### Engine
- UCI engine written in C++: principal-variation search, transposition table shared between threads,
  null-move, futility and late-move pruning, static exchange evaluation, history ordering, aspiration windows,
  Lazy SMP multi-threading and time management with a game clock.
- Hand-written evaluation (piece-square tables, pawn structure, rook files, bishop pair, king shield) tuned on
  Lichess evaluations, with mobility and king-zone terms.
- Roughly 2800-2900 against Stockfish limited to a given Elo with a 10 s + 0.1 s clock, and 2600-2650 at a
  fixed 0.1 s per move (single-threaded). These are benchmark figures against Stockfish's limited-strength mode,
  not an official rating.

### Tools
- `evaluate_engine.py` plays the engine against Stockfish (fixed time per move or a real game clock).
- `match_engines.py` plays two engine builds or option sets against each other, for testing changes.
- `tune_eval.py` fits the evaluation weights to the prepared Lichess data.
- `build_book.py` builds the opening book (Polyglot format).

### Requirements
Python with `pygame` and `python-chess`. The engine is built with CMake (`cmake -B build`, then
`cmake --build build --config Release`); the Windows download below already contains `engine/bin/chess_engine.exe`.
Stockfish is optional and not included. See the README for details.