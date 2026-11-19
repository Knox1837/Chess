"""
Build an opening book (standard Polyglot .bin format) for the game.
  python build_book.py                                  # engine/data/opening_lines.txt -> engine/data/book.bin
  python build_book.py --pgn games.pgn --min-elo 2300   # add the first moves of strong games from a PGN file
  python build_book.py --pgn a.pgn b.pgn --no-lines     # only from PGN files
"""
import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot

ROOT = Path(__file__).resolve().parent
DEFAULT_LINES = ROOT / "engine" / "data" / "opening_lines.txt"
DEFAULT_OUT = ROOT / "engine" / "data" / "book.bin"
CASTLING_FIX = {"e1g1": "e1h1", "e1c1": "e1a1", "e8g8": "e8h8", "e8c8": "e8a8"}   # Polyglot: king takes rook

def encode_move(board, move):
    uci = move.uci()
    if uci in CASTLING_FIX and board.piece_type_at(move.from_square) == chess.KING:
        move = chess.Move.from_uci(CASTLING_FIX[uci])
    promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]
    to_sq, from_sq = move.to_square, move.from_square
    return ((to_sq & 7) | ((to_sq >> 3) << 3) | ((from_sq & 7) << 6) | ((from_sq >> 3) << 9) | (promo << 12))

def add_line(counts, moves_san, source, max_plies):
    board = chess.Board()
    for ply, san in enumerate(moves_san[:max_plies]):
        try:
            move = board.parse_san(san)
        except ValueError:
            sys.exit(f"{source}: illegal or unreadable move '{san}' at ply {ply + 1} after: {' '.join(moves_san[:ply])}")
        counts[(chess.polyglot.zobrist_hash(board), encode_move(board, move))] += 1
        board.push(move)

def read_lines(path, max_plies):
    counts = Counter()
    n = 0
    for no, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#")[0].strip()
        if line:
            add_line(counts, line.split(), f"{path} line {no}", max_plies)
            n += 1
    print(f"  {n} opening lines from {path}")
    return counts

def read_pgn(paths, max_plies, min_elo, min_games):
    counts = Counter()
    games = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                try:
                    if min(int(game.headers.get("WhiteElo", 0)), int(game.headers.get("BlackElo", 0))) < min_elo:
                        continue
                except ValueError:
                    continue
                board = game.board()
                try:
                    for ply, move in enumerate(game.mainline_moves()):
                        if ply >= max_plies:
                            break
                        counts[(chess.polyglot.zobrist_hash(board), encode_move(board, move))] += 1
                        board.push(move)
                except Exception:
                    continue
                games += 1
    counts = Counter({k: v for k, v in counts.items() if v >= min_games})
    print(f"  {games} games used from {len(paths)} PGN file(s), {len(counts)} moves kept (seen >= {min_games} times)")
    return counts

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", default=str(DEFAULT_LINES), help="text file of opening lines (SAN)")
    ap.add_argument("--no-lines", action="store_true", help="do not use the lines file")
    ap.add_argument("--pgn", nargs="*", default=[], help="PGN files whose games are added")
    ap.add_argument("--min-elo", type=int, default=2300, help="PGN games need both players at least this rated")
    ap.add_argument("--min-games", type=int, default=3, help="PGN: keep moves seen at least this many times")
    ap.add_argument("--max-plies", type=int, default=24)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    counts = Counter()
    print("Building the book")
    if not a.no_lines:
        counts.update(read_lines(a.lines, a.max_plies))
    if a.pgn:
        counts.update(read_pgn(a.pgn, a.max_plies, a.min_elo, a.min_games))
    if not counts:
        sys.exit("Nothing to write.")
    entries = sorted(((key, move, min(w, 65535)) for (key, move), w in counts.items()),
                     key=lambda e: (e[0], -e[2]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        for key, move, weight in entries:
            f.write(struct.pack(">QHHI", key, move, weight, 0))
    print(f"Wrote {a.out}: {len(entries)} moves in {len({e[0] for e in entries})} positions")

if __name__ == "__main__":
    main()