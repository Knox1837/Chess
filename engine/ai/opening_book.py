"""Opening book reader (standard Polyglot .bin files, built by build_book.py)."""
from pathlib import Path

import chess
import chess.polyglot

DEFAULT_BOOK = Path(__file__).resolve().parents[1] / "data" / "book.bin"

def find_book(path=None):
    """The given book file, else engine/data/book.bin if it exists, else None."""
    for cand in (path, DEFAULT_BOOK):
        if cand and Path(cand).is_file():
            return str(cand)
    return None

class OpeningBook:
    def __init__(self, path):
        self.path = str(path)
        self.reader = chess.polyglot.open_reader(self.path)

    def pick(self, board):
        """A random book move for the position (weighted by how often it was played), or None."""
        try:
            return self.reader.weighted_choice(board).move
        except IndexError:
            return None

    def close(self):
        try:
            self.reader.close()
        except Exception:
            pass