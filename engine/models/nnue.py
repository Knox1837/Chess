"""
NNUE-style evaluator: shared feature definition + fast NumPy runtime.

Output: centipawns for the side to move (= logit * SCALE).
"""
import os
from pathlib import Path

import chess
import numpy as np

FEATURES = 768
SCALE = 400.0          # sigmoid(cp / SCALE) is the training target
KING_BUCKETS = 8       # v2: own-king square -> 4 file groups x 2 rank groups
MAX_CP = 2500          # clamp inside search so we never collide with mate scores

_scan = chess.scan_forward

def feature_indices(board):
    """Active feature indices (0..767) for the side to move."""
    stm = board.turn
    flip = 0 if stm else 56
    idx = []
    for pt in range(1, 7):
        base_own = (pt - 1) * 64
        base_opp = (pt + 5) * 64
        for sq in _scan(board.pieces_mask(pt, stm)):
            idx.append(base_own + (sq ^ flip))
        for sq in _scan(board.pieces_mask(pt, not stm)):
            idx.append(base_opp + (sq ^ flip))
    return idx

def pack_board(board):
    """12 bitboards (python ints) in the same plane order as feature_indices()."""
    stm = board.turn
    out = []
    for color in (stm, not stm):
        for pt in range(1, 7):
            bb = board.pieces_mask(pt, color)
            if not stm:
                bb = chess.flip_vertical(bb)
            out.append(bb)
    return out

class NNUEEval:
    """CPU inference: accumulate first-layer rows, then two tiny dense layers."""

    def __init__(self, path):
        d = np.load(path)
        self.W1 = np.ascontiguousarray(d["W1"], dtype=np.float32)   # (768, H)
        self.b1 = d["b1"].astype(np.float32)
        self.W2 = d["W2"].astype(np.float32)                         # (H, 32)
        self.b2 = d["b2"].astype(np.float32)
        self.W3 = d["W3"].astype(np.float32)                         # (32, 32)
        self.b3 = d["b3"].astype(np.float32)
        self.W4 = d["W4"].astype(np.float32).reshape(-1)             # (32,)
        self.b4 = float(np.asarray(d["b4"]).reshape(-1)[0])
        self.hidden = self.W1.shape[1]

    def logit(self, board):
        h = self.W1[feature_indices(board)].sum(axis=0)
        h += self.b1
        np.clip(h, 0.0, 1.0, out=h)
        h = h @ self.W2 + self.b2
        np.clip(h, 0.0, 1.0, out=h)
        h = h @ self.W3 + self.b3
        np.clip(h, 0.0, 1.0, out=h)
        return float(h @ self.W4) + self.b4

    def evaluate(self, board):
        """Centipawns from the side-to-move's point of view."""
        cp = int(self.logit(board) * SCALE)
        return MAX_CP if cp > MAX_CP else (-MAX_CP if cp < -MAX_CP else cp)

def _bucketed_indices(board, p):
    """v2 features for perspective colour p: bucket(own king) * 768 + plane * 64 + square."""
    flip = 0 if p else 56
    ksq = board.king(p) ^ flip
    base = (((ksq & 7) >> 1) + (4 if (ksq >> 3) >= 2 else 0)) * FEATURES
    idx = []
    for pt in range(1, 7):
        own = base + (pt - 1) * 64
        opp = base + (pt + 5) * 64
        for sq in _scan(board.pieces_mask(pt, p)):
            idx.append(own + (sq ^ flip))
        for sq in _scan(board.pieces_mask(pt, not p)):
            idx.append(opp + (sq ^ flip))
    return idx


def _build_tables():
    """_T[colour][abs_idx] -> feature index (without king bucket) seen from that colour's perspective.
    abs_idx = plane*64 + square with planes ordered White P N B R Q K, Black P N B R Q K."""
    out = {}
    for persp in (True, False):
        t = np.zeros(768, dtype=np.int64)
        for plane in range(12):
            white_piece = plane < 6
            rel_plane = (plane % 6) + (0 if white_piece == persp else 6)
            for sq in range(64):
                t[plane * 64 + sq] = rel_plane * 64 + (sq if persp else sq ^ 56)
        out[persp] = t
    return out

_T = _build_tables()

class NNUEEvalV2:
    """Two accumulators (side to move, opponent) -> concat -> 32 -> 32 -> 1."""

    def __init__(self, d):
        self.W1 = np.ascontiguousarray(d["W1"], dtype=np.float32)    # (8*768, H)
        self.b1 = d["b1"].astype(np.float32)
        W2 = d["W2"].astype(np.float32)                               # (2H, 32)
        self.W2a = np.ascontiguousarray(W2[: W2.shape[0] // 2])       # applies to the side-to-move half
        self.W2b = np.ascontiguousarray(W2[W2.shape[0] // 2:])        # applies to the opponent half
        self.b2 = d["b2"].astype(np.float32)
        self.W3 = d["W3"].astype(np.float32)                          # (32, 32)
        self.b3 = d["b3"].astype(np.float32)
        self.W4 = d["W4"].astype(np.float32).reshape(-1)              # (32,)
        self.b4 = float(np.asarray(d["b4"]).reshape(-1)[0])
        self.hidden = self.W1.shape[1]

    def logit(self, board):
        # 12 absolute bitboards -> bit array -> active (plane, square) list, all in NumPy.
        w = board.occupied_co[True]
        k = board.occupied_co[False]
        pw, nw, bw, rw, qw, kw = board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings
        arr = np.array((pw & w, nw & w, bw & w, rw & w, qw & w, kw & w,
                        pw & k, nw & k, bw & k, rw & k, qw & k, kw & k), dtype=np.uint64)
        act = np.flatnonzero(np.unpackbits(arr.view(np.uint8), bitorder="little"))
        stm = board.turn
        if stm:
            ku = (kw & w).bit_length() - 1
            kt = ((kw & k).bit_length() - 1) ^ 56
        else:
            ku = ((kw & k).bit_length() - 1) ^ 56
            kt = (kw & w).bit_length() - 1
        bu = (((ku & 7) >> 1) + (4 if (ku >> 3) >= 2 else 0)) * FEATURES
        bt = (((kt & 7) >> 1) + (4 if (kt >> 3) >= 2 else 0)) * FEATURES
        a = self.W1[_T[stm][act] + bu].sum(axis=0)
        a += self.b1
        b = self.W1[_T[not stm][act] + bt].sum(axis=0)
        b += self.b1
        np.clip(a, 0.0, 1.0, out=a)
        np.clip(b, 0.0, 1.0, out=b)
        h = a @ self.W2a + b @ self.W2b
        h += self.b2
        np.clip(h, 0.0, 1.0, out=h)
        h = h @ self.W3 + self.b3
        np.clip(h, 0.0, 1.0, out=h)
        return float(h @ self.W4) + self.b4

    def evaluate(self, board):
        cp = int(self.logit(board) * SCALE)
        return MAX_CP if cp > MAX_CP else (-MAX_CP if cp < -MAX_CP else cp)

def load_nnue(path):
    d = np.load(path)
    if "version" in d.files and int(d["version"]) == 2:
        return NNUEEvalV2(d)
    return NNUEEval(path)

def export_bin(npz_path, bin_path):
    """Write the flat float32 file the C++ engine reads ("NNU2" | H | W1 b1 W2 b2 W3 b3 W4 b4)."""
    d = np.load(npz_path)
    if "version" not in d.files or int(d["version"]) != 2:
        raise ValueError("the C++ engine needs a v2 network - run `python train_nnue.py train` first")
    H = int(d["W1"].shape[1])
    with open(bin_path, "wb") as f:
        f.write(b"NNU2")
        f.write(np.array([H], dtype="<i4").tobytes())
        for k in ("W1", "b1", "W2", "b2", "W3", "b3", "W4"):
            f.write(np.ascontiguousarray(d[k], dtype="<f4").tobytes())
        f.write(np.asarray(d["b4"], dtype="<f4").reshape(-1)[:1].tobytes())

def find_nnue(model_path=None):
    """Locate nnue.npz: explicit .npz path, next to model_path, or engine/models/saved/."""
    cands = []
    if model_path:
        p = Path(model_path)
        cands.append(p if p.suffix == ".npz" else p.parent / "nnue.npz")
    cands.append(Path(__file__).resolve().parent / "saved" / "nnue.npz")
    cands.append(Path("engine/models/saved/nnue.npz"))
    for c in cands:
        if c.is_file():
            return str(c)
    return None