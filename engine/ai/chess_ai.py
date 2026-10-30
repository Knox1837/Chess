"""
Chess AI - classical alpha-beta engine (drop-in replacement for the old ChessAI).
"""
import atexit
import math
import os
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.engine

try:
    from engine.models.nnue import load_nnue, find_nnue, export_bin
except ImportError:          # runs fine without the NNUE module
    load_nnue = None
    export_bin = None

    def find_nnue(_=None):
        return None

#  Evaluation tables (PeSTO). Index 0 = a8 ... 63 = h1, from White's viewpoint.
MG_VALUE = [0, 82, 337, 365, 477, 1025, 0]
EG_VALUE = [0, 94, 281, 297, 512, 936, 0]
PHASE_INC = [0, 0, 1, 1, 2, 4, 0]

_MG_PST = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        98, 134, 61, 95, 68, 126, 34, -11,
        -6, 7, 26, 31, 65, 56, 25, -20,
        -14, 13, 6, 21, 23, 12, 17, -23,
        -27, -2, -5, 12, 17, 6, 10, -25,
        -26, -4, -4, -10, 3, 3, 33, -12,
        -35, -1, -20, -23, -15, 24, 38, -22,
        0, 0, 0, 0, 0, 0, 0, 0],
    chess.KNIGHT: [
        -167, -89, -34, -49, 61, -97, -15, -107,
        -73, -41, 72, 36, 23, 62, 7, -17,
        -47, 60, 37, 65, 84, 129, 73, 44,
        -9, 17, 19, 53, 37, 69, 18, 22,
        -13, 4, 16, 13, 28, 19, 21, -8,
        -23, -9, 12, 10, 19, 17, 25, -16,
        -29, -53, -12, -3, -1, 18, -14, -19,
        -105, -21, -58, -33, -17, -28, -19, -23],
    chess.BISHOP: [
        -29, 4, -82, -37, -25, -42, 7, -8,
        -26, 16, -18, -13, 30, 59, 18, -47,
        -16, 37, 43, 40, 35, 50, 37, -2,
        -4, 5, 19, 50, 37, 37, 7, -2,
        -6, 13, 13, 26, 34, 12, 10, 4,
        0, 15, 15, 15, 14, 27, 18, 10,
        4, 15, 16, 0, 7, 21, 33, 1,
        -33, -3, -14, -21, -13, -12, -39, -21],
    chess.ROOK: [
        32, 42, 32, 51, 63, 9, 31, 43,
        27, 32, 58, 62, 80, 67, 26, 44,
        -5, 19, 26, 36, 17, 45, 61, 16,
        -24, -11, 7, 26, 24, 35, -8, -20,
        -36, -26, -12, -1, 9, -7, 6, -23,
        -45, -25, -16, -17, 3, 0, -5, -33,
        -44, -16, -20, -9, -1, 11, -6, -71,
        -19, -13, 1, 17, 16, 7, -37, -26],
    chess.QUEEN: [
        -28, 0, 29, 12, 59, 44, 43, 45,
        -24, -39, -5, 1, -16, 57, 28, 54,
        -13, -17, 7, 8, 29, 56, 47, 57,
        -27, -27, -16, -16, -1, 17, -2, 1,
        -9, -26, -9, -10, -2, -4, 3, -3,
        -14, 2, -11, -2, -5, 2, 14, 5,
        -35, -8, 11, 2, 8, 15, -3, 1,
        -1, -18, -9, 10, -15, -25, -31, -50],
    chess.KING: [
        -65, 23, 16, -15, -56, -34, 2, 13,
        29, -1, -20, -7, -8, -4, -38, -29,
        -9, 24, 2, -16, -20, 6, 22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49, -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1, 7, -8, -64, -43, -16, 9, 8,
        -15, 36, 12, -54, 8, -28, 24, 14],
}
_EG_PST = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100, 85, 67, 56, 53, 82, 84,
        32, 24, 13, 5, -2, 4, 17, 17,
        13, 9, -3, -7, -7, -8, 3, -1,
        4, 7, -6, 1, 0, -5, -1, -8,
        13, 8, 8, 10, 13, 0, 2, -7,
        0, 0, 0, 0, 0, 0, 0, 0],
    chess.KNIGHT: [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25, -8, -25, -2, -9, -25, -24, -52,
        -24, -20, 10, 9, -1, -9, -19, -41,
        -17, 3, 22, 22, 22, 11, 8, -18,
        -18, -6, 16, 25, 16, 17, 4, -18,
        -23, -3, -1, 15, 10, -3, -20, -22,
        -42, -20, -10, -5, -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64],
    chess.BISHOP: [
        -14, -21, -11, -8, -7, -9, -17, -24,
        -8, -4, 7, -12, -3, -13, -4, -14,
        2, -8, 0, -1, -2, 6, 0, 4,
        -3, 9, 12, 9, 14, 10, 3, 2,
        -6, 3, 13, 19, 7, 10, -3, -9,
        -12, -3, 8, 10, 13, 3, -7, -15,
        -14, -18, -7, -1, 4, -9, -15, -27,
        -23, -9, -23, -5, -9, -16, -5, -17],
    chess.ROOK: [
        13, 10, 18, 15, 12, 12, 8, 5,
        11, 13, 13, 11, -3, 3, 8, 3,
        7, 7, 7, 5, 4, -3, -5, -3,
        4, 3, 13, 1, 2, 1, -1, 2,
        3, 5, 8, 4, -5, -6, -8, -11,
        -4, 0, -5, -1, -7, -12, -8, -16,
        -6, -6, 0, 2, -9, -9, -11, -3,
        -9, 2, 3, -1, -5, -13, 4, -20],
    chess.QUEEN: [
        -9, 22, 22, 27, 27, 19, 10, 20,
        -17, 20, 32, 41, 58, 25, 30, 0,
        -20, 6, 9, 49, 47, 35, 19, 9,
        3, 22, 24, 45, 57, 40, 57, 36,
        -18, 28, 19, 47, 31, 34, 39, 23,
        -16, -27, 15, 6, 9, 17, 10, 5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43, -5, -32, -20, -41],
    chess.KING: [
        -74, -35, -18, -18, -11, 15, 4, -17,
        -12, 17, 14, 17, 17, 38, 23, 11,
        10, 17, 23, 15, 20, 45, 44, 13,
        -8, 22, 24, 27, 26, 33, 26, 3,
        -18, -4, 21, 24, 27, 23, 9, -11,
        -19, -3, 11, 21, 23, 16, 7, -9,
        -27, -11, 4, 13, 14, 4, -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43],
}

# Final per-colour lookup tables indexed [colour][piece_type][python-chess square]
# colour 0 = White, 1 = Black. White flips the rank (sq ^ 56); Black uses sq directly.
_MG = [[None] * 7 for _ in range(2)]
_EG = [[None] * 7 for _ in range(2)]
for _pt in range(1, 7):
    _MG[0][_pt] = [MG_VALUE[_pt] + _MG_PST[_pt][sq ^ 56] for sq in range(64)]
    _EG[0][_pt] = [EG_VALUE[_pt] + _EG_PST[_pt][sq ^ 56] for sq in range(64)]
    _MG[1][_pt] = [MG_VALUE[_pt] + _MG_PST[_pt][sq] for sq in range(64)]
    _EG[1][_pt] = [EG_VALUE[_pt] + _EG_PST[_pt][sq] for sq in range(64)]

# Pawn-structure / king-shield masks
_FILES = chess.BB_FILES
_ADJ_FILES = [
    (_FILES[f - 1] if f > 0 else 0) | (_FILES[f + 1] if f < 7 else 0) for f in range(8)
]

def _ranks_ahead(rank, white):
    m = 0
    rng = range(rank + 1, 8) if white else range(0, rank)
    for r in rng:
        m |= chess.BB_RANKS[r]
    return m

_PASSED_MASK = [[0] * 64, [0] * 64]   # squares that must hold no enemy pawn
_FRONT_SAME = [[0] * 64, [0] * 64]    # same-file squares in front (doubled test)
_SHIELD = [[0] * 64, [0] * 64]
for _sq in range(64):
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    for _ci, _white in ((0, True), (1, False)):
        _ahead = _ranks_ahead(_r, _white)
        _PASSED_MASK[_ci][_sq] = (_FILES[_f] | _ADJ_FILES[_f]) & _ahead
        _FRONT_SAME[_ci][_sq] = _FILES[_f] & _ahead
        _step = 1 if _white else -1
        _m = 0
        for _dr in (1, 2):
            _rr = _r + _step * _dr
            if 0 <= _rr < 8:
                for _ff in (_f - 1, _f, _f + 1):
                    if 0 <= _ff < 8:
                        _m |= chess.BB_SQUARES[chess.square(_ff, _rr)]
        _SHIELD[_ci][_sq] = _m

PASSED_MG = [0, 0, 5, 10, 20, 35, 60, 0]     # by pawn rank from own side (0-7)
PASSED_EG = [0, 5, 15, 30, 55, 90, 140, 0]

INF = 10 ** 6
MATE = 100_000
MAX_PLY = 64
_popcount = chess.popcount
_scan = chess.scan_forward

def evaluate(board):
    """Static evaluation in centipawns from the SIDE-TO-MOVE's point of view."""
    mg = [0, 0]
    eg = [0, 0]
    phase = 0
    pawns_bb = [board.pieces_mask(chess.PAWN, True), board.pieces_mask(chess.PAWN, False)]
    for ci in (0, 1):
        color = ci == 0
        m = e = 0
        mgt, egt = _MG[ci], _EG[ci]
        own_p, opp_p = pawns_bb[ci], pawns_bb[1 - ci]
        for pt in (1, 2, 3, 4, 5, 6):
            bb = board.pieces_mask(pt, color)
            if not bb:
                continue
            tm, te = mgt[pt], egt[pt]
            for sq in _scan(bb):
                m += tm[sq]
                e += te[sq]
                phase += PHASE_INC[pt]
                if pt == 1:
                    r = sq >> 3 if color else 7 - (sq >> 3)
                    f = sq & 7
                    if not (opp_p & _PASSED_MASK[ci][sq]):
                        m += PASSED_MG[r]
                        e += PASSED_EG[r]
                    if not (own_p & _ADJ_FILES[f]):
                        m -= 10
                        e -= 15
                    if own_p & _FRONT_SAME[ci][sq]:
                        m -= 12
                        e -= 20
                elif pt == 4:
                    fm = _FILES[sq & 7]
                    if not (own_p & fm):
                        if not (opp_p & fm):
                            m += 20
                            e += 10
                        else:
                            m += 10
                            e += 5
            if pt == 3 and _popcount(bb) >= 2:
                m += 30
                e += 50
            elif pt == 6:
                m += 9 * _popcount(own_p & _SHIELD[ci][bb.bit_length() - 1])
        mg[ci] = m
        eg[ci] = e
    if phase > 24:
        phase = 24
    mg_s = mg[0] - mg[1]
    eg_s = eg[0] - eg[1]
    score = int((mg_s * phase + eg_s * (24 - phase)) / 24)   # truncate toward 0 => colour-symmetric
    return (score + 10) if board.turn else (-score + 10)

#  Skill levels:  (max depth, time limit s, noise in centipawns)
#  noise > 0  -> every root move is scored and perturbed (human-like errors)
LEVELS = {
    1: (1, 0.3, 220),
    2: (2, 0.6, 90),
    3: (3, 1.2, 35),
    4: (10, 2.5, 0),
    5: (30, 6.0, 0),
}
_VAL = [0, 100, 320, 330, 500, 900, 20000]

# C++ backend levels: (max depth, seconds per move or None, noise in centipawns)
CPP_LEVELS = {
    1: (1, None, 220),
    2: (2, None, 90),
    3: (4, None, 40),
    4: (40, 0.6, 0),
    5: (60, 2.5, 0),
}

def find_cpp_engine():
    """engine/bin/chess_engine(.exe) or engine/cpp/chess_engine(.exe), if it has been built."""
    base = Path(__file__).resolve().parents[1]
    exe = "chess_engine.exe" if os.name == "nt" else "chess_engine"
    for d in (base / "bin", base / "cpp"):
        if (d / exe).is_file():
            return str(d / exe)
    return None

class ChessAI:
    def __init__(self, model_path=None, skill_level=1, use_nnue=False, time_limit=None, nnue_blend=1.0,
                 use_cpp=True):
        self.skill_level = max(1, min(5, int(skill_level)))
        self.time_limit = time_limit  # overrides the per-level time budget (seconds)
        self.model = None            # kept so old code that checks `ai.model` still works
        self.nnue = None
        self._eval = evaluate
        self.cpp = None
        self._lock = threading.Lock()
        self._game = object()
        if use_cpp and nnue_blend >= 1.0:
            self._start_cpp(model_path, use_nnue)
        if self.cpp is None and use_nnue and load_nnue is not None:
            path = find_nnue(model_path)
            if path:
                try:
                    self.nnue = load_nnue(path)
                    nn_eval = self.nnue.evaluate
                    w = float(nnue_blend)
                    if w < 1.0:                # e.g. 0.7 = 70% network + 30% hand-written eval
                        raw = lambda b: int(w * nn_eval(b) + (1.0 - w) * evaluate(b))
                    else:
                        raw = nn_eval
                    # The network is the slow part of the search, and the same leaf positions come back again (every deeper iteration, transpositions), so remember its answers.
                    cache = self._eval_cache = {}
                    self.cache_hits = self.cache_misses = 0

                    def cached(b):
                        key = (b.occupied_co[True], b.occupied_co[False], b.pawns, b.knights,
                               b.bishops, b.rooks, b.queens, b.turn)
                        v = cache.get(key)
                        if v is None:
                            self.cache_misses += 1
                            if len(cache) > 1_500_000:
                                cache.clear()
                            v = cache[key] = raw(b)
                        else:
                            self.cache_hits += 1
                        return v
                    self._eval = cached
                except Exception as e:           # bad/old file -> fall back silently to hand eval
                    print(f"AI: could not load NNUE ({e}); using hand-written eval")
        self.tt = {}
        self.nodes = 0
        self.stop = False
        self.deadline = 0.0
        self.killers = []
        self.history = {}
        self.last_info = {}
        kind = f"NNUE ({self.nnue.hidden} hidden)" if self.nnue else "hand-written eval"
        backend = "C++ engine" if self.cpp is not None else "Python engine"
        print(f"AI: {backend}, {kind}, skill level {self.skill_level}")

    #  C++ backend (UCI)
    def _start_cpp(self, model_path, use_nnue):
        exe = find_cpp_engine()
        if not exe:
            return
        try:
            eng = chess.engine.SimpleEngine.popen_uci(exe)
            cfg = {"Hash": 64, "UseNNUE": False}
            if use_nnue and find_nnue is not None and export_bin is not None:
                npz = find_nnue(model_path)
                if npz:
                    try:
                        binp = Path(npz).with_suffix(".bin")
                        if not binp.is_file() or binp.stat().st_mtime < Path(npz).stat().st_mtime:
                            export_bin(npz, str(binp))
                        import numpy as np
                        self.nnue = SimpleNamespace(hidden=int(np.load(npz)["W1"].shape[1]))
                        cfg.update({"UseNNUE": True, "EvalFile": str(binp)})
                    except Exception as e:      # e.g. an old v1 network: the C++ engine can't use it
                        print(f"AI: not using the network in the C++ engine ({e})")
            eng.configure(cfg)
            self.cpp = eng
            atexit.register(self.close)
        except Exception as e:
            print(f"AI: could not start the C++ engine ({e}); using the Python engine")
            self.cpp = None

    def new_game(self):
        """Forget the transposition table (call between games)."""
        self._game = object()
        self.tt.clear()

    def close(self):
        eng, self.cpp = self.cpp, None
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                pass

    def _cpp_choose(self, board):
        with self._lock:
            if not any(board.legal_moves):
                return None
            depth, secs, sigma = CPP_LEVELS[self.skill_level]
            if self.time_limit and sigma == 0:
                secs = self.time_limit
            if sigma == 0 and board.fullmove_number <= 3:       # opening variety, as in the Python engine
                depth, secs, sigma = 3, None, 18
            self.cpp.configure({"Noise": sigma})
            t0 = time.perf_counter()
            res = self.cpp.play(board, chess.engine.Limit(depth=depth, time=secs),
                                game=self._game, info=chess.engine.INFO_ALL)
        if res.move is None:
            return None
        sc = res.info.get("score")
        if sc is None:
            eval_pawns, display = 0.0, "+0.00"
        else:
            w = sc.white()
            if w.is_mate():
                m = w.mate()
                eval_pawns = 99.0 if m > 0 else -99.0
                display = f"#{m}" if m > 0 else f"#-{abs(m)}"
            else:
                eval_pawns = w.score() / 100.0
                display = f"{eval_pawns:+.2f}"
        self.last_info = {"depth": res.info.get("depth", 0), "nodes": res.info.get("nodes", 0),
                          "seconds": round(time.perf_counter() - t0, 2)}
        return {"move": res.move, "eval": eval_pawns, "eval_display": display,
                "move_san": board.san(res.move)}

    #  Compatibility helpers
    def evaluate_position(self, board):
        """Evaluation in pawns from White's perspective."""
        cp = self._eval(board)
        return (cp if board.turn == chess.WHITE else -cp) / 100.0

    evaluate_position_heuristic = evaluate_position

    #  Move ordering
    def _order(self, board, moves, tt_move, ply):
        killers = self.killers[ply] if ply < len(self.killers) else (None, None)
        hist = self.history
        scored = []
        for m in moves:
            if m == tt_move:
                s = 10_000_000
            elif board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or chess.PAWN  # en passant
                attacker = board.piece_type_at(m.from_square)
                s = 1_000_000 + 10 * _VAL[victim] - _VAL[attacker] // 10
                if m.promotion:
                    s += 5000
            elif m.promotion:
                s = 900_000 + m.promotion
            elif m == killers[0]:
                s = 800_000
            elif m == killers[1]:
                s = 790_000
            else:
                s = hist.get((m.from_square, m.to_square), 0)
            scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    #  Quiescence
    def _quiesce(self, board, alpha, beta, ply, qd):
        self.nodes += 1
        if (self.nodes & 1023) == 0 and time.perf_counter() > self.deadline:
            self.stop = True
        if self.stop:
            return 0
        if ply >= MAX_PLY:
            return self._eval(board)

        in_check = board.is_check()
        if in_check and qd < 3:
            moves = self._order(board, list(board.legal_moves), None, ply)
            if not moves:
                return -MATE + ply
            stand = -INF
        else:
            stand = self._eval(board)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            # Delta pruning: even winning a queen cannot reach alpha
            if stand + 1000 < alpha:
                return stand
            moves = self._order(board, list(board.generate_legal_captures()), None, ply)

        best = stand
        for m in moves:
            if not in_check:
                victim = board.piece_type_at(m.to_square) or chess.PAWN
                if stand + _VAL[victim] + 200 < alpha and not m.promotion:
                    continue
            board.push(m)
            score = -self._quiesce(board, -beta, -alpha, ply + 1, qd + 1)
            board.pop()
            if self.stop:
                return 0
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    #  Main search
    def _negamax(self, board, depth, alpha, beta, ply, allow_null):
        self.nodes += 1
        if (self.nodes & 1023) == 0 and time.perf_counter() > self.deadline:
            self.stop = True
        if self.stop:
            return 0

        if ply > 0:
            if board.halfmove_clock >= 100:
                return 0
            if board.halfmove_clock >= 4 and board.is_repetition(2):
                return 0
            if _popcount(board.occupied) <= 4 and board.is_insufficient_material():
                return 0
        if ply >= MAX_PLY:
            return self._eval(board)

        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply, 0)

        # mate-distance pruning
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha

        orig_alpha = alpha
        key = board._transposition_key()
        tt_move = None
        entry = self.tt.get(key)
        if entry is not None:
            e_depth, flag, e_score, tt_move = entry
            if e_depth >= depth:
                if e_score > MATE - 1000:
                    e_score -= ply
                elif e_score < -MATE + 1000:
                    e_score += ply
                if flag == 0:
                    return e_score
                if flag == 1 and e_score > alpha:      # lower bound
                    alpha = e_score
                elif flag == 2 and e_score < beta:     # upper bound
                    beta = e_score
                if alpha >= beta:
                    return e_score

        # null-move pruning
        if (allow_null and not in_check and depth >= 3 and beta < MATE - 1000
                and board.occupied_co[board.turn] & ~(board.pawns | board.kings)):
            board.push(chess.Move.null())
            score = -self._negamax(board, depth - 3 - (depth > 6), -beta, -beta + 1, ply + 1, False)
            board.pop()
            if self.stop:
                return 0
            if score >= beta:
                return beta

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
        moves = self._order(board, moves, tt_move, ply)

        best = -INF
        best_move = None
        for i, m in enumerate(moves):
            is_quiet = not board.is_capture(m) and not m.promotion
            board.push(m)
            if i == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
            else:
                red = 0
                if depth >= 3 and i >= 4 and is_quiet and not in_check and not board.is_check():
                    red = 2 if (i >= 10 and depth >= 5) else 1
                score = -self._negamax(board, depth - 1 - red, -alpha - 1, -alpha, ply + 1, True)
                if score > alpha and (red or score < beta):
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
            board.pop()
            if self.stop:
                return 0
            if score > best:
                best = score
                best_move = m
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if is_quiet:
                            k = self.killers[ply]
                            if k[0] != m:
                                k[1] = k[0]
                                k[0] = m
                            hk = (m.from_square, m.to_square)
                            self.history[hk] = self.history.get(hk, 0) + depth * depth
                        break

        if best <= orig_alpha:
            flag = 2
        elif best >= beta:
            flag = 1
        else:
            flag = 0
        stored = best
        if stored > MATE - 1000:
            stored += ply
        elif stored < -MATE + 1000:
            stored -= ply
        self.tt[key] = (depth, flag, stored, best_move)
        return best

    #  Root drivers
    def _reset_search(self, time_limit):
        self.nodes = 0
        self.stop = False
        self.deadline = time.perf_counter() + time_limit
        self.killers = [[None, None] for _ in range(MAX_PLY + 2)]
        self.history = {}
        if len(self.tt) > 1_500_000:
            self.tt.clear()

    def _iterative(self, board, moves, max_depth, time_limit):
        """Returns (best_move, score_for_side_to_move, depth_reached)."""
        self._reset_search(time_limit)
        t0 = time.perf_counter()
        best_move, best_score, reached = moves[0], 0, 0
        for depth in range(1, max_depth + 1):
            ordered = self._order(board, moves, best_move, 0)
            alpha = -INF
            it_best, it_score = None, -INF
            for i, m in enumerate(ordered):
                board.push(m)
                if i == 0:
                    s = -self._negamax(board, depth - 1, -INF, -alpha, 1, True)
                else:
                    s = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1, True)
                    if s > alpha:
                        s = -self._negamax(board, depth - 1, -INF, -alpha, 1, True)
                board.pop()
                if self.stop:
                    break
                if s > it_score:
                    it_score, it_best = s, m
                if s > alpha:
                    alpha = s
            if it_best is not None:
                best_move, best_score = it_best, it_score
                if not self.stop:
                    reached = depth
            if self.stop or abs(best_score) > MATE - 100:
                break
            if time.perf_counter() - t0 > time_limit * 0.45:   # next ply would not finish
                break
        return best_move, best_score, reached

    def _noisy_root(self, board, moves, depth, sigma, time_limit):
        """Score every root move at fixed depth, add gaussian noise, pick the best."""
        self._reset_search(time_limit)
        ordered = self._order(board, moves, None, 0)
        results = []
        for m in ordered:
            board.push(m)
            s = -self._negamax(board, depth - 1, -INF, INF, 1, True)
            board.pop()
            if self.stop:
                break
            results.append((s, m))
        if not results:
            return ordered[0], 0, depth
        # Never blunder into a forced mate if a non-losing alternative exists
        scored = [(s + random.gauss(0, sigma) if s > -MATE + 1000 else s - 10_000, s, m)
                  for s, m in results]
        scored.sort(key=lambda x: x[0], reverse=True)
        _, true_score, move = scored[0]
        return move, true_score, depth

    def choose_move(self, board):
        if self.cpp is not None:
            try:
                return self._cpp_choose(board)
            except Exception as e:
                print(f"AI: C++ engine failed ({e}); switching to the Python engine")
                self.close()
        return self._py_choose_move(board)

    def _py_choose_move(self, board):
        board = board.copy(stack=True)
        moves = list(board.legal_moves)
        if not moves:
            return None

        max_depth, limit, sigma = LEVELS[self.skill_level]
        if self.time_limit:
            limit = self.time_limit
        t0 = time.perf_counter()

        if len(moves) == 1:
            move, score, depth = moves[0], 0, 0
        elif sigma > 0:
            move, score, depth = self._noisy_root(board, moves, max_depth, sigma, limit)
        elif board.fullmove_number <= 3:
            # Opening variety: shallow noisy pick so the engine doesn't play the same game every time
            move, score, depth = self._noisy_root(board, moves, 3, 18, limit)
        else:
            move, score, depth = self._iterative(board, moves, max_depth, limit)

        # side-to-move score -> White's perspective
        white_score = score if board.turn == chess.WHITE else -score
        if abs(score) > MATE - 100:
            mate_in = (MATE - abs(score) + 1) // 2
            display = f"#{mate_in}" if white_score > 0 else f"#-{mate_in}"
        else:
            display = f"{white_score / 100:+.2f}"
        self.last_info = {
            "depth": depth, "nodes": self.nodes,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        return {
            'move': move,
            'eval': white_score / 100.0,
            'eval_display': display,
            'move_san': board.san(move),
        }