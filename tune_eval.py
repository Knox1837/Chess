"""
Tune the chess engine's hand-written evaluation on the prepared Lichess positions (NumPy only, no PyTorch).

  python tune_eval.py check                       # features match the engine's own evaluation? (run this first)
  python tune_eval.py tune                        # fit and write engine/models/saved/eval_params.txt
  python tune_eval.py tune --max-positions 3000000 --epochs 60
  python tune_eval.py tune --no-new-terms         # only re-fit the existing terms (mobility/king zone stay 0)
  python tune_eval.py tune --no-new-terms --max-change 15    # ...and keep every weight within +-15 cp of today's
  python tune_eval.py tune --only-pst --max-change 15        # only the piece-square tables (plus material values)
  python tune_eval.py bake --params engine\\models\\saved\\p4.txt   # make a params file the engine's built-in default
  python tune_eval.py verify --params engine\\models\\saved\\p4.txt # after rebuilding: engine defaults == that file?

Test the result with, for example:
  python match_engines.py --a engine\\bin\\chess_engine.exe --b engine\\bin\\chess_engine.exe ^ --a-opt EvalParams=engine\\models\\saved\\eval_params.txt --games 300 --movetime 0.1 --concurrency 4
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from engine.models.nnue import SCALE, pack_board
except Exception:                       # keep working without the package (check mode needs pack_board)
    SCALE, pack_board = 400.0, None

DEFAULT_DATA = ["engine/data/processed/nnue", "engine/data/processed/nnue2"]
DEFAULT_OUT = "engine/models/saved/eval_params.txt"

# Feature layout: 384 piece-square | 6 passed pawn (relative ranks 1-6) | isolated, doubled, rook open,
# rook semi-open, bishop pair, king shield | 4 mobility (N,B,R,Q) | 4 king-zone attacks (N,B,R,Q)
N_PST, N_PASSED, N_SCALAR, N_MOB, N_KZ = 384, 6, 6, 4, 4
F = N_PST + N_PASSED + N_SCALAR + N_MOB + N_KZ
SCALAR_NAMES = ["isolated", "doubled", "rook_open", "rook_semi", "bishop_pair", "shield"]
NEW_TERMS = slice(N_PST + N_PASSED + N_SCALAR, F)          # mobility + king zone
PHASE_INC = (0, 1, 1, 2, 4, 0)

U = np.uint64
FILE_A = U(0x0101010101010101)
FILE_B = FILE_A << U(1)
FILE_G = FILE_A << U(6)
FILE_H = FILE_A << U(7)

# engine parameter files
def find_engine(path=None):
    for cand in ([path] if path else []) + ["engine/bin/chess_engine.exe", "engine/bin/chess_engine"]:
        if cand and Path(cand).is_file():
            return str(cand)
    sys.exit("Engine not found. Build it first (cmake) or pass --engine.")

def run_engine(exe, commands):
    out = subprocess.run([exe], input="\n".join(commands) + "\nquit\n", capture_output=True, text=True, timeout=600)
    return out.stdout

def parse_params(text):
    d = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] in ("version", "end") or parts[0].startswith("#"):
            continue
        if parts[0] in ("pst_mg", "pst_eg"):
            d[(parts[0], int(parts[1]))] = [int(v) for v in parts[2:]]
        else:
            d[parts[0]] = [int(v) for v in parts[1:]]
    return d

def to_vectors(d):
    """Parameter dict -> (mg weights, eg weights, tempo) in feature order."""
    mg, eg = [], []
    for key, vec in (("pst_mg", mg), ("pst_eg", eg)):
        for pt in range(1, 7):
            vec.extend(d[(key, pt)])
    mg.extend(d["passed_mg"][1:7])
    eg.extend(d["passed_eg"][1:7])
    for name in SCALAR_NAMES:
        mg.append(d[name][0])
        eg.append(d[name][1])
    mg.extend(d["mob_mg"])
    eg.extend(d["mob_eg"])
    mg.extend(d["kz_mg"])
    eg.extend(d["kz_eg"])
    return np.array(mg, np.float64), np.array(eg, np.float64), float(d["tempo"][0])

def write_params(base, mg, eg, tempo, path):
    """Write a parameter file: `base` (dict) supplies everything the tuner does not change."""
    mg, eg = np.rint(mg).astype(int), np.rint(eg).astype(int)
    d = {k: list(v) for k, v in base.items()}
    for key, vec in (("pst_mg", mg), ("pst_eg", eg)):
        for pt in range(1, 7):
            d[(key, pt)] = [int(v) for v in vec[(pt - 1) * 64:pt * 64]]
    d["passed_mg"][1:7] = [int(v) for v in mg[N_PST:N_PST + 6]]
    d["passed_eg"][1:7] = [int(v) for v in eg[N_PST:N_PST + 6]]
    for i, name in enumerate(SCALAR_NAMES):
        d[name] = [int(mg[N_PST + 6 + i]), int(eg[N_PST + 6 + i])]
    k = N_PST + N_PASSED + N_SCALAR
    d["mob_mg"], d["mob_eg"] = [int(v) for v in mg[k:k + 4]], [int(v) for v in eg[k:k + 4]]
    d["kz_mg"], d["kz_eg"] = [int(v) for v in mg[k + 4:k + 8]], [int(v) for v in eg[k + 4:k + 8]]
    d["tempo"] = [int(round(tempo))]
    lines = ["version 1"]
    for key in ("pst_mg", "pst_eg"):
        for pt in range(1, 7):
            lines.append(f"{key} {pt} " + " ".join(map(str, d[(key, pt)])))
    for name in ("passed_mg", "passed_eg", "isolated", "doubled", "rook_open", "rook_semi", "bishop_pair",
                 "shield", "tempo", "mob_mg", "mob_eg", "kz_mg", "kz_eg"):
        lines.append(name + " " + " ".join(map(str, d[name])))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")

# features
def _masks():
    pm_w, pm_b, sh_w, sh_b = (np.zeros((64, 64), np.float32) for _ in range(4))
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        for t in range(64):
            tf, tr = t & 7, t >> 3
            if abs(tf - f) <= 1:
                if tr > r:
                    pm_w[sq, t] = 1
                if tr < r:
                    pm_b[sq, t] = 1
                if tr in (r + 1, r + 2):
                    sh_w[sq, t] = 1
                if tr in (r - 1, r - 2):
                    sh_b[sq, t] = 1
    return pm_w, pm_b, sh_w, sh_b

PM_W, PM_B, SH_W, SH_B = _masks()
PERM56 = np.arange(64) ^ 56

def unpack(x):
    """uint64 array (...) -> uint8 array (..., 64); element [..., sq] is bit sq (a1 = 0)."""
    x = np.ascontiguousarray(x)
    return np.unpackbits(x.view(np.uint8).reshape(x.shape + (8,)), axis=-1, bitorder="little")

def popcount(x):
    return unpack(x).sum(axis=-1, dtype=np.int32)

def _shl(x, n):
    return np.left_shift(x, U(n))

def _shr(x, n):
    return np.right_shift(x, U(n))

# (left?, shift, mask applied after the shift) for the 8 sliding directions
DIRS = {
    "N": (True, 8, None), "S": (False, 8, None),
    "E": (True, 1, ~FILE_A), "W": (False, 1, ~FILE_H),
    "NE": (True, 9, ~FILE_A), "NW": (True, 7, ~FILE_H),
    "SE": (False, 7, ~FILE_A), "SW": (False, 9, ~FILE_H),
}
ROOK_DIRS = ("N", "S", "E", "W")
BISHOP_DIRS = ("NE", "NW", "SE", "SW")

def _step(x, d):
    left, n, mask = DIRS[d]
    y = _shl(x, n) if left else _shr(x, n)
    return y if mask is None else y & mask

def sliding(gen, empty, dirs):
    """Squares attacked by sliders in `gen` (blockers included), union over all the pieces."""
    att = np.zeros_like(gen)
    for d in dirs:
        cur = gen
        for _ in range(7):
            cur = _step(cur, d)
            att |= cur
            cur = cur & empty
    return att

def knight_attacks(x):
    return ((_shl(x, 17) & ~FILE_A) | (_shl(x, 15) & ~FILE_H) | (_shl(x, 10) & ~(FILE_A | FILE_B)) |
            (_shl(x, 6) & ~(FILE_G | FILE_H)) | (_shr(x, 17) & ~FILE_H) | (_shr(x, 15) & ~FILE_A) |
            (_shr(x, 10) & ~(FILE_G | FILE_H)) | (_shr(x, 6) & ~(FILE_A | FILE_B)))

def king_attacks(x):
    out = np.zeros_like(x)
    for d in DIRS:
        out |= _step(x, d)
    return out

def features(bb):
    """bb: (n, 12) uint64 planes in the mover's frame (own P N B R Q K, then the opponent's, own side moves up).
    Returns (X int8 (n, F), phase int8 (n,))."""
    n = len(bb)
    bits = unpack(bb)                                   # (n, 12, 64) uint8
    own, opp = bits[:, :6], bits[:, 6:]
    X = np.zeros((n, F), np.int8)
    X[:, :N_PST] = (own[:, :, PERM56].astype(np.int8) - opp.astype(np.int8)).reshape(n, N_PST)

    p_own, p_opp = own[:, 0].astype(np.float32), opp[:, 0].astype(np.float32)
    passed_own = p_own * ((p_opp @ PM_W.T) == 0)
    passed_opp = p_opp * ((p_own @ PM_B.T) == 0)
    by_rank = passed_own.reshape(n, 8, 8).sum(2) - passed_opp.reshape(n, 8, 8).sum(2)[:, ::-1]
    X[:, N_PST:N_PST + 6] = by_rank[:, 1:7]

    own_f, opp_f = p_own.reshape(n, 8, 8).sum(1), p_opp.reshape(n, 8, 8).sum(1)      # pawns per file

    def isolated(f):
        adj = np.zeros_like(f)
        adj[:, 1:] += f[:, :-1]
        adj[:, :-1] += f[:, 1:]
        return (f * (adj == 0)).sum(1)

    base = N_PST + N_PASSED
    X[:, base + 0] = isolated(own_f) - isolated(opp_f)
    X[:, base + 1] = np.maximum(own_f - 1, 0).sum(1) - np.maximum(opp_f - 1, 0).sum(1)
    r_own = own[:, 3].astype(np.float32).reshape(n, 8, 8).sum(1)                       # rooks per file
    r_opp = opp[:, 3].astype(np.float32).reshape(n, 8, 8).sum(1)
    X[:, base + 2] = (r_own * (own_f == 0) * (opp_f == 0)).sum(1) - (r_opp * (opp_f == 0) * (own_f == 0)).sum(1)
    X[:, base + 3] = (r_own * (own_f == 0) * (opp_f > 0)).sum(1) - (r_opp * (opp_f == 0) * (own_f > 0)).sum(1)
    X[:, base + 4] = (own[:, 2].sum(1) >= 2).astype(np.int8) - (opp[:, 2].sum(1) >= 2).astype(np.int8)
    ks_own, ks_opp = own[:, 5].argmax(1), opp[:, 5].argmax(1)
    X[:, base + 5] = (SH_W[ks_own] * p_own).sum(1) - (SH_B[ks_opp] * p_opp).sum(1)

    # mobility and king-zone attacks from the union of squares each piece type attacks
    planes = {c: [bb[:, i] for i in range(c * 6, c * 6 + 6)] for c in (0, 1)}
    occ = np.bitwise_or.reduce(bb, axis=1)
    empty = ~occ
    att = {}
    for c in (0, 1):
        pl = planes[c]
        att[c] = [knight_attacks(pl[1]),
                  sliding(pl[2], empty, BISHOP_DIRS),
                  sliding(pl[3], empty, ROOK_DIRS),
                  sliding(pl[4], empty, ROOK_DIRS + BISHOP_DIRS)]
    pa_own = (_shl(planes[0][0], 9) & ~FILE_A) | (_shl(planes[0][0], 7) & ~FILE_H)
    pa_opp = (_shr(planes[1][0], 7) & ~FILE_A) | (_shr(planes[1][0], 9) & ~FILE_H)
    area_own = ~(planes[0][0] | planes[0][5]) & ~pa_opp
    area_opp = ~(planes[1][0] | planes[1][5]) & ~pa_own
    zone_own = king_attacks(planes[0][5]) | planes[0][5]
    zone_opp = king_attacks(planes[1][5]) | planes[1][5]
    k = base + 6
    for t in range(4):
        X[:, k + t] = popcount(att[0][t] & area_own) - popcount(att[1][t] & area_opp)
        X[:, k + 4 + t] = popcount(att[0][t] & zone_opp) - popcount(att[1][t] & zone_own)

    counts = bits.sum(axis=2)                           # (n, 12)
    phase = sum(PHASE_INC[pt] * (counts[:, pt] + counts[:, pt + 6]) for pt in range(6))
    return X, np.minimum(phase, 24).astype(np.int8)

def predict(X, phase, mg, eg, tempo):
    pm = (phase.astype(np.float32) / 24.0)
    Xf = X.astype(np.float32)
    return (Xf @ mg.astype(np.float32)) * pm + (Xf @ eg.astype(np.float32)) * (1.0 - pm) + tempo

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

# check
def random_boards(n, seed):
    import chess
    rng = random.Random(seed)
    boards = []
    while len(boards) < n:
        b = chess.Board()
        for _ in range(rng.randint(6, 90)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over() and not b.is_check():
            boards.append(b)
    return boards

def cmd_check(a):
    """Compare the NumPy features with the engine's own evaluation on random positions (both sides to move)."""
    if pack_board is None:
        sys.exit("engine.models.nnue.pack_board is needed for check (run from the project root).")
    exe = find_engine(a.engine)
    base = parse_params(run_engine(exe, ["evalparams"]))
    if a.params:
        base = {**base, **parse_params(Path(a.params).read_text())}
    mg, eg, tempo = to_vectors(base)
    boards = random_boards(a.n, a.seed)
    bb = np.array([pack_board(b) for b in boards], dtype=np.uint64)
    X, phase = features(bb)
    mine = predict(X, phase, mg, eg, tempo)

    cmds = []
    if a.params:
        cmds.append(f"setoption name EvalParams value {a.params}")
    for b in boards:
        cmds += [f"position fen {b.fen()}", "eval"]
    lines = [l for l in run_engine(exe, cmds).splitlines() if l.startswith("eval hand")]
    if len(lines) != len(boards):
        sys.exit(f"Engine answered {len(lines)} of {len(boards)} eval commands.")
    theirs = np.array([int(l.split()[2]) for l in lines], dtype=np.float64)
    diff = np.abs(mine - theirs)
    print(f"{len(boards)} random positions: mean |difference| {diff.mean():.3f} cp, max {diff.max():.2f} cp")
    ok = diff.max() <= 2.0
    if not ok:
        for i in np.argsort(-diff)[:3]:
            print(f"  {boards[i].fen()}  tuner {mine[i]:.1f}  engine {theirs[i]:.0f}")
    print("OK: features match the engine." if ok else "MISMATCH: do not tune until this is fixed.")
    sys.exit(0 if ok else 1)

# tune
def load_data(dirs, max_positions, seed):
    bbs, labs = [], []
    for d in dirs:
        p = Path(d)
        if (p / "bitboards.npy").exists():
            bbs.append(np.load(p / "bitboards.npy", mmap_mode="r"))
            labs.append(np.load(p / "labels.npy", mmap_mode="r"))
            print(f"  {d}: {len(labs[-1]):,} positions")
    if not bbs:
        sys.exit("No prepared data found. Pass --data folder(s) containing bitboards.npy and labels.npy.")
    total = sum(len(l) for l in labs)
    rng = np.random.default_rng(seed)
    take = min(total, max_positions)
    picks = np.sort(rng.choice(total, size=take, replace=False))
    offsets = np.cumsum([0] + [len(l) for l in labs])
    out_bb, out_y = [], []
    for i in range(len(bbs)):
        sel = picks[(picks >= offsets[i]) & (picks < offsets[i + 1])] - offsets[i]
        out_bb.append(np.asarray(bbs[i][sel]).view(np.uint64).reshape(-1, 12))
        out_y.append(np.asarray(labs[i][sel], dtype=np.float32))
    return np.concatenate(out_bb), np.concatenate(out_y)

def build_features(bb, chunk=40000):
    Xs, ps = [], []
    t0 = time.time()
    for i in range(0, len(bb), chunk):
        X, p = features(bb[i:i + chunk])
        Xs.append(X)
        ps.append(p)
        done = min(len(bb), i + chunk)
        print(f"\r  features {done:,}/{len(bb):,}  ({done / (time.time() - t0):,.0f}/s)", end="", flush=True)
    print()
    return np.concatenate(Xs), np.concatenate(ps)

def metrics(X, phase, y, mg, eg, tempo):
    pred = predict(X, phase, mg, eg, tempo)
    loss = float(np.mean((sigmoid(pred / SCALE) - sigmoid(y / SCALE)) ** 2))
    return loss, float(np.mean(np.abs(pred - y)))

def cmd_tune(a):
    exe = find_engine(a.engine)
    base = parse_params(run_engine(exe, ["evalparams"]))
    mg0, eg0, tempo0 = to_vectors(base)
    print("Loading data")
    bb, y = load_data(a.data, a.max_positions, a.seed)
    print(f"Computing features for {len(y):,} positions")
    X, phase = build_features(bb)
    del bb

    rng = np.random.default_rng(a.seed + 1)
    perm = rng.permutation(len(y))
    n_val = max(1000, int(a.val_frac * len(y)))
    va, tr = perm[:n_val], perm[n_val:]
    Xv, pv, yv = X[va], phase[va], y[va]

    trainable = np.ones(F, np.float32)
    if a.only_pst:
        trainable[N_PST:] = 0
        a.freeze_tempo = True
    if a.no_new_terms:
        trainable[NEW_TERMS] = 0
    if a.freeze_pst:
        trainable[:N_PST] = 0
    # material values only make sense relative to the pawn, and a king has no value: keep the king table fixed
    trainable[5 * 64:6 * 64] = 0 if a.freeze_king else trainable[5 * 64:6 * 64]

    mg, eg, tempo = mg0.copy(), eg0.copy(), tempo0
    l0, e0 = metrics(Xv, pv, yv, mg, eg, tempo)
    print(f"\nStart (current engine weights): validation loss {l0:.5f}, typical error {e0:.0f} cp")

    # Adam on (mg, eg, tempo); the model is linear in the weights so the gradient is exact and cheap
    params = [mg, eg, np.array([tempo])]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    steps_total = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    step, best = 0, (l0, mg.copy(), eg.copy(), tempo)
    t0 = time.time()
    for epoch in range(a.epochs):
        order = rng.permutation(tr)
        for i in range(0, len(order), a.batch):
            idx = np.sort(order[i:i + a.batch])
            Xb = X[idx].astype(np.float32)
            pm = phase[idx].astype(np.float32) / 24.0
            pred = (Xb @ params[0].astype(np.float32)) * pm + (Xb @ params[1].astype(np.float32)) * (1 - pm) + params[2][0]
            s = sigmoid(pred / SCALE)
            g = (2.0 * (s - sigmoid(y[idx] / SCALE)) * s * (1 - s) / SCALE / len(idx)).astype(np.float32)
            grads = [(Xb.T @ (g * pm)).astype(np.float64) * trainable, (Xb.T @ (g * (1 - pm))).astype(np.float64) * trainable,
                     np.array([g.sum()], np.float64) * (0 if a.freeze_tempo else 1)]
            step += 1
            lr = a.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * step / steps_total)))
            for k in range(3):
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                params[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-12)
            if a.max_change:                      # stay close to the current weights
                np.clip(params[0], mg0 - a.max_change, mg0 + a.max_change, out=params[0])
                np.clip(params[1], eg0 - a.max_change, eg0 + a.max_change, out=params[1])
                np.clip(params[2], tempo0 - a.max_change, tempo0 + a.max_change, out=params[2])
        lv, ev = metrics(Xv, pv, yv, params[0], params[1], params[2][0])
        if lv < best[0]:
            best = (lv, params[0].copy(), params[1].copy(), float(params[2][0]))
        print(f"  epoch {epoch + 1:>3}/{a.epochs}  validation loss {lv:.5f}  typical error {ev:.0f} cp"
              f"   [{time.time() - t0:.0f}s]", flush=True)

    lv, mg, eg, tempo = best
    # evaluate again with the rounded integer weights the engine will actually use
    mg, eg, tempo = np.rint(mg), np.rint(eg), float(round(tempo))
    lr_, er_ = metrics(Xv, pv, yv, mg, eg, tempo)
    print(f"\nStart  : loss {l0:.5f}, error {e0:.0f} cp\nTuned  : loss {lr_:.5f}, error {er_:.0f} cp (rounded weights)")
    if lr_ >= l0:
        print("The tuned weights are no better than the current ones on held-out positions; nothing written.")
        return
    write_params(base, mg, eg, tempo, a.out)
    print(f"Wrote {a.out}")
    print(f"Largest change: piece-square/material {max(abs(mg - mg0)[:N_PST].max(), abs(eg - eg0)[:N_PST].max()):.0f} cp, "
          f"other terms {max(abs(mg - mg0)[N_PST:].max(), abs(eg - eg0)[N_PST:].max()):.0f} cp")
    k = N_PST + N_PASSED
    print("\nTerms (middlegame / endgame), before -> after:")
    for i, name in enumerate(SCALAR_NAMES):
        print(f"  {name:<12} {mg0[k + i]:>5.0f} / {eg0[k + i]:<5.0f} -> {mg[k + i]:>5.0f} / {eg[k + i]:<5.0f}")
    k += N_SCALAR
    for t, name in enumerate("NBRQ"):
        print(f"  mobility {name}   {mg0[k + t]:>5.0f} / {eg0[k + t]:<5.0f} -> {mg[k + t]:>5.0f} / {eg[k + t]:<5.0f}")
    for t, name in enumerate("NBRQ"):
        print(f"  king zone {name}  {mg0[k + 4 + t]:>5.0f} / {eg0[k + 4 + t]:<5.0f} -> {mg[k + 4 + t]:>5.0f} / {eg[k + 4 + t]:<5.0f}")
    print(f"  tempo        {tempo0:.0f} -> {tempo:.0f}")
    print("\nTest it with match_engines.py using --a-opt EvalParams=<that file> before trusting it.")

BAKE_FILE = "engine/cpp/eval_default.inc"
REQUIRED = ["passed_mg", "passed_eg", "isolated", "doubled", "rook_open", "rook_semi", "bishop_pair", "shield",
            "tempo", "mob_mg", "mob_eg", "kz_mg", "kz_eg"]

def cmd_bake(a):
    text = Path(a.params).read_text()
    d = parse_params(text)
    missing = [k for k in REQUIRED if k not in d] + [k for k in (("pst_mg", p) for p in range(1, 7)) if k not in d] \
        + [k for k in (("pst_eg", p) for p in range(1, 7)) if k not in d]
    if missing:
        sys.exit(f"{a.params} is not a complete parameter file (missing {missing[:3]}...). Use a file written by `tune`.")
    out = Path(a.out)
    out.write_text("// Generated by `python tune_eval.py bake` from " + Path(a.params).name + ".\n"
                   "// Delete the text between the quotes to go back to the original hand-written weights.\n"
                   "static const char* const BAKED_EVAL_PARAMS = R\"EVALP(\n" + text.strip() + "\n)EVALP\";\n")
    print(f"Wrote {out}. Now rebuild:  cmake --build build --config Release")
    print("Afterwards check it:        python tune_eval.py verify --params " + a.params)
    print("The old weights stay available with the UCI option EvalParams=original.")

def cmd_verify(a):
    exe = find_engine(a.engine)
    eng = parse_params(run_engine(exe, ["evalparams"]))
    want = parse_params(Path(a.params).read_text())
    bad = [k for k in want if eng.get(k) != want[k]]
    if bad:
        print(f"Engine defaults differ from {a.params} in: {bad[:5]}. Did you rebuild after `bake`?")
        sys.exit(1)
    print(f"OK: the engine's built-in weights equal {a.params}.")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="verify the features reproduce the engine's evaluation")
    p.add_argument("--engine", default=None)
    p.add_argument("--params", default=None, help="parameter file to load in both the tuner and the engine")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("tune", help="fit the weights and write the parameter file")
    p.add_argument("--engine", default=None)
    p.add_argument("--data", nargs="+", default=DEFAULT_DATA, help="prepared folders (bitboards.npy + labels.npy)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--max-positions", type=int, default=1_000_000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1.0, help="Adam step in centipawns (decays to 5%%)")
    p.add_argument("--val-frac", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-new-terms", action="store_true", help="keep mobility and king-zone weights at 0")
    p.add_argument("--freeze-pst", action="store_true", help="only fit the non-table terms")
    p.add_argument("--only-pst", action="store_true", help="only fit the piece-square/material tables")
    p.add_argument("--max-change", type=float, default=0, metavar="CP",
                   help="keep every weight within +-CP centipawns of the current engine's (0 = no limit)")
    p.add_argument("--freeze-king", action="store_true", help="keep the king piece-square table as it is")
    p.add_argument("--freeze-tempo", action="store_true")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("bake", help="make a parameter file the engine's built-in default")
    p.add_argument("--params", required=True)
    p.add_argument("--out", default=BAKE_FILE)
    p.set_defaults(fn=cmd_bake)

    p = sub.add_parser("verify", help="check the engine's built-in weights equal a parameter file")
    p.add_argument("--engine", default=None)
    p.add_argument("--params", required=True)
    p.set_defaults(fn=cmd_verify)

    a = ap.parse_args()
    a.fn(a)

if __name__ == "__main__":
    main()