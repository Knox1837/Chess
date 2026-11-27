"""
Play two UCI engine builds (or two option sets of the same build) against each other.
Much more precise than Stockfish for "is this change better?" tests.

python match_engines.py --a engine/bin/chess_engine.exe --b engine/bin/chess_engine_old.exe
python match_engines.py --a engine/bin/chess_engine.exe --b engine/bin/chess_engine.exe --a-opt SoftTime=100 --b-opt SoftTime=20
python match_engines.py --a new.exe --b old.exe --games 400 --movetime 0.1 --concurrency 4
python match_engines.py --a e.exe --b e.exe --a-opt TimeManage=1 --b-opt TimeManage=0 --tc 10+0.1 --games 200 --concurrency 4

Games are played in pairs: each opening (a few random plies, same seed) is played twice with colours swapped.
--games is the total number of games (rounded up to an even number). Result is from A's point of view.
Run one game per free CPU core at most (--concurrency), otherwise timing becomes unfair.
"""
import argparse
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine

TIME_MARGIN = 0.0      # a move that takes the clock to 0 or below loses on time

RESIGN_CP = 1000       # a side resigns after RESIGN_MOVES of its own moves at or below -RESIGN_CP
RESIGN_MOVES = 4
MATE_CP = 10000

def parse_opts(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"Bad option '{item}', use Name=Value")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def open_engine(exe, opts, hash_mb, label):
    eng = chess.engine.SimpleEngine.popen_uci(exe)
    cfg = {}
    if "Hash" in eng.options:
        cfg["Hash"] = hash_mb
    if "UseNNUE" in eng.options:
        cfg["UseNNUE"] = False                     # hand-written eval unless overridden
    for name, value in opts.items():
        if name not in eng.options:
            known = ", ".join(eng.options)
            eng.quit()
            sys.exit(f"Engine {label} ({exe}) has no UCI option '{name}'. "
                     f"Is it the right build? Options: {known}")
        opt = eng.options[name]
        if opt.type == "spin":
            cfg[name] = int(value)
        elif opt.type == "check":
            cfg[name] = value.lower() in ("1", "true", "yes", "on")
        else:
            cfg[name] = value
    eng.configure(cfg)
    return eng

def cp_of(info, color):
    score = info.get("score")
    if score is None:
        return None
    return score.pov(color).score(mate_score=MATE_CP)

def stats(used, a_color):
    b_color = not a_color
    return [used[a_color][0], used[a_color][1], used[b_color][0], used[b_color][1]]

def play_game(args, pair_index, a_white):
    """Returns (result from A's view: 1 / 0.5 / 0, plies, reason, [time used by A, moves of A, time used by B, moves of B])"""
    rng = random.Random(args.seed + pair_index)
    board = chess.Board()
    for _ in range(args.opening_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))

    eng_a = open_engine(args.a, parse_opts(args.a_opt), args.hash, "A")
    eng_b = None
    try:
        eng_b = open_engine(args.b, parse_opts(args.b_opt), args.hash, "B")
        limit = chess.engine.Limit(depth=args.depth) if args.depth else chess.engine.Limit(time=args.movetime)
        base, inc = args.tc_base, args.tc_inc            # game clock instead of a fixed time per move (--tc)
        clock = {chess.WHITE: base, chess.BLACK: base}
        used = {chess.WHITE: [0.0, 0], chess.BLACK: [0.0, 0]}
        game_token = object()
        bad_streak = {chess.WHITE: 0, chess.BLACK: 0}
        a_color = chess.WHITE if a_white else chess.BLACK
        reason = None
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                break
            if board.ply() >= args.max_plies:
                return 0.5, board.ply(), "max plies", stats(used, a_color)
            side = board.turn
            eng = eng_a if side == a_color else eng_b
            if base is not None:
                limit = chess.engine.Limit(white_clock=clock[chess.WHITE], black_clock=clock[chess.BLACK],
                                           white_inc=inc, black_inc=inc)
            t0 = time.perf_counter()
            res = eng.play(board, limit, info=chess.engine.INFO_SCORE, game=game_token)
            elapsed = time.perf_counter() - t0
            used[side][0] += elapsed
            used[side][1] += 1
            if base is not None:
                clock[side] -= elapsed
                if clock[side] <= TIME_MARGIN:
                    return (0.0 if side == a_color else 1.0), board.ply(), "time", stats(used, a_color)
                clock[side] += inc
            if res.move is None:
                break
            cp = cp_of(res.info, side)
            if not args.no_resign and cp is not None:
                bad_streak[side] = bad_streak[side] + 1 if cp <= -RESIGN_CP else 0
                if bad_streak[side] >= RESIGN_MOVES:
                    return (0.0 if side == a_color else 1.0), board.ply(), "resign", stats(used, a_color)
            board.push(res.move)
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.5, board.ply(), "draw", stats(used, a_color)
        return (1.0 if outcome.winner == a_color else 0.0), board.ply(), "checkmate", stats(used, a_color)
    finally:
        eng_a.quit()
        if eng_b is not None:
            eng_b.quit()

def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def elo(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / p - 1)

def summary(w, d, l):
    n = w + d + l
    if n == 0:
        return "no games finished"
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    se = math.sqrt(var / n)
    lo, hi = elo(s - 1.96 * se), elo(s + 1.96 * se)
    los = phi((w - l) / math.sqrt(w + l)) if (w + l) > 0 else 0.5
    return (f"A vs B: +{w} ={d} -{l}  ({n} games)  score {s * 100:.1f}%\n"
            f"Elo difference (A - B): {elo(s):+.0f}   95% interval [{lo:+.0f}, {hi:+.0f}]\n"
            f"Likelihood A is stronger: {los * 100:.0f}%")

def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="engine A (the new one) executable")
    ap.add_argument("--b", required=True, help="engine B (the baseline) executable")
    ap.add_argument("--a-opt", action="append", metavar="Name=Value", help="UCI option for A (repeatable)")
    ap.add_argument("--b-opt", action="append", metavar="Name=Value", help="UCI option for B (repeatable)")
    ap.add_argument("--games", type=int, default=100, help="total games (default 100)")
    ap.add_argument("--movetime", type=float, default=0.1, help="seconds per move for both engines")
    ap.add_argument("--depth", type=int, default=0, help="fixed search depth per move for both engines (ignores time; compares evaluation quality, not speed)")
    ap.add_argument("--tc", default=None, metavar="BASE+INC", help="game clock in seconds for each side, e.g. 10+0.1 (replaces --movetime; a side that runs out of time loses)")
    ap.add_argument("--concurrency", type=int, default=1, help="games at the same time (<= free CPU cores)")
    ap.add_argument("--hash", type=int, default=64, help="Hash MB for each engine")
    ap.add_argument("--opening-plies", type=int, default=4, help="random opening plies per pair")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-resign", action="store_true", help="play every game to the end (slower)")
    args = ap.parse_args()

    args.tc_base = args.tc_inc = None
    if args.depth:
        args.tc = None                                   # a fixed depth replaces any clock
    if args.tc:
        try:
            base, inc = args.tc.split("+")
            args.tc_base, args.tc_inc = float(base), float(inc)
        except ValueError:
            sys.exit("--tc must look like 10+0.1 (base seconds + increment seconds)")
    for p in (args.a, args.b):
        if not Path(p).is_file():
            sys.exit(f"Engine not found: {p}")

    # Fail early (in the main thread) if an engine does not start or lacks a requested option
    for exe, opts, label in ((args.a, args.a_opt, "A"), (args.b, args.b_opt, "B")):
        open_engine(exe, parse_opts(opts), args.hash, label).quit()

    pairs = (args.games + 1) // 2
    jobs = [(i, a_white) for i in range(pairs) for a_white in (True, False)]
    w = d = l = flags = 0
    tot = [0.0, 0, 0.0, 0]
    start = time.time()
    done = 0
    ex = ThreadPoolExecutor(max_workers=max(1, args.concurrency))
    futures = {ex.submit(play_game, args, i, aw): (i, aw) for i, aw in jobs}
    try:
        for fut in as_completed(futures):
            i, aw = futures[fut]
            score, plies, reason, st = fut.result()
            for k in range(4):
                tot[k] += st[k]
            flags += reason == "time"
            done += 1
            if score == 1:
                w += 1
            elif score == 0:
                l += 1
            else:
                d += 1
            s = (w + 0.5 * d) / done
            print(f"game {done:>4}/{len(jobs)}  pair {i + 1:>3} A={'white' if aw else 'black'}  "
                  f"{'A wins' if score == 1 else 'B wins' if score == 0 else 'draw  '}  "
                  f"({reason}, {plies} plies)   running +{w} ={d} -{l}  {s * 100:.1f}%", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted, showing the games finished so far.")
        ex.shutdown(wait=False, cancel_futures=True)
    else:
        ex.shutdown()

    print("\n" + summary(w, d, l))
    print(f"Total time: {fmt_time(time.time() - start)}")
    if tot[1] and tot[3]:
        print(f"Average time per move: A {1000 * tot[0] / tot[1]:.0f} ms, B {1000 * tot[2] / tot[3]:.0f} ms"
              + (f"   |   games lost on time: {flags}" if args.tc else ""))
    print(f"Settings: {('depth ' + str(args.depth)) if args.depth else (('clock ' + args.tc) if args.tc else str(args.movetime) + 's/move')}, hash {args.hash} MB, A options {parse_opts(args.a_opt)}, "
          f"B options {parse_opts(args.b_opt)}")

if __name__ == "__main__":
    main()