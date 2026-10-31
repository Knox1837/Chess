"""
Measure the chess AI against Stockfish at limited strength (the same ChessAI class the game uses).

    python evaluate_engine.py                                   # level 4, Elo 1800 / 2200 / 2600, 10 games each
    python evaluate_engine.py --movetime 0.1 --games 20         # quicker moves, more games
    python evaluate_engine.py --level 5 --elos 2400 2800
    python evaluate_engine.py --nnue                            # test the neural network instead of the hand-written eval

Needs Stockfish (stockfishchess.org): engine\\stockfish.exe, or pass --stockfish PATH.
Games alternate colours from random 4-ply openings. Games run one after another (no parallel
workers), so the time-limited search is not starved of CPU.

Read the Elo numbers as a way to COMPARE versions of your engine, not as an official rating:
Stockfish's Elo limiter is calibrated for long time controls and plays weaker when it only gets
a fraction of a second per move.
"""
import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess
import chess.engine

from engine.ai.chess_ai import ChessAI

def find_stockfish(arg):
    cands = ([arg] if arg else []) + ["engine/stockfish.exe", "engine/stockfish", "stockfish.exe", "stockfish"]
    for c in cands:
        if Path(c).is_file():
            return str(Path(c))
        found = shutil.which(c)
        if found:
            return found
    sys.exit("Stockfish not found. Download it from stockfishchess.org and pass --stockfish PATH")

def play_game(ai, sf, sf_time, ai_white, opening_seed, max_plies):
    rng = random.Random(opening_seed)
    board = chess.Board()
    for _ in range(4):
        board.push(rng.choice(list(board.legal_moves)))
    ai.new_game()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == ai_white:
            board.push(ai.choose_move(board)["move"])
        else:
            board.push(sf.play(board, chess.engine.Limit(time=sf_time)).move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw", board.ply()
    return ("win" if outcome.winner == ai_white else "loss"), board.ply()

def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

def perf_diff(w, d, l):
    n = w + d + l
    p = (w + 0.5 * d) / n
    p = min(max(p, 0.5 / n), 1 - 0.5 / n)          # avoid +/- infinity on 100% / 0% scores
    return -400 * math.log10(1 / p - 1)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4, 5], help="AI skill level (default 4)")
    ap.add_argument("--movetime", type=float, default=None, help="seconds per AI move for levels 4-5")
    ap.add_argument("--nnue", action="store_true", help="use the neural network (needs nnue.npz)")
    ap.add_argument("--elos", type=int, nargs="+", default=[1800, 2200, 2600], help="Stockfish Elo levels")
    ap.add_argument("--games", type=int, default=10, help="games per Elo level (even number is best)")
    ap.add_argument("--sf-time", type=float, default=0.1, help="seconds per Stockfish move")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--stockfish", default=None)
    a = ap.parse_args()

    ai = ChessAI(skill_level=a.level, use_nnue=a.nnue, time_limit=a.movetime)
    sf = chess.engine.SimpleEngine.popen_uci(find_stockfish(a.stockfish))
    opt = sf.options["UCI_Elo"]
    estimates = []
    start = time.time()
    try:
        for elo in a.elos:
            elo = max(opt.min, min(opt.max, elo))
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1})
            w = d = l = 0
            t0 = time.time()
            for g in range(a.games):
                res, plies = play_game(ai, sf, a.sf_time, g % 2 == 0, g // 2, a.max_plies)
                w, d, l = w + (res == "win"), d + (res == "draw"), l + (res == "loss")
                print(f"  vs Stockfish {elo}: game {g + 1}/{a.games} {res} in {plies} plies   (+{w} ={d} -{l})   [{fmt_time(time.time() - start)}]", flush=True)
            n = w + d + l
            diff = perf_diff(w, d, l)
            pts = w + 0.5 * d
            word = "at least" if pts == n else ("at most" if pts == 0 else "about")   # 100% / 0% = only a bound
            print(f"Stockfish {elo}: {w}W {d}D {l}L  score {100 * pts / n:.0f}%  "
                  f"-> performance {word} {elo + diff:.0f}   ({fmt_time(time.time() - t0)})\n", flush=True)
            if 0 < w + 0.5 * d < n:                 # a 0% or 100% score only gives a bound, not an estimate
                estimates.append(elo + diff)
    finally:
        sf.quit()
        ai.close()
    print(f"Total evaluation time: {fmt_time(time.time() - start)}")
    if estimates:
        print(f"Overall estimate (average of the informative levels): about {sum(estimates) / len(estimates):.0f}")
    else:
        print("Every level was won or lost completely - pick closer Elo levels (--elos) for an estimate.")

if __name__ == "__main__":
    main()