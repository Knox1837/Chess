"""
Measure the chess AI against Stockfish at limited strength (the same ChessAI class the game uses).

python evaluate_engine.py                                   # level 4, Elo 1800 / 2200 / 2600, 10 games each
python evaluate_engine.py --movetime 0.1 --games 20         # quicker moves, more games
python evaluate_engine.py --level 5 --elos 2400 2800
python evaluate_engine.py --nnue                            # test the neural network instead of the hand-written eval
python evaluate_engine.py --concurrency 4                   # play 4 games at once (about 4x faster, use <= free CPU cores)
"""
import argparse
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess
import chess.engine

from engine.ai.chess_ai import ChessAI

RESIGN_CP = 1000       # Stockfish's own score: game is decided after RESIGN_MOVES of its moves at/beyond +-RESIGN_CP
RESIGN_MOVES = 4
MATE_CP = 10000

def find_stockfish(arg):
    cands = ([arg] if arg else []) + ["engine/stockfish.exe", "engine/stockfish", "stockfish.exe", "stockfish"]
    for c in cands:
        if Path(c).is_file():
            return str(Path(c))
        found = shutil.which(c)
        if found:
            return found
    sys.exit("Stockfish not found. Download it from stockfishchess.org and pass --stockfish PATH")

def play_game(ai, sf, sf_time, ai_white, opening_seed, max_plies, resign=True):
    rng = random.Random(opening_seed)
    board = chess.Board()
    for _ in range(4):
        board.push(rng.choice(list(board.legal_moves)))
    ai.new_game()
    streak = 0                                   # >0: Stockfish clearly winning, <0: clearly losing
    sf_color = chess.BLACK if ai_white else chess.WHITE
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == ai_white:
            board.push(ai.choose_move(board)["move"])
        else:
            res = sf.play(board, chess.engine.Limit(time=sf_time), info=chess.engine.INFO_SCORE)
            board.push(res.move)
            if resign and res.info.get("score") is not None:
                cp = res.info["score"].pov(sf_color).score(mate_score=MATE_CP)
                if cp >= RESIGN_CP:
                    streak = streak + 1 if streak >= 0 else 1
                elif cp <= -RESIGN_CP:
                    streak = streak - 1 if streak <= 0 else -1
                else:
                    streak = 0
                if streak >= RESIGN_MOVES:
                    return "loss", board.ply()
                if streak <= -RESIGN_MOVES:
                    return "win", board.ply()
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw", board.ply()
    return ("win" if outcome.winner == ai_white else "loss"), board.ply()

def run_game(a, sf_path, elo, g):
    """One game with its own AI and Stockfish process (so several can run at once)"""
    ai = ChessAI(skill_level=a.level, use_nnue=a.nnue, time_limit=a.movetime, ponder=a.ponder)
    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        opt = sf.options["UCI_Elo"]
        cfg = {"UCI_LimitStrength": True, "UCI_Elo": max(opt.min, min(opt.max, elo)), "Threads": 1}
        if "Hash" in sf.options:
            cfg["Hash"] = 16
        sf.configure(cfg)
        return play_game(ai, sf, a.sf_time, g % 2 == 0, g // 2, a.max_plies, not a.no_resign)
    finally:
        sf.quit()
        ai.close()

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
    ap.add_argument("--ponder", action="store_true", help="let the AI think on Stockfish's time (needs 2+ CPU cores to be fair; raises the score)")
    ap.add_argument("--elos", type=int, nargs="+", default=[1800, 2200, 2600], help="Stockfish Elo levels")
    ap.add_argument("--games", type=int, default=10, help="games per Elo level (even number is best)")
    ap.add_argument("--sf-time", type=float, default=0.1, help="seconds per Stockfish move")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--stockfish", default=None)
    ap.add_argument("--concurrency", type=int, default=1, help="games played at the same time (<= free CPU cores / 2; each game uses 2 engines)")
    ap.add_argument("--no-resign", action="store_true", help="play every game to the end instead of stopping decided games early (slower)")
    a = ap.parse_args()

    sf_path = find_stockfish(a.stockfish)
    probe = chess.engine.SimpleEngine.popen_uci(sf_path)
    opt = probe.options["UCI_Elo"]
    probe.quit()
    elos = [max(opt.min, min(opt.max, e)) for e in a.elos]
    cores = os.cpu_count() or 1
    if a.concurrency * 2 > cores:
        print(f"Note: {a.concurrency} games at once = {a.concurrency * 2} busy engines on {cores} logical cores; "
              f"timing may be unfair. Lower --concurrency if results look odd.")
    if a.concurrency > 1 and a.ponder:
        print("Note: --ponder with --concurrency > 1 makes the AI compete for CPU with the other games.")

    estimates = []
    start = time.time()
    totals = {elo: [0, 0, 0, 0] for elo in elos}      # wins, draws, losses, finished
    ex = ThreadPoolExecutor(max_workers=max(1, a.concurrency))
    futures = {ex.submit(run_game, a, sf_path, elo, g): (elo, g) for elo in elos for g in range(a.games)}
    try:
        for fut in as_completed(futures):
            elo, g = futures[fut]
            res, plies = fut.result()
            t = totals[elo]
            t[0] += res == "win"; t[1] += res == "draw"; t[2] += res == "loss"; t[3] += 1
            w, d, l, done = t
            print(f"  vs Stockfish {elo}: game {done}/{a.games} {res} in {plies} plies   (+{w} ={d} -{l})   [{fmt_time(time.time() - start)}]", flush=True)
            if done == a.games:
                diff = perf_diff(w, d, l)
                pts = w + 0.5 * d
                word = "at least" if pts == a.games else ("at most" if pts == 0 else "about")   # 100% / 0% = only a bound
                print(f"Stockfish {elo}: {w}W {d}D {l}L  score {100 * pts / a.games:.0f}%  "
                      f"-> performance {word} {elo + diff:.0f}\n", flush=True)
                if 0 < pts < a.games:               # a 0% or 100% score only gives a bound, not an estimate
                    estimates.append(elo + diff)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    print(f"Total evaluation time: {fmt_time(time.time() - start)}")
    if estimates:
        print(f"Overall estimate (average of the informative levels): about {sum(estimates) / len(estimates):.0f}")
    else:
        print("Every level was won or lost completely - pick closer Elo levels (--elos) for an estimate.")


if __name__ == "__main__":
    main()