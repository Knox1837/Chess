"""Compare engine speed (nodes per second) with the hand-written eval and with the NNUE net.

    python bench_nps.py [--exe engine\\bin\\chess_engine.exe] [--net engine\\models\\saved\\nnue.bin]
"""
import argparse
import subprocess
import sys


def bench(exe, setoptions):
    cmds = "".join(f"setoption name {n} value {v}\n" for n, v in setoptions) + "bench\nquit\n"
    out = subprocess.run([exe], input=cmds, capture_output=True, text=True, timeout=600).stdout
    for line in out.splitlines():
        if line.startswith("bench nodes"):
            return line, out
    return None, out


ap = argparse.ArgumentParser()
ap.add_argument("--exe", default=r"engine\bin\chess_engine.exe")
ap.add_argument("--net", default=r"engine\models\saved\nnue.bin")
a = ap.parse_args()

runs = [("hand eval", [("UseNNUE", "false")]),
        ("NNUE", [("UseNNUE", "true"), ("EvalFile", a.net)])]
for name, opts in runs:
    line, out = bench(a.exe, opts)
    if "could not load net" in out:
        print(f"{name:10s} could not load the net, check --net (it would silently fall back to the hand eval)")
        continue
    print(f"{name:10s} {line or 'no bench result'}")