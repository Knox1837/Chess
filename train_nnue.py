"""
Train a new evaluation network for the chess AI. The game picks nnue.npz up automatically
"""
import argparse
import io
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.models.nnue import FEATURES, KING_BUCKETS, SCALE, pack_board  # noqa: E402

DEFAULT_DATA = Path("engine/data/processed/nnue")
DEFAULT_OUT = Path("engine/models/saved")
PIECE_VAL = [1, 3, 3, 5, 9, 0]

#  1) prepare: lichess eval dump -> packed bitboards + labels
def _iter_lines(path):
    path = str(path)
    if path.endswith(".zst"):
        import zstandard
        with open(path, "rb") as fh:
            text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                    encoding="utf-8", errors="replace")
            try:
                for line in text:
                    yield line
            except zstandard.ZstdError:
                print("(archive ended early - using what was read; fine for a partial download)")
    else:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                yield line

def cmd_prepare(a):
    import chess
    import numpy as np

    n_max = a.max_positions
    bbs = np.zeros((n_max, 12), dtype=np.uint64)
    cps = np.zeros(n_max, dtype=np.int16)
    white_to_move = np.zeros(n_max, dtype=bool)
    material = np.zeros(n_max, dtype=np.int16)      # side-to-move material minus opponent
    n = seen = 0
    t0 = time.time()
    pc = chess.popcount

    for line in _iter_lines(a.src):
        seen += 1
        if a.keep_prob < 1.0 and random.random() > a.keep_prob:
            continue
        try:
            rec = json.loads(line)
            ev = max(rec["evals"], key=lambda e: e.get("depth", 0))
            if ev.get("depth", 0) < a.min_depth:
                continue
            pv = ev["pvs"][0]
            if "cp" not in pv:                      # forced mates teach nothing useful here
                continue
            cp = int(pv["cp"])
            if abs(cp) > a.max_cp:
                continue
            board = chess.Board(rec["fen"])
            if board.is_check():
                continue
            best = chess.Move.from_uci(pv["line"].split()[0])
            if board.is_capture(best) or best.promotion:   # keep QUIET positions only;
                continue                                    # the search handles tactics
        except (ValueError, KeyError, IndexError, TypeError):
            continue

        planes = pack_board(board)
        bbs[n] = planes
        cps[n] = cp
        white_to_move[n] = board.turn
        material[n] = sum(PIECE_VAL[i] * (pc(planes[i]) - pc(planes[i + 6])) for i in range(6))
        n += 1
        if n % 200_000 == 0:
            print(f"  {n:,} positions kept from {seen:,} lines  ({n / (time.time() - t0):,.0f}/s)")
        if n >= n_max:
            break

    if n < 1000:
        sys.exit("Too few usable positions - is --src the lichess_db_eval file?")
    bbs, cps, wtm, mat = bbs[:n], cps[:n], white_to_move[:n], material[:n]

    # The dump's cp may be White-relative or side-to-move-relative. Decide from the data:
    # real labels correlate with material from the mover's point of view.
    raw = cps.astype(np.float32)
    as_white = np.where(wtm, raw, -raw)             # label if raw is White-relative
    c_stm = np.corrcoef(raw, mat)[0, 1]
    c_white = np.corrcoef(as_white, mat)[0, 1]
    if a.cp_perspective == "auto":
        mode = "stm" if c_stm >= c_white else "white"
    else:
        mode = a.cp_perspective
    print(f"material correlation  if cp is side-to-move-relative: {c_stm:.3f}   if White-relative: {c_white:.3f}"
          f"  -> using '{mode}'")
    labels = (raw if mode == "stm" else as_white).astype(np.int16)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "bitboards.npy", bbs)
    np.save(out / "labels.npy", labels)
    (out / "meta.json").write_text(json.dumps({"positions": int(n), "perspective": mode, "lines_read": seen}))
    print(f"Saved {n:,} positions to {out}  ({bbs.nbytes / 1e6:.0f} MB)")

#  2) train  (v2: king-bucketed features, both perspectives)
_PERM = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]     # swap own <-> opponent planes

def build_model(H):
    import torch
    import torch.nn as nn

    class NNUEv2(nn.Module):
        def __init__(s):
            super().__init__()
            s.ft = nn.EmbeddingBag(KING_BUCKETS * FEATURES, H, mode="sum")   # shared feature transformer
            s.ft_bias = nn.Parameter(torch.zeros(H))
            s.l2 = nn.Linear(2 * H, 32)
            s.l3 = nn.Linear(32, 32)
            s.out = nn.Linear(32, 1)
            nn.init.normal_(s.ft.weight, 0.0, 0.05)
            s.register_buffer("shifts", torch.arange(64, dtype=torch.int64), persistent=False)
            s.register_buffer("perm", torch.tensor(_PERM), persistent=False)

        def _bag(s, bits):                           # bits: (B, 12, 64) uint8, own planes first
            B = bits.shape[0]
            ksq = bits[:, 5].float().argmax(-1)      # own king square
            bucket = (ksq & 7) // 2 + 4 * ((ksq >> 3) >= 2).long()
            nz = bits.nonzero()                      # (N, 3) = (position, plane, square), sorted by position
            idx = bucket[nz[:, 0]] * FEATURES + nz[:, 1] * 64 + nz[:, 2]
            cnt = bits.reshape(B, -1).sum(1)
            off = torch.cumsum(cnt, 0) - cnt
            return s.ft(idx, off) + s.ft_bias

        def forward(s, x):                           # x: (B, 12) int64 side-to-move-relative bitboards
            B = x.shape[0]
            us = ((x.unsqueeze(-1) >> s.shifts) & 1).to(torch.uint8)
            them = us[:, s.perm].reshape(B, 12, 8, 8).flip(2).reshape(B, 12, 64)
            h = torch.cat([torch.clamp(s._bag(us), 0.0, 1.0), torch.clamp(s._bag(them), 0.0, 1.0)], 1)
            h = torch.clamp(s.l2(h), 0.0, 1.0)
            h = torch.clamp(s.l3(h), 0.0, 1.0)
            return s.out(h).squeeze(-1)              # logit; cp = logit * SCALE

    return NNUEv2()

def cmd_train(a):
    import numpy as np
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    bbs, labs = [], []
    for d in a.data:
        bbs.append(np.load(Path(d) / "bitboards.npy").view(np.int64))
        labs.append(np.load(Path(d) / "labels.npy").astype(np.float32))
    bb = torch.from_numpy(np.concatenate(bbs))
    lab = torch.from_numpy(np.concatenate(labs))
    N = len(lab)
    target = torch.sigmoid(lab / SCALE)             # win-probability-like target in (0, 1)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    n_val = max(1000, int(0.03 * N))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    on_dev = N * 100 < 3e9                          # keep everything on the GPU if it fits
    if on_dev:
        bb, target = bb.to(dev), target.to(dev)
        val_idx, train_idx = val_idx.to(dev), train_idx.to(dev)

    def batch(idx):
        x, y = bb[idx], target[idx]
        if not on_dev:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        return x, y

    H = a.hidden
    model = build_model(H).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_idx) / a.batch)
    total = steps_per_epoch * a.epochs
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    def val_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(val_idx), 32768):
                x, y = batch(val_idx[i:i + 32768])
                tot += ((torch.sigmoid(model(x)) - y) ** 2).sum().item()
        model.train()
        return tot / len(val_idx)

    base = float(target[val_idx].float().var().item())
    print(f"{N:,} positions  ({len(train_idx):,} train / {len(val_idx):,} val)   hidden={H}  (v2: king buckets x 2 perspectives)")
    print(f"Baseline val loss (always guess the mean): {base:.5f}  - the model must get far below this\n"
          f"(your first network reached 0.00735 on a similar sample)\n")

    best, out = float("inf"), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        p = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]
        run = 0.0
        for i in range(0, len(p), a.batch):
            x, y = batch(p[i:i + a.batch])
            loss = ((torch.sigmoid(model(x)) - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(y)
        v = val_loss()
        mark = ""
        if v < best:
            best = v
            _export(model, out)
            mark = "  <- saved"
        print(f"epoch {epoch:3d}/{a.epochs}  train {run / len(p):.5f}  val {v:.5f}  "
              f"(explains {100 * (1 - v / base):.1f}% of variance)  {time.time() - t0:.0f}s{mark}")
    print(f"\nBest val loss {best:.5f}. Network written to {out / 'nnue.npz'}")

def _export(model, out):
    import numpy as np
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(out / "nnue.npz", version=np.array(2),
             W1=sd["ft.weight"], b1=sd["ft_bias"],
             W2=sd["l2.weight"].T, b2=sd["l2.bias"],
             W3=sd["l3.weight"].T, b3=sd["l3.bias"],
             W4=sd["out.weight"].T, b4=sd["out.bias"])

#  3) match: new network vs hand-written eval, same search, same time
def cmd_match(a):
    import chess
    from engine.ai.chess_ai import ChessAI

    nn_ai = ChessAI(skill_level=4, use_nnue=True, time_limit=a.movetime * a.nn_time_factor, nnue_blend=a.blend)
    hc_ai = ChessAI(skill_level=4, use_nnue=False, time_limit=a.movetime)
    if nn_ai.nnue is None:
        sys.exit("No nnue.npz found - train first.")
    if a.nn_time_factor != 1.0:
        print(f"NNUE gets {a.nn_time_factor:g}x the thinking time ({a.movetime * a.nn_time_factor:g}s vs {a.movetime:g}s)")
    w = d = l = 0
    stats = {"nn": [0, 0], "hc": [0, 0]}            # [sum of search depths, moves]
    ends = {"checkmate": 0, "draw by rule": 0, "ply limit": 0}
    lengths = {"win": [], "loss": []}
    for g in range(a.games):
        rng = random.Random(g // 2)                 # each opening is played twice, colours swapped
        board = chess.Board()
        for _ in range(6):
            board.push(rng.choice(list(board.legal_moves)))
        nn_white = g % 2 == 0
        nn_ai.new_game()
        hc_ai.new_game()
        while not board.is_game_over(claim_draw=True) and board.ply() < 240:
            use_nn = (board.turn == chess.WHITE) == nn_white
            ai = nn_ai if use_nn else hc_ai
            r = ai.choose_move(board)
            if ai.last_info.get("depth"):
                st = stats["nn" if use_nn else "hc"]
                st[0] += ai.last_info["depth"]
                st[1] += 1
            board.push(r["move"])
        o = board.outcome(claim_draw=True)
        if o is None:
            ends["ply limit"] += 1
        elif o.winner is None:
            ends["draw by rule"] += 1
        else:
            ends["checkmate"] += 1
        if o is None or o.winner is None:
            d += 1
            res = "draw"
        elif o.winner == nn_white:
            w += 1
            res = "NNUE wins"
            lengths["win"].append(board.ply())
        else:
            l += 1
            res = "hand-eval wins"
            lengths["loss"].append(board.ply())
        print(f"game {g + 1}/{a.games}: {res} in {board.ply()} plies   (NNUE +{w} ={d} -{l})", flush=True)
    score = (w + 0.5 * d) / max(1, a.games)
    margin = 100 * math.sqrt(0.25 / max(1, a.games))
    print(f"\nNNUE score: {100 * score:.0f}% +/- {margin:.0f}  ({w}W {d}D {l}L)")
    avg = lambda v: sum(v) / len(v) if v else 0
    print(f"Average search depth:  NNUE {stats['nn'][0] / max(1, stats['nn'][1]):.1f}   hand-written {stats['hc'][0] / max(1, stats['hc'][1]):.1f}")
    print(f"Average game length (plies):  NNUE wins {avg(lengths['win']):.0f}   NNUE losses {avg(lengths['loss']):.0f}")
    print(f"Game endings: {ends}")
    if score > 0.5:
        elo = -400 * math.log10(1 / max(0.01, min(0.99, score)) - 1)
        print(f"Roughly {elo:+.0f} Elo vs the hand-written eval (noisy with few games).")

def cmd_compare(a):
    """Score hand-written eval vs network vs blends on the SAME held-out positions (no games needed)."""
    import chess
    import numpy as np
    import torch
    from engine.ai.chess_ai import evaluate as hand_eval
    from engine.models.nnue import find_nnue, load_nnue

    path = find_nnue(a.net)
    if not path:
        sys.exit("No nnue.npz found - train first.")
    net = load_nnue(path)

    bbs, labs = [], []
    for d in a.data:# same order + seed as `train` => same validation split
        bbs.append(np.load(Path(d) / "bitboards.npy"))
        labs.append(np.load(Path(d) / "labels.npy").astype(np.float32))
    bb, lab = np.concatenate(bbs), np.concatenate(labs)
    N = len(lab)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    val = perm[:max(1000, int(0.03 * N))][:a.n].numpy()

    hand, nnv, truth = [], [], []
    for i in val:
        board = chess.Board.empty()                  # side-to-move-relative frame: mover = White
        for plane in range(12):
            color = chess.WHITE if plane < 6 else chess.BLACK
            for sq in chess.scan_forward(int(bb[i, plane])):
                board.set_piece_at(sq, chess.Piece(plane % 6 + 1, color))
        hand.append(hand_eval(board))
        nnv.append(net.evaluate(board))
        truth.append(lab[i])
    hand, nnv, truth = (np.array(v, dtype=np.float64) for v in (hand, nnv, truth))

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x / SCALE))

    y = sig(truth)
    print(f"{len(y):,} held-out positions   (lower loss = closer to Stockfish)\n")
    print(f"  always guess the mean      {y.var():.5f}")
    for name, pred in (("hand-written eval", hand), ("network", nnv),
                       ("blend 70% net / 30% hand", 0.7 * nnv + 0.3 * hand),
                       ("blend 50% / 50%", 0.5 * nnv + 0.5 * hand)):
        print(f"  {name:<26} {np.mean((sig(pred) - y) ** 2):.5f}   "
              f"(typical error {np.mean(np.abs(pred - truth)):.0f} cp)")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="build training arrays from lichess_db_eval.jsonl.zst")
    p.add_argument("--src", required=True)
    p.add_argument("--out", default=str(DEFAULT_DATA))
    p.add_argument("--max-positions", type=int, default=5_000_000)
    p.add_argument("--min-depth", type=int, default=12)
    p.add_argument("--max-cp", type=int, default=2000)
    p.add_argument("--keep-prob", type=float, default=1.0,
                   help="randomly keep only this fraction of lines (use <1 to sample across a big file)")
    p.add_argument("--cp-perspective", choices=["auto", "white", "stm"], default="auto")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="train the network and export nnue.npz")
    p.add_argument("--data", nargs="+", default=[str(DEFAULT_DATA)],
                   help="one or more prepared folders (they are combined)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("match", help="play the new network against the hand-written eval")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--movetime", type=float, default=0.2)
    p.add_argument("--nn-time-factor", type=float, default=1.0,
                   help="give the network this many times the thinking time (2.0 tests 'would it win if it were as fast?')")
    p.add_argument("--blend", type=float, default=1.0,
                   help="1.0 = network only; 0.7 = 70%% network + 30%% hand-written eval")
    p.set_defaults(fn=cmd_match)

    p = sub.add_parser("compare", help="accuracy of hand-written eval vs network on held-out positions")
    p.add_argument("--data", nargs="+", default=[str(DEFAULT_DATA)])
    p.add_argument("--net", default=None, help="path to nnue.npz (default: engine/models/saved/nnue.npz)")
    p.add_argument("--n", type=int, default=30000)
    p.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    a.fn(a)

if __name__ == "__main__":
    main()