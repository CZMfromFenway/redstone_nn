#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
奇数 b 的构造流程：先造 b+1（偶数 b），再砍。

  1. 用 b=9 的权重填进 b=10 板的槽位 1..9（槽位 0 空：S=0、火把常亮）
  2. 跑 b=10 板 → head 应 = N
  3. 砍掉 z=0,1,2   → L=35
  4. 再砍掉 z=0     → L=34
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
from cmp_sim import Board, DIRV  # noqa: E402
from plane_gen import build, geom  # noqa: E402
from draw_diag import diagnose, render  # noqa: E402

PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]


def cut_rows(g, k):
    """删掉最上面 k 行，其余上移。"""
    g2 = Board(g.W, g.L - k)
    for (x, z), v in g.solid.items():
        if z >= k:
            g2.solid[(x, z - k)] = v
    for (x, z), v in g.barrel.items():
        if z >= k:
            g2.barrel[(x, z - k)] = v
    for (x, z), v in g.torch.items():
        if z >= k:
            g2.torch[(x, z - k)] = v
    for (x, z), c in g.cmp.items():
        if z >= k:
            g2.set_cmp((x, z - k), c["out"], c["mode"], c["init"])
    return g2


def run_case(n, b9, src, J=0, tag="", cuts=()):
    """b9 = 目标位宽（奇数）。先造 b9+1 的偶数板，再依次砍 cuts。"""
    b10 = b9 + 1
    al = WREF / (np.abs(src).max(1) * DELTA / 4)
    U = np.rint(al[:, None, None] * src[:, :, None] * (DELTA / (1 << b9))
                * (1 << np.arange(b9))[None, None, :]).astype(np.int64)
    S9 = (15 + np.abs(U[J])).astype(int)
    wsg9 = (U[J] > 0).astype(int)
    rng = np.random.default_rng(11)
    inp9 = rng.integers(0, 2, size=(n, b9))

    # 填进 b10 的槽位 1..b9
    S = np.zeros((n, b10), dtype=int); S[:, 1:] = S9
    wsg = np.ones((n, b10), dtype=int); wsg[:, 1:] = wsg9
    inp = np.zeros((n, b10), dtype=int); inp[:, 1:] = inp9
    msb = np.abs(np.concatenate([[0], np.abs(U[J, :, b9 - 1])])).tolist()
    msb[1:] = np.abs(U[J, :, b9 - 1]).tolist()
    G = geom(n, b10)
    Jc = (G["zmsb"] - 2) // 6 + 1
    K = int(S.sum())
    W0 = Jc * K - 15 * n * b9
    g = build(n, b10, S, K, W0, inp, msb, wsg)
    g.run()
    port = (G["xw"], 4)
    h0 = int(g.cmp.get(port, {}).get("state", -1))
    Nt = int((U[J] * inp9).sum() - np.minimum(0, U[J]).sum())

    tc = {}
    for k in range(b10):
        zc = 3*k+2 if k % 2 == 0 else 3*k+3
        col = [2*i+3 for i in range(n)] if k % 2 == 0 else [2*i+4 for i in range(n)]
        for x in col:
            tc[(x, zc)] = (k-1) % 8 if k >= 1 else 7
    for i in range(n):
        tc[(4+2*i, G["zmsb"])] = 7

    out = {}
    zoff = 0
    for k in cuts:
        g = cut_rows(g, k)
        zoff += k
    L = g.L
    port2 = (G["xw"], 4 - zoff)
    d, o, st = diagnose(g, port2)
    g.run()
    h = int(g.cmp.get(port2, {}).get("state", -1))
    tc2 = {(x, z - zoff): v for (x, z), v in tc.items() if z >= zoff}
    fn = os.path.join(RESULTS, f"odd_b_{tag}.png")
    render(g, fn,
           [f"{tag}", f"b={b9}（由 b={b10} 砍成）", f"n={n}  L={L}",
            f"链数 J={Jc}", f"砍掉 {zoff} 行", "",
            f"head = {h}", f"N    = {Nt}",
            ">> OK <<" if h == Nt else ">> FAIL <<", "",
            f"死比较器 {len(d)}", f"孤儿方块 {len(o)}", f"饿环 {len(st)}"],
           port2, d, o, st, tc2, cs=30)
    print(f"  {tag:16s} b={b9} n={n} L={L} 砍{zoff}: head={h} N={Nt} "
          f"{'OK' if h == Nt else 'FAIL'}  死{len(d)} 孤{len(o)} 饿{len(st)}")
    return h, Nt, len(d), len(o)


if __name__ == "__main__":
    P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
    mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                        encoding="utf-8"))
    W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
    V2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
    DELTA, WREF = 7.0, 882.0

    run_case(4, 9, W1[:, :4], 0, "n4_b9_cut3", cuts=(3,))
    run_case(4, 9, W1[:, :4], 0, "n4_b9_cut3p1", cuts=(3, 1))
    run_case(5, 9, V2[:, :5], 0, "n5_b9_cut3p1", cuts=(3, 1))
    run_case(63, 9, V2[:, :63], 0, "n63_b9_cut3p1", cuts=(3, 1))
    print("ODDB_DONE")
