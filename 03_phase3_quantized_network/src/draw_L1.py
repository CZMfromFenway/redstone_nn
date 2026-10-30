#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""画层1 的 4 个通道（覆盖 5 种品红端排列）+ 层2 单通道。"""
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
from cmp_sim import DIRV  # noqa: E402
from cmp_variants import build  # noqa: E402
from plane_gen import classify_pair  # noqa: E402

PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
ODD, EVEN = [3, 5, 7, 9], [4, 6, 8, 10]


def render(g, fn, lines, port, cs=40):
    Wg, Lg = g.W, g.L
    C, PADL, PADT, LEG = cs, 56, 62, 250
    img = Image.new("RGB", (PADL + Wg*C + LEG, PADT + Lg*C + 12), (16, 16, 20))
    dr = ImageDraw.Draw(img)

    def T(xy, s, fill=(235, 235, 245), anchor="mm"):
        dr.text(xy, s, fill=fill, anchor=anchor)

    colbit = getattr(g, "colbit", {})
    for z in range(Lg):
        for x in range(Wg):
            p = (x, z)
            x0, y0 = PADL + x*C, PADT + z*C
            cx, cy = x0 + C//2, y0 + C//2
            if p in g.cmp:
                c = g.cmp[p]; sub = c["mode"] == "subtract"; v = int(c["state"])
                dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2],
                             fill=(72, 48, 62) if sub else (44, 44, 56),
                             outline=(215, 100, 100) if sub else (110, 110, 130))
                vv = DIRV[c["out"]]
                ac = (255, 95, 95) if sub else (255, 195, 85)
                dr.line([cx-vv[0]*7, cy-vv[1]*7-5, cx+vv[0]*7, cy+vv[1]*7-5],
                        fill=ac, width=2)
                dr.polygon([(cx+vv[0]*11, cy+vv[1]*11-5),
                            (cx+vv[0]*5+vv[1]*4, cy+vv[1]*5-vv[0]*4-5),
                            (cx+vv[0]*5-vv[1]*4, cy+vv[1]*5+vv[0]*4-5)], fill=ac)
                if C >= 34:
                    T((cx, cy+9), str(v), (120, 255, 150) if v else (110, 110, 120))
                if p == port:
                    dr.ellipse([x0+1, y0+1, x0+C-2, y0+C-2],
                               outline=(90, 220, 255), width=3)
            elif p in g.barrel:
                dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3], fill=(150, 110, 70),
                             outline=(90, 60, 30))
                if C >= 30:
                    T((cx, cy), str(int(g.barrel[p])), (255, 245, 220))
            elif p in g.solid:
                if p in g.torch:
                    k = colbit.get(p, 7); on = g.torch[p] > 0
                    base = PAL[k % 8]
                    dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                                 fill=base if on else tuple(int(q*0.40) for q in base),
                                 outline=(215, 215, 215))
                    if C >= 30:
                        T((cx, cy), "15" if on else "·",
                          (10, 10, 10) if on else (215, 215, 215))
                else:
                    wt = g.solid[p] == "white"
                    dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                                 fill=(235, 235, 235) if wt else (110, 110, 120))
    for x in range(0, Wg, 2):
        T((PADL+x*C+C//2, PADT-16), str(x), (190, 190, 200))
    for z in range(Lg):
        T((6, PADT+z*C+C//2), f"z{z}", (190, 190, 200), anchor="lm")
    for t, s in enumerate(lines):
        T((PADL+Wg*C+10, PADT+t*17), s, (228, 228, 238), anchor="la")
    img.save(fn)


if __name__ == "__main__":
    P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
    mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                        encoding="utf-8"))
    W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
    DELTA, WREF, B, N = 7.0, 882.0, 8, 4
    alpha = WREF / (np.abs(W1).max(1) * DELTA / 4)
    U = np.rint(alpha[:, None, None] * W1[:, :, None] * (DELTA / (1 << B))
                * (1 << np.arange(B))[None, None, :]).astype(np.int64)
    K0 = None

    def mkchan(J):
        S = (15 + np.abs(U[J])).astype(int)
        msb = np.abs(U[J, :, B - 1]).tolist()
        wsg = (U[J] > 0).astype(int)
        K = int(S.sum()); W0 = 4 * K - 15 * N * B
        rng = np.random.default_rng(11)
        inp = rng.integers(0, 2, size=(N, B))
        return S, msb, wsg, K, W0, inp

    rng = np.random.default_rng(11)
    INP = rng.integers(0, 2, size=(N, B))

    for J in (0, 2, 3, 4):
        S = (15 + np.abs(U[J])).astype(int)
        msb = np.abs(U[J, :, B - 1]).tolist()
        wsg = (U[J] > 0).astype(int)
        K = int(S.sum()); W0 = 4 * K - 15 * N * B
        g = build(N, B, S, K, W0, INP, msb, wsg, grey_up=True, white_up=True, gshift=2)
        tc = {}
        for k in range(8):
            col = ODD if k % 2 == 0 else EVEN
            zc = 3*k+2 if k % 2 == 0 else 3*k+3
            for i in range(4):
                tc[(col[i], zc)] = k
        for i in range(4):
            tc[(4+2*i, 24)] = 7
        for x in (4, 6, 8, 10):
            for zz in (27, 29, 31):
                tc[(x, zz)] = 7
        g.colbit = tc
        g.run()
        head = int(g.cmp[(11, 4)]["state"])
        Nt = int((U[J] * INP).sum() - np.minimum(0, U[J]).sum())
        cs = [classify_pair(int(S[0][7]), int(S[1][7])),
              classify_pair(int(S[2][7]), int(S[3][7]))]
        fn = os.path.join(RESULTS, f"L1_ch{J}.png")
        render(g, fn, [f"层1 通道 {J}", f"pair0 = {cs[0]}", f"pair1 = {cs[1]}",
                       f"MSB {(S[0][7],S[1][7])}", f"    {(S[2][7],S[3][7])}",
                       f"head={head}", f"N={Nt}", "OK" if head == Nt else "X"],
               (11, 4))
        print(f"  通道{J}: {cs}  head={head} N={Nt} {'OK' if head==Nt else 'X'}")
    print("L1_DONE")
