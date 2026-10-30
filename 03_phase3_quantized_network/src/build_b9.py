#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""用显式行表构建 b=9 的两套方案，摆出全部比较器并验证数值。"""
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

FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]


def build_tab(n, b, S, K, P, inp, msb, wsign, bitrow, msbrow, chainrows,
              brow, gz, wz, zr, L):
    W = 2 * n + 6
    xw = 2 * n + 3
    g = Board(W, L)
    ODD = [2 * i + 3 for i in range(n)]
    EVEN = [2 * i + 4 for i in range(n)]

    def C(p, fac, mode="compare", init=0):
        g.set_cmp(p, FLIP[fac], mode, init)

    # 灰环
    for (x, z, f) in ((1, gz, "E"), (2, gz + 1, "S"), (1, gz + 2, "W"), (0, gz + 1, "N")):
        C((x, z), f, "compare", K)
    for (x, z) in ((0, gz), (2, gz), (0, gz + 2), (2, gz + 2)):
        g.set_solid((x, z), "base")
    # 白环
    for (x, z, f) in ((xw + 1, wz, "W"), (xw + 2, wz + 1, "N"),
                      (xw + 1, wz + 2, "E"), (xw, wz + 1, "S")):
        C((x, z), f, "compare", P)
    for (x, z) in ((xw, wz), (xw + 2, wz), (xw, wz + 2), (xw + 2, wz + 2)):
        g.set_solid((x, z), "white")
    # 输入行
    for k in range(b - 1):
        col = ODD if k % 2 == 0 else EVEN
        for i in range(n):
            on = (inp[i][k] == 1) != (wsign[i][k] > 0)
            g.set_solid((col[i], bitrow[k]), "col")
            g.set_torch((col[i], bitrow[k]), 15 if on else 0)
            g.set_barrel((col[i], brow[k]), int(S[i][k]))
    # 链
    for zc in chainrows:
        for x in ODD:
            C((x, zc - 1), "N")
        for x in ODD + EVEN:
            C((x, zc), "W", "subtract")
        for x in EVEN:
            C((x, zc + 1), "S")
    # 左总线
    zlc = max(chainrows)
    for z in range(5, zlc):
        p = (2, z)
        if p in g.cmp:
            continue
        if z % 2 == 0:
            g.set_solid(p, "base")
        else:
            C(p, "S")
    g.set_solid((2, 4), "base")
    g.set_solid((2, zlc), "base")
    # 右总线
    for z in range(4, zlc + 1):
        p = (xw, z)
        if z % 2 == 0:
            C(p, "S", "subtract" if z in chainrows else "compare")
        else:
            g.set_solid(p, "white")
    # MSB 行
    for i in range(n):
        on = (inp[i][b - 1] == 1) != (wsign[i][b - 1] > 0)
        g.set_solid((EVEN[i], msbrow), "col")
        g.set_torch((EVEN[i], msbrow), 15 if on else 0)
    # 配对区
    for pi in range(n // 2):
        xa, xb = EVEN[2 * pi], EVEN[2 * pi + 1]
        Sa, Sb = int(S[2 * pi][b - 1]), int(S[2 * pi + 1][b - 1])
        big_xa = Sa >= Sb
        g.set_barrel((xa + 1, zr), abs(Sa - Sb))
        C((xa, zr), "S"); C((xb, zr), "S")
        C((xa, zr + 1), "S"); C((xa + 1, zr + 1), "N"); C((xb, zr + 1), "S")
        for x in (xa, xb):
            for dz in (2, 4, 6):
                g.set_solid((x, zr + dz), "col")
                g.set_torch((x, zr + dz), 0)
        C((xa + 1, zr + 2), "W" if big_xa else "E", "subtract")
        C((xa, zr + 3) if big_xa else (xb, zr + 3), "S")
        sm = max(Sa, Sb)
        C((xa + 1, zr + 4), "E", "compare", sm)
        C((xa, zr + 5), "N", "compare", sm)
        C((xb, zr + 5), "S", "compare", sm)
        C((xa + 1, zr + 6), "W", "compare", sm)
    return g, xw


def render(g, fn, lines, port, cs=30):
    Wg, Lg = g.W, g.L
    C, PADL, PADT, LEG = cs, 46, 46, 250
    img = Image.new("RGB", (PADL + Wg*C + LEG, PADT + Lg*C + 12), (16, 16, 20))
    dr = ImageDraw.Draw(img)

    def T(xy, s, fill=(235, 235, 245), anchor="mm"):
        dr.text(xy, s, fill=fill, anchor=anchor)

    for z in range(Lg):
        for x in range(Wg):
            p = (x, z)
            x0, y0 = PADL + x*C, PADT + z*C
            if p in g.cmp:
                c = g.cmp[p]; sub = c["mode"] == "subtract"; v = int(c["state"])
                dr.rectangle([x0, y0, x0+C-1, y0+C-1],
                             fill=(92, 50, 72) if sub else (44, 44, 56),
                             outline=(215, 95, 95) if sub else (95, 95, 110))
                vv = DIRV[c["out"]]
                ac = (255, 95, 95) if sub else (255, 195, 85)
                dr.line([x0+C//2-vv[0]*5, y0+C//2-vv[1]*5,
                         x0+C//2+vv[0]*5, y0+C//2+vv[1]*5], fill=ac, width=2)
                T((x0+C//2, y0+C//2+8), str(v),
                  (120, 255, 150) if v else (105, 105, 115))
            elif p in g.barrel:
                dr.rectangle([x0, y0, x0+C-1, y0+C-1], fill=(150, 110, 70),
                             outline=(90, 60, 30))
                T((x0+C//2, y0+C//2), str(int(g.barrel[p])), (255, 245, 220))
            elif p in g.solid:
                if p in g.torch:
                    k = getattr(g, "colbit", {}).get(p, 7)
                    on = g.torch[p] > 0; base = PAL[k % 8]
                    dr.rectangle([x0, y0, x0+C-1, y0+C-1],
                                 fill=base if on else tuple(int(q*0.35) for q in base))
                    T((x0+C//2, y0+C//2), "15" if on else "·",
                      (10, 10, 10) if on else (215, 215, 215))
                else:
                    wt = g.solid[p] == "white"
                    dr.rectangle([x0, y0, x0+C-1, y0+C-1],
                                 fill=(235, 235, 235) if wt else (110, 110, 120))
    for x in range(Wg):
        T((PADL+x*C+C//2, PADT-14), str(x), (185, 185, 195))
    for z in range(Lg):
        T((2, PADT+z*C+C//2), f"z{z}", (185, 185, 195), anchor="lm")
    for t, s in enumerate(lines):
        T((PADL+Wg*C+8, PADT+t*15), s, (228, 228, 238), anchor="la")
    img.save(fn)


if __name__ == "__main__":
    P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
    mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                        encoding="utf-8"))
    W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
    DELTA, WREF = 7.0, 882.0
    n, b = 4, 9
    Wm = W1[:, :n]
    al = WREF / (np.abs(Wm).max(1) * DELTA / 4)
    U = np.rint(al[:, None, None] * Wm[:, :, None] * (DELTA / (1 << b))
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    J = 0
    S = (15 + np.abs(U[J])).astype(int)
    msb = np.abs(U[J, :, b - 1]).tolist()
    wsg = (U[J] > 0).astype(int)
    K = int(S.sum())
    rng = np.random.default_rng(11)
    inp = rng.integers(0, 2, size=(n, b))
    Nt = int((U[J] * inp).sum() - np.minimum(0, U[J]).sum())

    SCHEMES = {
        "A": dict(bitrow={0: 6, 1: 8, 2: 12, 3: 14, 4: 18, 5: 20, 6: 24, 7: 26},
                  msbrow=30, chains=[10, 16, 22, 28],
                  brow={0: 5, 1: 7, 2: 7, 3: 13, 4: 13, 5: 19, 6: 19, 7: 25},
                  gz=28, wz=29, zr=31, L=38),
        "B": dict(bitrow={0: 2, 1: 6, 2: 8, 3: 12, 4: 14, 5: 18, 6: 20, 7: 24},
                  msbrow=26, chains=[4, 10, 16, 22, 28],
                  brow={0: 1, 1: 7, 2: 7, 3: 13, 4: 13, 5: 19, 6: 19, 7: 25},
                  gz=28, wz=25, zr=27, L=34),
    }
    for tag, sc in SCHEMES.items():
        Jc = len(sc["chains"])
        W0 = Jc * K - 15 * n * b
        g, xw = build_tab(n, b, S, K, W0, inp, msb, wsg,
                          sc["bitrow"], sc["msbrow"], sc["chains"],
                          sc["brow"], sc["gz"], sc["wz"], sc["zr"], sc["L"])
        tc = {}
        for k, zc in sc["bitrow"].items():
            col = [2*i+3 for i in range(n)] if k % 2 == 0 else [2*i+4 for i in range(n)]
            for x in col:
                tc[(x, zc)] = k % 8
        for i in range(n):
            tc[(4 + 2*i, sc["msbrow"])] = 7
        g.colbit = tc
        g.run()
        head = int(g.cmp.get((xw, 4), {}).get("state", -1))
        fn = os.path.join(RESULTS, f"full_{tag}.png")
        render(g, fn, [f"方案 {tag}", f"n={n} b={b}  L={sc['L']}  链={Jc}",
                       f"MSB 行 z={sc['msbrow']}", f"配对区 z={sc['zr']}..",
                       "", f"K={K}", f"白环={W0}",
                       "", f"head = {head}", f"N    = {Nt}",
                       ">> OK <<" if head == Nt else ">> FAIL <<",
                       "", "MSB(S)=" + str([int(S[i][8]) for i in range(n)])],
               (xw, 4))
        print(f"方案 {tag}: L={sc['L']} 链={Jc} head={head} N={Nt} "
              f"{'OK' if head == Nt else 'FAIL'}")
    print("FULL_DONE")
