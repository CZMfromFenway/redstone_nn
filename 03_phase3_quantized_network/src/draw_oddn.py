#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""奇数 n + 偶数 b 的多例验证与出图。"""
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
from plane_gen import build, geom  # noqa: E402
from draw_diag import diagnose  # noqa: E402

PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]


def render(g, fn, lines, port, dead, orphan, starved, tc, cs):
    Wg, Lg = g.W, g.L
    C, PADL, PADT, LEG = cs, 44, 46, 250
    img = Image.new("RGB", (PADL + Wg*C + LEG, PADT + Lg*C + 12), (16, 16, 20))
    dr = ImageDraw.Draw(img)
    dense = C < 22

    def T(xy, s, fill=(235, 235, 245), anchor="mm"):
        dr.text(xy, s, fill=fill, anchor=anchor)

    ds, orp, st = set(dead), set(orphan), set(starved)
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
                dr.line([x0+C//2-vv[0]*(C//5), y0+C//2-vv[1]*(C//5),
                         x0+C//2+vv[0]*(C//5), y0+C//2+vv[1]*(C//5)],
                        fill=ac, width=2 if not dense else 1)
                if not dense:
                    T((x0+C//2, y0+C//2+8), str(v),
                      (120, 255, 150) if v else (105, 105, 115))
            elif p in g.barrel:
                dr.rectangle([x0, y0, x0+C-1, y0+C-1], fill=(150, 110, 70),
                             outline=(90, 60, 30))
                if not dense:
                    T((x0+C//2, y0+C//2), str(int(g.barrel[p])), (255, 245, 220))
            elif p in g.solid:
                if p in g.torch:
                    k = tc.get(p, 7); on = g.torch[p] > 0; base = PAL[k % 8]
                    dr.rectangle([x0, y0, x0+C-1, y0+C-1],
                                 fill=base if on else tuple(int(q*0.35) for q in base))
                    if not dense:
                        T((x0+C//2, y0+C//2), "15" if on else "·",
                          (10, 10, 10) if on else (215, 215, 215))
                else:
                    wt = g.solid[p] == "white"
                    dr.rectangle([x0, y0, x0+C-1, y0+C-1],
                                 fill=(235, 235, 235) if wt else (110, 110, 120))
            if p in ds:
                dr.line([x0+1, y0+1, x0+C-2, y0+C-2], fill=(255, 40, 40), width=2)
                dr.line([x0+1, y0+C-2, x0+C-2, y0+1], fill=(255, 40, 40), width=2)
            if p in orp:
                dr.rectangle([x0, y0, x0+C-2, y0+C-2], outline=(255, 60, 220), width=2)
            if p in st:
                dr.ellipse([x0+C-8, y0+1, x0+C-1, y0+8], outline=(60, 230, 255), width=2)
    step = 1 if C >= 26 else (5 if C >= 14 else 10)
    for x in range(0, Wg, step):
        T((PADL+x*C+C//2, PADT-13), str(x), (185, 185, 195))
    for z in range(Lg):
        T((2, PADT+z*C+C//2), f"z{z}", (185, 185, 195), anchor="lm")
    for t, s in enumerate(lines):
        T((PADL+Wg*C+8, PADT+t*15), s, (228, 228, 238), anchor="la")
    img.save(fn)
    return os.path.getsize(fn)


def run_case(n, b, JJ, tag, src, cs=None):
    al = WREF / (np.abs(src).max(1) * DELTA / 4)
    U = np.rint(al[:, None, None] * src[:, :, None] * (DELTA / (1 << b))
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    G = geom(n, b); Jc = (G["zmsb"] - 2) // 6 + 1
    ok = 0; tot = 0
    for J in range(0, U.shape[0], 7):
        S = (15 + np.abs(U[J])).astype(int)
        msb = np.abs(U[J, :, b - 1]).tolist()
        wsg = (U[J] > 0).astype(int)
        K = int(S.sum()); W0 = Jc * K - 15 * n * b
        rng = np.random.default_rng(11)
        inp = rng.integers(0, 2, size=(n, b))
        g = build(n, b, S, K, W0, inp, msb, wsg)
        d, o, _ = diagnose(g, (G["xw"], 4))
        g.run()
        h = int(g.cmp.get((G["xw"], 4), {}).get("state", -1))
        nt = int((U[J] * inp).sum() - np.minimum(0, U[J]).sum())
        ok += (h == nt); tot += 1
        if J == JJ:
            tc = {}
            for k in range(b):
                for i in range(n):
                    zc = 3*k+2 if k % 2 == 0 else 3*k+3
                    tc[(2*i+3 if k % 2 == 0 else 2*i+4, zc)] = k % 8
            if (n % 2) and int(S[n-1][b-1]) > 897:
                for dz in (2, 4, 6):
                    tc.setdefault((2*(n-1)+4, G["zmsb"]+1+dz), 7)
            cs_ = cs or (30 if G["W"] <= 20 else (18 if G["W"] <= 40 else 10))
            fn = os.path.join(RESULTS, f"oddn_{tag}.png")
            kb = render(g, fn,
                        [tag, f"n={n} b={b}", f"平面 {G['W']}x{G['L']}",
                         f"链数 J={Jc}", f"MSB 行 z={G['zmsb']}",
                         f"配对 {n//2} 对 + 孤立 1", "",
                         f"通道 {J}", f"head = {h}", f"N    = {nt}",
                         ">> OK <<" if h == nt else ">> FAIL <<", "",
                         f"死比较器 {len(d)}", f"孤儿方块 {len(o)}",
                         "", "MSB(S)=" + str([int(S[i][b-1]) for i in range(n)])[:40]],
                        (G["xw"], 4), d, o, [], tc, cs_)
    print(f"  {tag:14s} n={n:2d} b={b:2d} 平面{G['W']}x{G['L']} J={Jc}: "
          f"{ok}/{tot}")


if __name__ == "__main__":
    P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
    mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                        encoding="utf-8"))
    W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
    V2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
    DELTA, WREF = 7.0, 882.0

    run_case(3, 8, 0, "n3_b8", W1[:, :3])
    run_case(5, 8, 0, "n5_b8", V2[:, :5])
    run_case(7, 8, 0, "n7_b8", V2[:, :7])
    run_case(5, 10, 0, "n5_b10", V2[:, :5])
    run_case(7, 10, 0, "n7_b10", V2[:, :7])
    run_case(15, 10, 0, "n15_b10", V2[:, :15])
    run_case(63, 10, 0, "n63_b10", V2[:, :63])
    print("MULTI_DONE")
