#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""奇数 b / 奇数 n 的验证图。"""
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

PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]


def render(g, fn, lines, port, cs=26, tc=None):
    Wg, Lg = g.W, g.L
    C, PADL, PADT, LEG = cs, 46, 48, 250
    img = Image.new("RGB", (PADL + Wg*C + LEG, PADT + Lg*C + 10), (16, 16, 20))
    dr = ImageDraw.Draw(img)
    tc = tc or {}

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
                if C >= 24:
                    T((x0+C//2, y0+C//2+8), str(v), (120, 255, 150) if v else (105, 105, 115))
                if p == port:
                    dr.ellipse([x0, y0, x0+C-1, y0+C-1], outline=(90, 220, 255), width=3)
            elif p in g.barrel:
                dr.rectangle([x0, y0, x0+C-1, y0+C-1], fill=(150, 110, 70),
                             outline=(90, 60, 30))
                if C >= 24:
                    T((x0+C//2, y0+C//2), str(int(g.barrel[p])), (255, 245, 220))
            elif p in g.solid:
                if p in g.torch:
                    k = tc.get(p, 7); on = g.torch[p] > 0; base = PAL[k % 8]
                    dr.rectangle([x0, y0, x0+C-1, y0+C-1],
                                 fill=base if on else tuple(int(q*0.35) for q in base))
                    if C >= 24:
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
    return os.path.getsize(fn)


if __name__ == "__main__":
    P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
    mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                        encoding="utf-8"))
    W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
    DELTA, WREF = 7.0, 882.0

    cases = [(4, 9, 0, "oddB"), (3, 8, 0, "oddN_ok"), (3, 8, 35, "oddN_fail"),
             (3, 9, 0, "both_odd")]
    for (n, b, J, tag) in cases:
        Wm = W1[:, :n]
        al = WREF / (np.abs(Wm).max(1) * DELTA / 4)
        U = np.rint(al[:, None, None] * Wm[:, :, None] * (DELTA / (1 << b))
                    * (1 << np.arange(b))[None, None, :]).astype(np.int64)
        G = geom(n, b)
        Jc = (G["zmsb"] - 2) // 6 + 1
        S = (15 + np.abs(U[J])).astype(int)
        msb = np.abs(U[J, :, b - 1]).tolist()
        wsg = (U[J] > 0).astype(int)
        K = int(S.sum())
        W0 = Jc * K - 15 * n * b
        rng = np.random.default_rng(11)
        inp = rng.integers(0, 2, size=(n, b))
        g = build(n, b, S, K, W0, inp, msb, wsg)
        g.run()
        port = (G["xw"], 4)
        head = int(g.cmp.get(port, {}).get("state", -1))
        nt = int((U[J] * inp).sum() - np.minimum(0, U[J]).sum())
        tc = {}
        for k in range(b):
            for i in range(n):
                zc = 3*k+2 if k % 2 == 0 else 3*k+3
                tc[(2*i+3 if k % 2 == 0 else 2*i+4, zc)] = k % 8
        fn = os.path.join(RESULTS, f"par_{tag}.png")
        render(g, fn, [f"{tag}  n={n} b={b}", f"平面 {G['W']}x{G['L']}",
                       f"链数 J={Jc}", f"msb行 z={G['zmsb']}",
                       f"灰环 z={G['zlast_chain']+2-2}",
                       f"K={K}", f"白环={W0}",
                       "", f"head={head}", f"N={nt}",
                       "OK" if head == nt else "FAIL",
                       "", f"MSB(S)={[int(S[i][b-1]) for i in range(n)]}"], port)
        print(f"{tag:10s} n={n} b={b} J={Jc}: head={head:6d} N={nt:6d} "
              f"{'OK' if head == nt else 'FAIL'}  {os.path.getsize(fn)//1024}KB")
    print("PAR_DONE")
