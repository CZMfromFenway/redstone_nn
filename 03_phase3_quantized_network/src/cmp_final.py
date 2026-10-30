#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""四个变体（环平移 2 格）+ 完整断头检测 + 出图。"""
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

PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
ODD, EVEN = [3, 5, 7, 9], [4, 6, 8, 10]


def deadends(g, port=None):
    """返回 (死比较器, 孤儿方块)。两类节点都查。"""
    d_cmp, d_blk = [], []
    for p, c in g.cmp.items():
        d = DIRV[c["out"]]
        b = (p[0] + d[0], p[1] + d[1])
        used = False
        if b in g.cmp:
            cc = g.cmp[b]; bd = DIRV[cc["out"]]
            if (b[0] - bd[0], b[1] - bd[1]) == p:
                used = True
            for s in [(-bd[1], bd[0]), (bd[1], -bd[0])]:
                if (b[0] + s[0], b[1] + s[1]) == p and d == (-s[0], -s[1]):
                    used = True
        elif b in g.barrel or b in g.solid:
            for q, cc in g.cmp.items():
                if q == p:
                    continue
                if (q[0] - DIRV[cc["out"]][0], q[1] - DIRV[cc["out"]][1]) == b:
                    used = True
                    break
        if p == port:
            used = True   # 输出端口本来就指向空气
        if not used:
            d_cmp.append((p, c["out"], int(c["state"]), b))
    for p in list(g.solid.keys()) + list(g.barrel.keys()):
        if p in g.torch and g.torch[p] > 0:
            continue
        chg = any((q[0] + DIRV[cc["out"]][0], q[1] + DIRV[cc["out"]][1]) == p
                  for q, cc in g.cmp.items())
        rd = False
        for q, cc in g.cmp.items():
            d = DIRV[cc["out"]]
            r1 = (q[0] - d[0], q[1] - d[1])
            r2 = (q[0] - 2*d[0], q[1] - 2*d[1])
            if r1 == p or (r2 == p and r1 in g.solid):
                rd = True; break
        if not chg and not rd:
            d_blk.append(p)
    return d_cmp, d_blk


def render(g, fn, lines, port):
    Wg, Lg = g.W, g.L
    C, PADL, PADT, LEG = 44, 60, 66, 260
    img = Image.new("RGB", (PADL + Wg*C + LEG, PADT + Lg*C + 12), (16, 16, 20))
    dr = ImageDraw.Draw(img)

    def T(xy, s, fill=(235, 235, 245), anchor="mm"):
        dr.text(xy, s, fill=fill, anchor=anchor)

    colbit = {}
    for k in range(8):
        col = ODD if k % 2 == 0 else EVEN
        zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
        for i in range(4):
            colbit[(col[i], zc)] = k
    for i in range(4):
        colbit[(4 + 2 * i, 24)] = 7

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
                dr.line([cx-vv[0]*8, cy-vv[1]*8-6, cx+vv[0]*8, cy+vv[1]*8-6],
                        fill=ac, width=2)
                dr.polygon([(cx+vv[0]*12, cy+vv[1]*12-6),
                            (cx+vv[0]*6+vv[1]*4, cy+vv[1]*6-vv[0]*4-6),
                            (cx+vv[0]*6-vv[1]*4, cy+vv[1]*6+vv[0]*4-6)], fill=ac)
                T((cx, cy+10), str(v), (120, 255, 150) if v else (110, 110, 120))
                if p == port:
                    dr.ellipse([x0+1, y0+1, x0+C-2, y0+C-2],
                               outline=(90, 220, 255), width=3)
            elif p in g.barrel:
                dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3], fill=(150, 110, 70),
                             outline=(90, 60, 30))
                T((cx, cy), str(int(g.barrel[p])), (255, 245, 220))
            elif p in g.solid:
                if p in g.torch:
                    k = colbit.get(p, 0); on = g.torch[p] > 0
                    base = PAL[k % 8]
                    dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                                 fill=base if on else tuple(int(q*0.40) for q in base),
                                 outline=(215, 215, 215))
                    T((cx, cy), "15" if on else "·",
                      (10, 10, 10) if on else (215, 215, 215))
                else:
                    wt = g.solid[p] == "white"
                    dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                                 fill=(235, 235, 235) if wt else (110, 110, 120))
                    rv = g.read(p, toward=DIRV["N"])
                    if rv:
                        T((cx, cy), str(int(rv)),
                          (10, 10, 10) if wt else (240, 240, 240))
    for x in range(Wg):
        T((PADL+x*C+C//2, PADT-18), str(x), (190, 190, 200))
    for z in range(Lg):
        T((6, PADT+z*C+C//2), f"z{z}", (190, 190, 200), anchor="lm")
    for t, s in enumerate(lines):
        T((PADL+Wg*C+10, PADT+t*18), s, (228, 228, 238), anchor="la")
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
    J = 8
    S = (15 + np.abs(U[J])).astype(int)
    msb = np.abs(U[J, :, B - 1]).tolist()
    wsg = (U[J] > 0).astype(int)
    K = int(S.sum()); W0 = 4 * K - 15 * N * B
    rng = np.random.default_rng(11)
    inp = rng.integers(0, 2, size=(N, B))
    Nt = int((U[J] * inp).sum() - np.minimum(0, U[J]).sum())

    print(f"K={K} 白环={W0} N={Nt}")
    for gu in (True, False):
        for wu in (True, False):
            g = build(N, B, S, K, W0, inp, msb, wsg, grey_up=gu,
                      white_up=wu, gshift=2)
            dc, db = deadends(g, port=((11, 4) if wu else (11, 22)))
            g.run()
            port = (11, 4) if wu else (11, 22)
            head = int(g.cmp.get(port, {}).get("state", -1))
            fn = os.path.join(RESULTS, f"final_{int(gu)}{int(wu)}.png")
            render(g, fn, [f"灰{'上' if gu else '下'} 白{'上' if wu else '下'} 环平移2",
                           f"端口{port} = {head}", f"N = {Nt}",
                           f"死比较器 {len(dc)}", f"孤儿方块 {len(db)}"], port)
            print(f"  灰{'上' if gu else '下'} 白{'上' if wu else '下'}: "
                  f"head={head:6d} {'OK' if head == Nt else 'X'}")
            for p, o, v, b in dc:
                print(f"      死比较器 {p} out={o} 值={v} → {b}")
            for p in db:
                print(f"      孤儿方块 {p}")
    print("FINAL_DONE")
