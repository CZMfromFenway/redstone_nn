#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""渲染 cmp_sim3 的状态：每个比较器标输出值。"""
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
from cmp_sim3 import build  # noqa: E402

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
np.random.seed(1)
inp = np.random.randint(0, 2, size=(N, B))
K = int(S.sum())
Ntrue = int((U[J] * inp).sum() - np.minimum(0, U[J]).sum())

g = build(N, B, S, K, K, inp, True, msb)
it, ok = g.run()
print(f"K={K} 白环={K} N_true={Ntrue} 迭代{it} 收敛{ok}")
print(f"右总线: " + " ".join(f"{z}:{g.cmp[(11,z)]['state']}"
                             for z in range(4, 23, 2)))

Wg, Lg = g.W, g.L
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
C, PADL, PADT, LEG = 50, 70, 66, 290
img = Image.new("RGB", (PADL + Wg * C + LEG, PADT + Lg * C + 14), (16, 16, 20))
dr = ImageDraw.Draw(img)


def T(xy, s, fill=(235, 235, 245), anchor="mm"):
    dr.text(xy, s, fill=fill, anchor=anchor)


colbit = {}
for k in range(B - 1):
    col = [3, 5, 7, 9] if k % 2 == 0 else [4, 6, 8, 10]
    zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
    for i in range(N):
        colbit[(col[i], zc)] = k
for i in range(N):
    colbit[(4 + 2 * i, 24)] = B - 1

for z in range(Lg):
    for x in range(Wg):
        p = (x, z)
        x0, y0 = PADL + x * C, PADT + z * C
        cx, cy = x0 + C // 2, y0 + C // 2
        if p in g.cmp:
            c = g.cmp[p]
            sub = c["mode"] == "subtract"
            v = int(c["state"])
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2],
                         fill=(70, 48, 62) if sub else (44, 44, 56),
                         outline=(210, 100, 100) if sub else (110, 110, 130),
                         width=2)
            vv = DIRV[c["out"]]
            ac = (255, 95, 95) if sub else (255, 195, 85)
            dr.line([cx - vv[0]*9, cy - vv[1]*9 - 6, cx + vv[0]*9,
                     cy + vv[1]*9 - 6], fill=ac, width=2)
            dr.polygon([(cx + vv[0]*13, cy + vv[1]*13 - 6),
                        (cx + vv[0]*7 + vv[1]*5, cy + vv[1]*7 - vv[0]*5 - 6),
                        (cx + vv[0]*7 - vv[1]*5, cy + vv[1]*7 + vv[0]*5 - 6)],
                       fill=ac)
            T((cx, cy + 11), str(v), (120, 255, 150) if v else (110, 110, 120))
            if sub:
                T((x0 + C - 5, y0 + 2), "-", (255, 90, 90), anchor="ra")
        elif p in g.barrel:
            dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3], fill=(150, 110, 70),
                         outline=(90, 60, 30))
            T((cx, cy), str(int(g.barrel[p])), (255, 245, 220))
        elif p in g.solid:
            if p in g.torch:
                k = colbit.get(p, 0)
                on = g.torch[p] > 0
                base = PAL[k % 8]
                dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                             fill=base if on else tuple(int(v*0.40) for v in base),
                             outline=(210, 210, 210))
                T((cx, cy), "15" if on else "0",
                  (10, 10, 10) if on else (215, 215, 215))
            else:
                wt = g.solid[p] == "white"
                dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                             fill=(235, 235, 235) if wt else (110, 110, 120))
                rv = g.read(p, toward=DIRV["N"])
                if rv:
                    T((cx, cy), str(int(rv)), (10, 10, 10) if wt else (240, 240, 240))

for x in range(Wg):
    T((PADL + x*C + C//2, PADT - 20), str(x), (190, 190, 200))
for z in range(Lg):
    T((10, PADT + z*C + C//2), f"z{z}", (190, 190, 200), anchor="lm")

info = [f"通道 {J}", f"K={K} 白环={K}", f"N_true={Ntrue}", "",
        "绿=比较器输出", "灰=0", "- = subtract", "",
        "彩块 15/0 = 火把", "灰/白块数字 = 充能值"]
for t, s in enumerate(info):
    T((PADL + Wg*C + 12, PADT + t*19), s, (228, 228, 238), anchor="la")

out = os.path.join(RESULTS, "sim_state2.png")
img.save(out)
print("wrote", out, os.path.getsize(out)//1024, "KB")
print("R2_DONE")
