#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""当前理解下的状态图 + 断点标注。"""
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

P_WHITE = 4 * K - 15 * N * B      # 由"全亮→0, 全灭→N_max"标定的固定值
print("白环烧 =", P_WHITE, "(= 4K - 15nb)")
g = build(N, B, S, K, P_WHITE, inp, True, msb, wsign=(U[J] > 0).astype(int))
g.run()

Wg, Lg = g.W, g.L
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
C, PADL, PADT, LEG = 52, 74, 70, 300
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
                         fill=(72, 48, 62) if sub else (44, 44, 56),
                         outline=(215, 100, 100) if sub else (110, 110, 130),
                         width=2)
            vv = DIRV[c["out"]]
            ac = (255, 95, 95) if sub else (255, 195, 85)
            dr.line([cx - vv[0]*10, cy - vv[1]*10 - 7, cx + vv[0]*10,
                     cy + vv[1]*10 - 7], fill=ac, width=2)
            dr.polygon([(cx + vv[0]*14, cy + vv[1]*14 - 7),
                        (cx + vv[0]*8 + vv[1]*5, cy + vv[1]*8 - vv[0]*5 - 7),
                        (cx + vv[0]*8 - vv[1]*5, cy + vv[1]*8 + vv[0]*5 - 7)],
                       fill=ac)
            T((cx, cy + 12), str(v), (120, 255, 150) if v else (110, 110, 120))
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

# ---- 标注
# 右总线框
dr.rectangle([PADL + 11*C - 7, PADT + 4*C - 7,
              PADL + 12*C + 6, PADT + 23*C + 6], outline=(255, 200, 60), width=3)
T((PADL + 11*C + C//2, PADT + 2*C), "右总线", (255, 200, 60))
# 断裂点
dr.rectangle([PADL + 11*C - 9, PADT + 16*C - 9,
              PADL + 12*C + 8, PADT + 17*C + 8], outline=(255, 60, 60), width=4)
T((PADL + 11*C + C + 10, PADT + 16*C + C//2), "← 断", (255, 80, 80), anchor="lm")
# 输出端
T((PADL + 11*C + C + 10, PADT + 4*C + C//2), "← 输出端", (120, 220, 255), anchor="lm")
# 两个环
dr.rectangle([PADL - 6, PADT + 24*C - 6, PADL + 3*C + 4, PADT + 27*C + 4],
             outline=(90, 255, 160), width=3)
T((PADL, PADT + 24*C - 20), "灰环", (90, 255, 160), anchor="lm")
dr.rectangle([PADL + 11*C - 6, PADT + 23*C - 6,
              PADL + 14*C + 4, PADT + 26*C + 4], outline=(120, 220, 255), width=3)
T((PADL + 11*C, PADT + 23*C - 20), "白环", (120, 220, 255), anchor="lm")

for x in range(Wg):
    T((PADL + x*C + C//2, PADT - 22), str(x), (190, 190, 200))
for z in range(Lg):
    T((10, PADT + z*C + C//2), f"z{z}", (190, 190, 200), anchor="lm")

chain = [g.cmp[(10, z)]["state"] for z in (4, 10, 16, 22)]
bus = [(z, g.cmp[(11, z)]["state"]) for z in range(4, 23, 2)]
info = [f"通道 {J}", f"K = {K}", f"N_理论 = {Ntrue}", "",
        "四条链末端 (x=10):",
        f"  z4  {int(chain[0])}", f"  z10 {int(chain[1])}",
        f"  z16 {int(chain[2])}", f"  z22 {int(chain[3])}", "",
        "右总线 (x=11):"]
for z, v in bus:
    info.append(f"  z{z}  {int(v)}")
for t, s in enumerate(info):
    T((PADL + Wg*C + 12, PADT + t*17), s, (228, 228, 238), anchor="la")

out = os.path.join(RESULTS, "sim_state9.png")
img.save(out)
print("wrote", out, os.path.getsize(out)//1024, "KB")
print("链末端:", [int(v) for v in chain])
print("右总线:", [(z, int(g.cmp[(11, z)]["state"])) for z in range(4, 23, 2)])
print("输出端 (11,4) =", int(g.cmp[(11, 4)]["state"]), " 应为 N =", Ntrue)
print("S3_DONE")
