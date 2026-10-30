#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""把模拟器当前的静态状态画出来：每个比较器标上它的输出值。"""
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
from cmp_sim import build, DIRV  # noqa: E402

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

brd = build(msb, S, K, K // 2, inp)
it, ok = brd.run()
print(f"通道 {J}  K={K}  白环初值={K//2}  N_true={Ntrue}")
print(f"迭代 {it} 次，收敛 {ok}")
print(f"非零比较器 {sum(1 for c in brd.cmp.values() if c['state'])}/{len(brd.cmp)}")

Wg, Lg = brd.W, brd.L
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
C, PADL, PADT, LEG = 50, 70, 66, 300
img = Image.new("RGB", (PADL + Wg * C + LEG, PADT + Lg * C + 14), (16, 16, 20))
dr = ImageDraw.Draw(img)


def T(xy, s, fill=(235, 235, 245), anchor="mm"):
    dr.text(xy, s, fill=fill, anchor=anchor)


# 彩色方块 -> 位号
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
        if p in brd.cmp:
            c = brd.cmp[p]
            sub = c["mode"] == "subtract"
            v = c["state"]
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2],
                         fill=(70, 48, 62) if sub else (44, 44, 56),
                         outline=(210, 100, 100) if sub else (110, 110, 130),
                         width=2)
            vv = DIRV[c["out"]]
            acol = (255, 95, 95) if sub else (255, 195, 85)
            dr.line([cx - vv[0]*9, cy - vv[1]*9 - 6, cx + vv[0]*9, cy + vv[1]*9 - 6],
                    fill=acol, width=2)
            dr.polygon([(cx + vv[0]*13, cy + vv[1]*13 - 6),
                        (cx + vv[0]*7 + vv[1]*5, cy + vv[1]*7 - vv[0]*5 - 6),
                        (cx + vv[0]*7 - vv[1]*5, cy + vv[1]*7 + vv[0]*5 - 6)],
                       fill=acol)
            tcol = (120, 255, 150) if v else (110, 110, 120)
            T((cx, cy + 11), str(v), tcol)
            if sub:
                T((x0 + C - 6, y0 + 3), "-", (255, 90, 90), anchor="ra")
        elif p in brd.barrel:
            dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3], fill=(150, 110, 70),
                         outline=(90, 60, 30))
            T((cx, cy), str(brd.barrel[p]), (255, 245, 220))
        elif p in brd.solid:
            if p in brd.torch:
                k = colbit.get(p, 0)
                on = brd.torch[p] > 0
                dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                             fill=PAL[k % 8] if on else tuple(
                                 int(v*0.42) for v in PAL[k % 8]),
                             outline=(200, 200, 200))
                T((cx, cy), "15" if on else "0",
                  (10, 10, 10) if on else (210, 210, 210))
            else:
                wt = brd.solid[p] == "white"
                dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                             fill=(235, 235, 235) if wt else (110, 110, 120))
                rv = brd.read(p, toward=DIRV["N"])
                if rv:
                    T((cx, cy), str(rv), (10, 10, 10) if wt else (255, 255, 255))

for x in range(Wg):
    T((PADL + x*C + C//2, PADT - 20), str(x), (190, 190, 200))
for z in range(Lg):
    T((10, PADT + z*C + C//2), f"z{z}", (190, 190, 200), anchor="lm")

info = [f"通道 {J}", f"K={K}  白环={K//2}", f"N_true={Ntrue}",
        "", "绿字 = 比较器输出值", "灰字 = 0（无源）",
        "− = subtract", "", "彩块上 15/0 = 火把状态",
        "", "灰/白块上的数字 =", "它被充能后的值"]
for t, s in enumerate(info):
    T((PADL + Wg*C + 12, PADT + t*19), s, (228, 228, 238), anchor="la")

out = os.path.join(RESULTS, "sim_state.png")
img.save(out)
print("wrote", out, os.path.getsize(out)//1024, "KB")

# 死区诊断
print()
print("右总线 x=11 的现状:")
for z in range(4, 24):
    p = (11, z)
    if p in brd.cmp:
        c = brd.cmp[p]
        print(f"  (11,{z:2d}) out={c['out']} {c['mode']:8s} = {c['state']}")
print()
print("左总线 x=2 的现状:")
for z in range(1, 24):
    p = (2, z)
    if p in brd.cmp:
        print(f"  (2,{z:2d}) out={brd.cmp[p]['out']} = {brd.cmp[p]['state']}")
print("STATE_DONE")
