#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
层1 单平面 逐块完整图  v2（按用户指正修正）

修正
 ① 比较器朝向整体旋转 180°：schem 里的 facing 与图上的信号流向相反
 ② 灰环 K 从左上搬到白环 P 同侧（x=11..13, z=26..28）
 ③ 白环线路上 4 个比较器改成 subtract：(11,4) (11,10) (11,16) (11,22)
 ④ 补上品红环的输入通道：(6,28) 与 (10,28) 两个朝上的比较器
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
msb = np.abs(U[J, :, B - 1])
PAIRS = [(0, 1), (2, 3)]
CASE = []
for a, b in PAIRS:
    lo, hi = min(msb[a], msb[b]), max(msb[a], msb[b])
    CASE.append("I" if lo > 882 else ("II" if hi > 882 else "III"))
print(f"通道 {J}  MSB={msb.tolist()}  配对情形={CASE}")

FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}
Wg, Lg = 14, 32
G = [[None] * Wg for _ in range(Lg)]


def put(x, z, v):
    if 0 <= x < Wg and 0 <= z < Lg:
        G[z][x] = v


ODD, EVEN = [3, 5, 7, 9], [4, 6, 8, 10]

# ① 灰环 K 搬走 -> 左上三行空出
# ② 灰环 K 放在白环下方 (x=11..13, z=26..28)
put(12, 26, ("cmp", FLIP["E"])); put(13, 27, ("cmp", FLIP["S"]))
put(12, 28, ("cmp", FLIP["W"])); put(11, 27, ("cmp", FLIP["N"]))
for (x, z) in ((11, 26), (13, 26), (11, 28), (13, 28)):
    put(x, z, "base")

# 主体
for k in range(B - 1):
    col = ODD if k % 2 == 0 else EVEN
    zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
    zb = 3 * k + 1 if k % 2 == 0 else 3 * k + 4
    for i in range(N):
        put(col[i], zc, ("col", k))
        put(col[i], zb, ("bar", k, i, int(S[i, k])))

for j in range(4):
    z3, z4, z5 = 6 * j + 3, 6 * j + 4, 6 * j + 5
    put(2, z3, ("cmp", FLIP["N"]))
    for x in ODD:
        put(x, z3, ("cmp", FLIP["N"]))
    for x in ODD + EVEN:
        put(x, z4, ("sub", FLIP["W"]))
    put(2, z5, ("cmp", FLIP["N"]))
    for x in EVEN:
        put(x, z5, ("cmp", FLIP["S"]))

# 左总线
for z in range(1, 25):
    if G[z][2] is not None:
        continue
    put(2, z, ("cmp", FLIP["S"]) if z == 1
        else (("cmp", FLIP["N"]) if z % 2 else "base"))

# 右总线 + ③ 四个减法
SUBS = {4, 10, 16, 22}
for z in range(4, 23):
    if z in SUBS:
        put(11, z, ("sub", FLIP["S"]))
    elif z % 2 == 0:
        put(11, z, ("cmp", FLIP["S"]))
    else:
        put(11, z, "white")

for i in range(N):
    put(EVEN[i], 24, ("col", B - 1))

# 白环 P (11..13, 23..25)
put(12, 23, ("cmp", FLIP["E"])); put(13, 24, ("cmp", FLIP["S"]))
put(12, 25, ("cmp", FLIP["W"])); put(11, 24, ("cmp", FLIP["N"]))
for (x, z) in ((11, 23), (13, 23), (11, 25), (13, 25)):
    put(x, z, "white")

# 配对区
for pi, (a, b) in enumerate(PAIRS):
    xa, xb = EVEN[a], EVEN[b]
    case = CASE[pi]
    if case == "III":
        put(xa, 25, ("bar", B - 1, a, int(S[a, B - 1])))
        put(xb, 25, ("bar", B - 1, b, int(S[b, B - 1])))
        continue
    hi_i = a if S[a, B - 1] >= S[b, B - 1] else b
    lo_i = b if hi_i == a else a
    put(xa, 25, ("cmp", FLIP["S"])); put(xb, 25, ("cmp", FLIP["S"]))
    put(xa + 1, 25, ("bar", "diff", pi, int(S[hi_i, B - 1] - S[lo_i, B - 1])))
    put(xa, 26, ("cmp", FLIP["S"])); put(xb, 26, ("cmp", FLIP["S"]))
    if case == "I":
        put(xa + 1, 26, ("cmp", FLIP["N"]))
        put(xa, 27, ("ring", B - 1)); put(xb, 27, ("ring", B - 1))
        put(xa + 1, 27, ("sub", FLIP["W"]))
    # ④ 环输入通道
    put(xb, 28, ("cmp", FLIP["S"]))
    put(xa, 29, ("ring", B - 1)); put(xb, 29, ("ring", B - 1))
    put(xa + 1, 29, ("cmp", FLIP["E"]))
    put(xa, 30, ("cmp", FLIP["N"])); put(xb, 30, ("cmp", FLIP["S"]))
    put(xa, 31, ("ring", B - 1)); put(xb, 31, ("ring", B - 1))
    put(xa + 1, 31, ("cmp", FLIP["W"]))

# ---------------- 画
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
C, PADL, PADT, LEG = 44, 78, 80, 310
img = Image.new("RGB", (PADL + Wg * C + LEG, PADT + Lg * C + 16), (20, 20, 26))
dr = ImageDraw.Draw(img)


def txt(xy, s, fill=(235, 235, 245), anchor="mm"):
    dr.text(xy, s, fill=fill, anchor=anchor)


for z in range(Lg):
    for x in range(Wg):
        v = G[z][x]
        if v is None:
            continue
        x0, y0 = PADL + x * C, PADT + z * C
        cx, cy = x0 + C // 2, y0 + C // 2
        if isinstance(v, tuple) and v[0] == "bar":
            kind, i, val = v[1], v[2], v[3]
            col = (150, 110, 70) if kind == "diff" else PAL[kind % 8]
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2], fill=col, outline=(15, 15, 15))
            txt((cx, cy - 7), f"{val}", (10, 10, 10))
            txt((cx, cy + 8), ("Δ" if kind == "diff" else f"i{i}"), (10, 10, 10))
        elif isinstance(v, tuple) and v[0] in ("col", "ring"):
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2], fill=PAL[v[1] % 8],
                         outline=(80, 80, 80))
            if v[0] == "col":
                txt((cx, cy), "in", (20, 20, 20))
        elif isinstance(v, tuple) and v[0] in ("cmp", "sub"):
            d = v[1]
            sub = v[0] == "sub"
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2],
                         fill=(64, 50, 70) if sub else (52, 52, 62),
                         outline=(200, 100, 100) if sub else (115, 115, 130), width=2)
            vv = {"E": (1, 0), "W": (-1, 0), "N": (0, -1), "S": (0, 1)}[d]
            col = (255, 100, 100) if sub else (255, 200, 90)
            dr.line([cx - vv[0]*11, cy - vv[1]*11, cx + vv[0]*11, cy + vv[1]*11],
                    fill=col, width=3)
            dr.polygon([(cx + vv[0]*16, cy + vv[1]*16),
                        (cx + vv[0]*9 + vv[1]*7, cy + vv[1]*9 - vv[0]*7),
                        (cx + vv[0]*9 - vv[1]*7, cy + vv[1]*9 + vv[0]*7)], fill=col)
            txt((x0 + 5, y0 + 3), d, col, anchor="la")
            if sub:
                txt((x0 + C - 7, y0 + 3), "-", (255, 120, 120), anchor="ra")
        elif v == "white":
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2], fill=(242, 242, 242),
                         outline=(150, 150, 150))
        else:
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2], fill=(122, 122, 132))

dr.rectangle([PADL + 11*C - 6, PADT + 23*C - 6,
              PADL + 14*C + 2, PADT + 29*C + 2], outline=(120, 220, 255), width=4)
txt((PADL + 11*C - 4, PADT + 23*C - 24), "白环 P + 灰环 K（同侧）",
    (120, 220, 255), anchor="la")
for (px, pz) in ((11, 4), (11, 10), (11, 16), (11, 22)):
    dr.ellipse([PADL + px*C - 4, PADT + pz*C - 4,
                PADL + px*C + C - 2, PADT + pz*C + C - 2],
               outline=(255, 80, 80), width=3)

for x in range(Wg):
    txt((PADL + x*C + C//2, PADT - 24), str(x), (195, 195, 205))
for z in range(Lg):
    txt((8, PADT + z*C + C//2), f"z{z}", (195, 195, 205), anchor="lm")

info = [f"层1 通道 {J}  b=8 n=4", f"平面 {Wg}×{Lg}",
        "", "箭头 = 信号流向",
        "（= schem 的 facing 转 180°）",
        "", "红圈 = 白环线路上的 4 个减法",
        "         z=4,10,16,22 @ x=11",
        "", "z=28 的 N↑ = 品红环输入通道",
        "", "灰环已搬到白环同侧",
        "左上 3 列空出 → W 可降到 12"]
for t, s in enumerate(info):
    txt((PADL + Wg*C + 14, PADT + t*20), s, (228, 228, 238), anchor="la")

out = os.path.join(RESULTS, "plane_full_v2.png")
img.save(out)
print(f"wrote {out} {os.path.getsize(out)//1024} KB")
cmp_n = sum(1 for r in G for v in r if isinstance(v, tuple) and v[0] == "cmp")
sub_n = sum(1 for r in G for v in r if isinstance(v, tuple) and v[0] == "sub")
bar_n = sum(1 for r in G for v in r if isinstance(v, tuple) and v[0] == "bar")
print(f"compare {cmp_n}  subtract {sub_n}  containers {bar_n}")
print("V2_DONE")
