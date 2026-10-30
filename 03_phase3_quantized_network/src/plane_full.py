#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
层1 单平面的逐块完整图（n=4, b=8）。
主体严格照 rsnn.schem 的形状，MSB 侧按 renn_sr.schem 的配对环展开。
每个比较器都画出朝向与模式；灰环 K 与白环 P 完整画出。
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
B1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
DELTA, XLO, WREF, B, N = 7.0, -3.5, 882.0, 8, 4

alpha = WREF / (np.abs(W1).max(1) * DELTA / 4)
U = np.rint(alpha[:, None, None] * W1[:, :, None] * (DELTA / (1 << B))
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)

J = 8                                    # 选一个两对都是情形Ⅰ的通道
S = (15 + np.abs(U[J])).astype(int)      # (4,8)
msb = np.abs(U[J, :, B - 1])
print("通道", J, "  MSB:", msb.tolist())

PAIRS = [(0, 1), (2, 3)]
CASE = []
for a, b in PAIRS:
    lo = min(msb[a], msb[b])
    hi = max(msb[a], msb[b])
    CASE.append("I" if lo > 882 else ("II" if hi > 882 else "III"))
print("配对情形:", CASE)

# ---------------- 版面
Wg, Lg = 14, 32
G = [[None] * Wg for _ in range(Lg)]


def put(x, z, v):
    if 0 <= x < Wg and 0 <= z < Lg:
        G[z][x] = v


ODD, EVEN = [3, 5, 7, 9], [4, 6, 8, 10]

# --- 灰环 K（左上 3×3）：(1,0)E (2,1)S (1,2)W (0,1)N
put(1, 0, ("cmp", "E")); put(2, 1, ("cmp", "S"))
put(1, 2, ("cmp", "W")); put(0, 1, ("cmp", "N"))
for (x, z) in ((0, 0), (2, 0), (0, 2), (2, 2)):
    put(x, z, "base")

# --- 彩色行 + 木桶（位 0..6）
for k in range(B - 1):
    col = ODD if k % 2 == 0 else EVEN
    zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
    zb = 3 * k + 1 if k % 2 == 0 else 3 * k + 4
    for i in range(N):
        put(col[i], zc, ("col", k))
        put(col[i], zb, ("bar", k, i, int(S[i, k])))

# --- 四个周期：cmp 行 + 减法链
for j in range(4):
    z3, z4, z5 = 6 * j + 3, 6 * j + 4, 6 * j + 5
    put(2, z3, ("cmp", "N"))
    for x in ODD:
        put(x, z3, ("cmp", "N"))
    for x in ODD + EVEN:
        put(x, z4, ("sub", "W"))
    put(2, z5, ("cmp", "N"))
    for x in EVEN:
        put(x, z5, ("cmp", "S"))

# --- 左总线 x=2
for z in range(1, 25):
    if G[z][2] is not None:
        continue
    put(2, z, ("cmp", "S") if z == 1 else (("cmp", "N") if z % 2 else "base"))

# --- 右总线 x=11
for z in range(4, 23):
    put(11, z, ("cmp", "S") if z % 2 == 0 else "white")

# --- MSB 行 z=24
for i in range(N):
    put(EVEN[i], 24, ("col", B - 1))

# --- 白环 P（右下 3×3）：(12,23)E (13,24)S (12,25)W (11,24)N
put(12, 23, ("cmp", "E")); put(13, 24, ("cmp", "S"))
put(12, 25, ("cmp", "W")); put(11, 24, ("cmp", "N"))
for (x, z) in ((11, 23), (13, 23), (11, 25), (13, 25)):
    put(x, z, "white")

# --- MSB 配对区 z=25..31
for pi, (a, b) in enumerate(PAIRS):
    xa, xb = EVEN[a], EVEN[b]
    case = CASE[pi]
    if case == "III":
        put(xa, 25, ("bar", B - 1, a, int(S[a, B - 1])))
        put(xb, 25, ("bar", B - 1, b, int(S[b, B - 1])))
        continue
    hi_i = a if S[a, B - 1] >= S[b, B - 1] else b
    lo_i = b if hi_i == a else a
    put(xa, 25, ("cmp", "S")); put(xb, 25, ("cmp", "S"))
    put(xa + 1, 25, ("bar", "diff", pi, int(S[hi_i, B - 1] - S[lo_i, B - 1])))
    put(xa, 26, ("cmp", "S")); put(xb, 26, ("cmp", "S"))
    if case == "I":
        put(xa + 1, 26, ("cmp", "N"))
        put(xa, 27, ("ring", B - 1)); put(xb, 27, ("ring", B - 1))
        put(xa + 1, 27, ("sub", "W" if a % 2 == 0 else "E"))
    put(xa, 29, ("ring", B - 1)); put(xb, 29, ("ring", B - 1))
    put(xa + 1, 29, ("cmp", "E"))
    put(xa, 30, ("cmp", "N")); put(xb, 30, ("cmp", "S"))
    put(xa, 31, ("ring", B - 1)); put(xb, 31, ("ring", B - 1))
    put(xa + 1, 31, ("cmp", "W"))
    if case == "II":
        put(xb, 25, ("bar", B - 1, b, int(S[b, B - 1])))

# ---------------- 画
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
C = 44
PADL, PADT, LEG = 74, 74, 300
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
            txt((cx, cy - 6), f"{val}", (10, 10, 10))
            txt((cx, cy + 8), ("Δ" if kind == "diff" else f"i{i}"),
                (10, 10, 10))
        elif isinstance(v, tuple) and v[0] in ("col", "ring"):
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2], fill=PAL[v[1] % 8],
                         outline=(80, 80, 80))
            if v[0] == "col":
                txt((cx, cy), "in", (20, 20, 20))
        elif isinstance(v, tuple) and v[0] in ("cmp", "sub"):
            d = v[1]
            sub = v[0] == "sub"
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2],
                         fill=(62, 52, 68) if sub else (52, 52, 62),
                         outline=(190, 105, 105) if sub else (115, 115, 130), width=2)
            vv = {"E": (1, 0), "W": (-1, 0), "N": (0, -1), "S": (0, 1)}[d]
            col = (255, 105, 105) if sub else (255, 200, 90)
            dr.line([cx - vv[0]*12, cy - vv[1]*12, cx + vv[0]*12, cy + vv[1]*12],
                    fill=col, width=3)
            dr.polygon([(cx + vv[0]*15, cy + vv[1]*15),
                        (cx + vv[0]*9 + vv[1]*6, cy + vv[1]*9 - vv[0]*6),
                        (cx + vv[0]*9 - vv[1]*6, cy + vv[1]*9 + vv[0]*6)], fill=col)
            txt((x0 + 6, y0 + 5), ("S" if sub else "") + d, col, anchor="la")
        elif v == "white":
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2], fill=(242, 242, 242),
                         outline=(150, 150, 150))
        else:
            dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2], fill=(122, 122, 132))

for (xa, za, lab, col) in ((0, 0, "灰环 K", (90, 255, 160)),
                           (11, 23, "白环 P", (120, 220, 255))):
    dr.rectangle([PADL + xa*C - 6, PADT + za*C - 6,
                  PADL + (xa+3)*C + 2, PADT + (za+3)*C + 2], outline=col, width=4)
    txt((PADL + xa*C - 2, PADT + za*C - 22), lab, col, anchor="la")

for x in range(Wg):
    txt((PADL + x*C + C//2, PADT - 22), str(x), (195, 195, 205))
for z in range(Lg):
    txt((8, PADT + z*C + C//2), f"z{z}", (195, 195, 205), anchor="lm")

info = [f"层1 通道 {J}   b=8  n=4",
        f"平面 {Wg}×{Lg} = {Wg*Lg}",
        "",
        "彩色 = 输入线（位标记）",
        "木桶 = 容器，数字 = S=15+|W|",
        "   i0..i3 = 输入号",
        "   Δ = 差值桶",
        "",
        "灰框 = 比较器，字母 = 朝向",
        "   前缀 S = subtract",
        "",
        "灰环 K / 白环 P 完整画出",
        "",
        f"配对情形: {CASE[0]} / {CASE[1]}"]
for t, s in enumerate(info):
    txt((PADL + Wg*C + 14, PADT + t*20), s, (228, 228, 238), anchor="la")

out = os.path.join(RESULTS, "plane_full.png")
img.save(out)
print(f"wrote {out} {os.path.getsize(out)//1024} KB")
print()
print("S 值表（输入 × 位）:")
for i in range(N):
    print(f"  i{i}: " + " ".join(f"{S[i,k]:5d}" for k in range(B)))
print()
print("非空块数:", sum(1 for r in G for v in r if v is not None))
cmp_n = sum(1 for r in G for v in r if isinstance(v, tuple) and v[0] == "cmp")
sub_n = sum(1 for r in G for v in r if isinstance(v, tuple) and v[0] == "sub")
bar_n = sum(1 for r in G for v in r if isinstance(v, tuple) and v[0] == "bar")
print(f"  compare {cmp_n}   subtract {sub_n}   容器 {bar_n}")
print("FULL_DONE")
