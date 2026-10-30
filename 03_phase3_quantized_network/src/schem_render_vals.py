#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""把编译结果写进样机：每个木桶的信号值 + 两个环的常数，出图。"""
import json
import os
import sys

from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

grid = json.load(open(os.path.join(RESULTS, "schem_grid.json"), encoding="utf-8"))
vals = json.load(open(os.path.join(RESULTS, "schem_values.json"), encoding="utf-8"))
G, W, L = grid["grid"], grid["W"], grid["L"]
S = vals["S"]
U = vals["weights"]
K, P, J = vals["K"], vals["P"], vals["channel"]

# (i,k) -> (x, z)
CELL = {}
for i in range(4):
    for k in range(8):
        if k % 2 == 0:
            CELL[(i, k)] = (2 * i + 3, 1 + 6 * (k // 2))
        else:
            CELL[(i, k)] = (2 * i + 4, 7 + 6 * (k // 2))

POS2CELL = {v: kp for kp, v in CELL.items()}

BITCOL = {
    0: (200, 45, 45), 1: (235, 140, 35), 2: (240, 225, 55), 3: (105, 215, 55),
    4: (35, 170, 190), 5: (50, 70, 200), 6: (135, 50, 195), 7: (215, 80, 200),
}
C = 96
PADL, PADT = 104, 96
WID = PADL + W * C + 320
HEI = PADT + L * C + 30
img = Image.new("RGB", (WID, HEI), (22, 22, 28))
dr = ImageDraw.Draw(img)


def txt(xy, s, fill=(235, 235, 245), anchor=None):
    dr.text(xy, s, fill=fill, anchor=anchor)


for z in range(L):
    for x in range(W):
        nm = G[z][x]
        x0, y0 = PADL + x * C, PADT + z * C
        if nm == "minecraft:air":
            continue
        if nm.startswith("minecraft:comparator"):
            fac = nm.split("facing=")[1].split(",")[0]
            sub = "subtract" in nm
            dr.rectangle([x0 + 2, y0 + 2, x0 + C - 4, y0 + C - 4],
                         fill=(52, 52, 62), outline=(110, 110, 125))
            cx, cy = x0 + C // 2, y0 + C // 2
            v = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}[fac]
            col = (255, 110, 110) if sub else (255, 205, 95)
            dr.line([cx - v[0] * 22 + v[1] * 10, cy - v[1] * 22 + v[0] * 10,
                     cx + v[0] * 22, cy + v[1] * 22], fill=col, width=4)
            if sub:
                txt((x0 + 6, y0 + 6), "SUB", col)
        elif nm.startswith("minecraft:barrel"):
            kp = POS2CELL.get((x, z))
            if kp is None:
                dr.rectangle([x0 + 6, y0 + 6, x0 + C - 8, y0 + C - 8],
                             fill=(150, 110, 70))
                return_ = None
            else:
                i, k = kp
                s = S[i][k]
                w = U[i][k]
                dr.rectangle([x0 + 5, y0 + 5, x0 + C - 7, y0 + C - 7],
                             fill=BITCOL[k], outline=(20, 20, 20), width=3)
                txt((x0 + C // 2 - 3, y0 + 16), f"S={s}", (15, 15, 15), "ma")
                txt((x0 + C // 2 - 3, y0 + 34), f"k{k}", (15, 15, 15), "ma")
                txt((x0 + C // 2 - 3, y0 + 52), f"in{i}", (15, 15, 15), "ma")
                txt((x0 + C // 2 - 3, y0 + 70), f"W={w}", (15, 15, 15), "ma")
        elif nm == "minecraft:white_concrete":
            dr.rectangle([x0 + 2, y0 + 2, x0 + C - 4, y0 + C - 4],
                         fill=(240, 240, 240), outline=(150, 150, 150))
        else:
            dr.rectangle([x0 + 2, y0 + 2, x0 + C - 4, y0 + C - 4],
                         fill=(120, 120, 130))

# 两个环标注
for (xa, za, lbl, const, desc) in (
        (0, 0, "灰环", K, "K = 15·32 + Σ|W|"),
        (11, 23, "白环", P, "P = 预激活零点")):
    dr.rectangle([PADL + xa * C - 6, PADT + za * C - 6,
                  PADL + (xa + 3) * C + 2, PADT + (za + 3) * C + 2],
                 outline=(90, 255, 160), width=5)
    txt((PADL + xa * C - 2, PADT + za * C - 30), lbl, (90, 255, 160))
    txt((PADL + xa * C - 2, PADT + za * C - 14), f"{desc} = {const}", (90, 255, 160))

for x in range(W):
    txt((PADL + x * C + 40, PADT - 34), str(x), (200, 200, 210), "ma")
for z in range(L):
    txt((12, PADT + z * C + 42), f"z{z}", (200, 200, 210))

info = [
    f"通道 j = {J}  (b=8, W_ref=882)",
    "",
    "每个木桶显示：",
    "  S   = 15 + |W|  写入容器的信号",
    "  k   = 位号（颜色对应）",
    "  in  = 输入号",
    "  W   = 位权重（带符号）",
    "",
    "环：",
    f"  灰环 K = {K}",
    f"  白环 P = {P}",
    "",
    "终点：N = K − Σ出",
    "      U = N − P",
    "",
    "Σ|W| = " + str(sum(abs(w) for row in U for w in row)),
]
for t, line in enumerate(info):
    txt((PADL + W * C + 18, PADT + t * 22), line, (225, 225, 235))

out = os.path.join(RESULTS, "schem_values.png")
img.save(out)
print("wrote", out, os.path.getsize(out) // 1024, "KB")

# 校验：Σ出 的极值
tot = sum(15 * 32,)
print()
print("校验：")
print(f"  Σ出 最小 = 15×32 = 480  → N = {K-480} = Σ|W|")
print(f"  Σ出 最大 = 480 + Σ|W| = {480+sum(abs(w) for row in U for w in row)}  → N = 0")
print(f"  所以 N ∈ [0, {sum(abs(w) for row in U for w in row)}]")
print(f"  U = N - P ∈ [{-P}, {sum(abs(w) for row in U for w in row)-P}]")
