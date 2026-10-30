#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按 rsnn.schem 反推出的规则，生成任意 (n, b) 的平面图。

规律（从 b=8, n=4 的样机反推并验证）
    位 k 的彩色行   z = 3k+2（k 偶，奇数列）／ z = 3k+3（k 奇，偶数列）
    位 k 的木桶行   z = 3k+1（k 偶）        ／ z = 3k+4（k 奇）
    周期 j = 0..:   cmp 行 z=6j+3、SUB 行 z=6j+4、cmp 行 z=6j+5
    输入 i 两列     x = 2i+3（偶位）／ 2i+4（奇位）
    宽 W = 2n+6      长 L = 3b+2
"""
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

BITCOL = {
    0: (200, 45, 45), 1: (235, 140, 35), 2: (240, 225, 55), 3: (105, 215, 55),
    4: (35, 170, 190), 5: (50, 70, 200), 6: (135, 50, 195), 7: (215, 80, 200),
    8: (150, 220, 225), 9: (245, 160, 200),
}
BASE = (120, 120, 130)
WHITE = (240, 240, 240)
BARREL = (150, 110, 70)
CMP = (56, 56, 66)
SUB = (56, 50, 62)
AIR = None


def build(n, b):
    W, L = 2 * n + 6, 3 * b + 2
    g = [[AIR] * W for _ in range(L)]

    def put(x, z, v):
        if 0 <= x < W and 0 <= z < L:
            g[z][x] = v

    ODD = [2 * i + 3 for i in range(n)]
    EVEN = [2 * i + 4 for i in range(n)]

    # 环 A（左上 3×3）
    for (x, z, v) in ((1, 0, "E"), (2, 1, "S"), (1, 2, "W"), (0, 1, "N")):
        put(x, z, ("cmp", v))
    for (x, z) in ((0, 0), (2, 0), (0, 2), (2, 2)):
        put(x, z, "base")

    # 每位：彩色行 + 木桶
    cells = []
    for k in range(b):
        col = ODD if k % 2 == 0 else EVEN
        zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
        zb = 3 * k + 1 if k % 2 == 0 else 3 * k + 4
        for i in range(n):
            put(col[i], zc, ("colour", k))
            put(col[i], zb, ("barrel", k, i))
            cells.append((i, k, col[i], zb))

    # 每个周期：cmp 行 + SUB 行
    J = (b + 1) // 2
    for j in range(J):
        z3, z4, z5 = 6 * j + 3, 6 * j + 4, 6 * j + 5
        if z3 < L:
            put(2, z3, ("cmp", "N"))
            for x in ODD:
                put(x, z3, ("cmp", "N"))
        if z4 < L:
            for x in ODD + EVEN:
                put(x, z4, ("sub", "W"))
        if z5 < L:
            put(2, z5, ("cmp", "N"))
            for x in EVEN:
                put(x, z5, ("cmp", "S"))

    # 灰色总线 x=2
    for z in range(1, L):
        if g[z][2] is not AIR:
            continue
        if z % 2 == 1:
            put(2, z, ("cmp", "S" if z == 1 else "N"))
        else:
            put(2, z, "base")

    # 白色总线 x=2n+3
    xw = 2 * n + 3
    for z in range(4, L - 3):
        if z % 2 == 0:
            put(xw, z, ("cmp", "S"))
        else:
            put(xw, z, "white")
    put(xw, L - 2, ("cmp", "N"))
    put(xw, L - 1, "white")

    # 环 B（右下 3×3）
    put(xw + 1, L - 3, ("cmp", "E"))
    put(xw + 2, L - 2, ("cmp", "S"))
    put(xw + 1, L - 1, ("cmp", "W"))
    put(xw + 0, L - 2, ("cmp", "N"))
    for (x, z) in ((xw, L - 3), (xw + 2, L - 3), (xw, L - 1), (xw + 2, L - 1)):
        put(x, z, "white")
    return g, W, L, cells


def draw(n, b, fn):
    g, W, L, cells = build(n, b)
    C, PADL, PADT = 26, 62, 74
    LEG = 300
    img = Image.new("RGB", (PADL + W * C + LEG, PADT + L * C + 20), (22, 22, 28))
    dr = ImageDraw.Draw(img)

    for z in range(L):
        for x in range(W):
            v = g[z][x]
            if v is AIR:
                continue
            x0, y0 = PADL + x * C, PADT + z * C
            if isinstance(v, tuple) and v[0] == "barrel":
                k, i = v[1], v[2]
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2],
                             fill=BITCOL[k % len(BITCOL)], outline=(20, 20, 20))
                dr.text((x0 + C // 2, y0 + C // 2 - 4), f"{i}", (15, 15, 15), anchor="mm")
            elif isinstance(v, tuple) and v[0] == "colour":
                k = v[1]
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2],
                             fill=BITCOL[k % len(BITCOL)], outline=(90, 90, 90))
            elif isinstance(v, tuple) and v[0] in ("cmp", "sub"):
                d = v[1]
                is_sub = v[0] == "sub"
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2],
                             fill=SUB if is_sub else CMP,
                             outline=(150, 90, 90) if is_sub else (110, 110, 125))
                cx, cy = x0 + C // 2, y0 + C // 2
                vv = {"E": (1, 0), "W": (-1, 0), "N": (0, -1), "S": (0, 1)}[d]
                col = (255, 110, 110) if is_sub else (255, 205, 95)
                dr.line([cx - vv[0] * 6, cy - vv[1] * 6, cx + vv[0] * 6,
                         cy + vv[1] * 6], fill=col, width=2)
            elif v == "white":
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2], fill=WHITE)
            else:
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2], fill=BASE)

    dr.rectangle([PADL - 4, PADT - 4, PADL + 3 * C + 2, PADT + 3 * C + 2],
                 outline=(90, 255, 160), width=2)
    dr.text((PADL, PADT - 22), "环A", (90, 255, 160))
    xw = 2 * n + 3
    dr.rectangle([PADL + xw * C - 4, PADT + (L - 3) * C - 4,
                  PADL + (xw + 3) * C + 2, PADT + L * C],
                 outline=(90, 255, 160), width=2)
    dr.text((PADL + xw * C, PADT + (L - 3) * C - 22), "环B", (90, 255, 160))

    for x in range(W):
        dr.text((PADL + x * C + C // 2, PADT - 36), str(x), (190, 190, 200), anchor="ma")
    for z in range(0, L, 2):
        dr.text((8, PADT + z * C + C // 2), f"z{z}", (190, 190, 200), anchor="lm")

    info = [
        f"b = {b},  输入 n = {n}",
        f"平面 {W} × {L} = {W*L} 格",
        f"容器 n·b = {n*b} 个",
        f"减法比较器 n·b = {n*b} 个",
        f"周期数 ⌈b/2⌉ = {(b+1)//2}",
        "",
        f"高度 2 格/层 → 体积 {2*W*L}",
        f"（单输出平面）",
        "",
        "木桶里的数字 = 输入号",
        "颜色 = 位号",
    ]
    for t, line in enumerate(info):
        dr.text((PADL + W * C + 14, PADT + t * 20), line, (225, 225, 235))

    img.save(fn)
    print(f"  n={n}, b={b}:  {W}×{L} = {W*L} 格,  容器 {n*b},  "
          f"体积(2格高) {2*W*L:,}   -> {os.path.basename(fn)}")
    return W, L, n * b


print("=" * 70)
print("生成平面图：b=9，n=8 / n=16")
print("=" * 70)
r1 = draw(8, 9, os.path.join(RESULTS, "plane_b9_n8.png"))
r2 = draw(16, 9, os.path.join(RESULTS, "plane_b9_n16.png"))
print()
print("=" * 70)
print("对照：原样机 b=8, n=4")
print("=" * 70)
r0 = draw(4, 8, os.path.join(RESULTS, "plane_b8_n4.png"))
print()
print("=" * 70)
print("尺寸与体积")
print("=" * 70)
print(f"  {'配置':>12s} {'W':>4s} {'L':>4s} {'面积':>7s} {'容器':>6s} {'体积(2格高)':>11s}")
print("  " + "-" * 52)
for (n, b) in ((4, 8), (8, 9), (16, 9)):
    Wd, Ld = 2 * n + 6, 3 * b + 2
    print(f"  {f'n={n},b={b}':>12s} {Wd:4d} {Ld:4d} {Wd*Ld:7d} {n*b:6d} {2*Wd*Ld:11,}")

print()
print("=" * 70)
print("本项目两层（高度 2 格/层）")
print("=" * 70)
v1 = 2 * (2 * 4 + 6) * (3 * 8 + 2) * 64
v2 = 2 * (2 * 64 + 6) * (3 * 10 + 2) * 32
print(f"  层1 (n=4,b=8,m=64)   : {v1:,}")
print(f"  层2 (n=64,b=10,m=32) : {v2:,}")
print(f"  合计                 : {v1+v2:,}")
print(f"  之前按 1 格高估的      : 160,512")
print(f"  更早按 2m 估的         : 297,984")
print(f"  ⇒ 与最早估算 {(v1+v2)/297984:.2f}×，同量级 ✓")

json.dump({"shapes": {"4,8": [14, 26], "8,9": [22, 29], "16,9": [38, 29]},
           "volume_layer1": v1, "volume_layer2": v2, "total": v1 + v2},
          open(os.path.join(RESULTS, "plane_shapes.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/plane_shapes.json")
print("PLANE_DONE")
