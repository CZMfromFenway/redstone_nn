#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
修正版平面图生成器。

修正 1  白环下移：主结构 z=0..L0-1（L0=3b+2），白环占 z=L0..L0+2，总长 L0+3
修正 2  位的方向由奇偶决定：
          偶 b：红(顶) = bit0 = LSB，最低一行颜色 = bit(b-1) = MSB
          奇 b：红(顶) = bit(b-1) = MSB，最低一行 = bit0 = LSB
        （因为只有"开放端"能挂环：偶 b 的开放端在底部，奇 b 的开放端在顶部）

方向约定（按用户定义）
    主方向 = 减法比较器朝向（本图为西）
    次方向 = 主方向逆时针 90°（西 → 南）
    环的比较器可朝次方向或反次方向 → 生成选项 ORIENT
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

PALETTE = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
           (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200),
           (150, 220, 225), (245, 160, 200), (180, 240, 160), (250, 210, 120)]
BASE, WHITE, CMPF, SUBF = (120, 120, 130), (240, 240, 240), (56, 56, 66), (56, 50, 62)
AIR = None


def build(n, b, orient="S"):
    L0 = 3 * b + 2
    L = L0 + 3
    W = 2 * n + 6
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

    # 颜色行序号 r -> 位 k
    #   偶 b: k = r
    #   奇 b: k = b-1-r  （红=最大）
    def bit_of_row(r):
        return r if b % 2 == 0 else (b - 1 - r)

    for r in range(b):
        k = bit_of_row(r)
        col = ODD if r % 2 == 0 else EVEN
        zc = 3 * r + 2 if r % 2 == 0 else 3 * r + 3
        zb = 3 * r + 1 if r % 2 == 0 else 3 * r + 4
        for i in range(n):
            put(col[i], zc, ("colour", k))
            put(col[i], zb, ("barrel", k, i))

    J = (b + 1) // 2
    for j in range(J):
        z3, z4, z5 = 6 * j + 3, 6 * j + 4, 6 * j + 5
        if z3 < L0:
            put(2, z3, ("cmp", "N"))
            for x in ODD:
                put(x, z3, ("cmp", "N"))
        if z4 < L0:
            for x in ODD + EVEN:
                put(x, z4, ("sub", "W"))
        if z5 < L0:
            put(2, z5, ("cmp", "N"))
            for x in EVEN:
                put(x, z5, ("cmp", "S"))

    for z in range(1, L0):
        if g[z][2] is not AIR:
            continue
        put(2, z, ("cmp", "S" if z == 1 else "N") if z % 2 else "base")

    # 白色总线（到 L0+2）
    xw = 2 * n + 3
    for z in range(4, L0 + 2):
        if z % 2 == 0:
            put(xw, z, ("cmp", "S"))
        else:
            put(xw, z, "white")
    # 白环（下移到主结构之下）
    za = L0
    put(xw + 1, za + 0, ("cmp", "E"))
    put(xw + 2, za + 1, ("cmp", orient))
    put(xw + 1, za + 2, ("cmp", "W"))
    put(xw + 0, za + 1, ("cmp", "S" if orient == "N" else "N"))
    for (x, z) in ((xw, za + 0), (xw + 2, za + 0), (xw, za + 2), (xw + 2, za + 2)):
        put(x, z, "white")
    return g, W, L, L0


def draw(n, b, fn, orient="S"):
    g, W, L, L0 = build(n, b, orient)
    C, PADL, PADT, LEG = 24, 62, 78, 320
    img = Image.new("RGB", (PADL + W * C + LEG, PADT + L * C + 16), (22, 22, 28))
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
                             fill=PALETTE[k % 12], outline=(15, 15, 15))
                dr.text((x0 + C // 2, y0 + C // 2), f"{i}", (10, 10, 10), anchor="mm")
            elif isinstance(v, tuple) and v[0] == "colour":
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2],
                             fill=PALETTE[v[1] % 12], outline=(90, 90, 90))
            elif isinstance(v, tuple) and v[0] in ("cmp", "sub"):
                d = v[1]
                sub = v[0] == "sub"
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2],
                             fill=SUBF if sub else CMPF,
                             outline=(160, 90, 90) if sub else (110, 110, 125))
                cx, cy = x0 + C // 2, y0 + C // 2
                vv = {"E": (1, 0), "W": (-1, 0), "N": (0, -1), "S": (0, 1)}[d]
                col = (255, 110, 110) if sub else (255, 205, 95)
                dr.line([cx - vv[0] * 6, cy - vv[1] * 6, cx + vv[0] * 6,
                         cy + vv[1] * 6], fill=col, width=2)
            elif v == "white":
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2], fill=WHITE)
            else:
                dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2], fill=BASE)

    dr.rectangle([PADL - 5, PADT - 5, PADL + 3 * C + 2, PADT + 3 * C + 2],
                 outline=(90, 255, 160), width=2)
    dr.text((PADL, PADT - 24), "环A", (90, 255, 160))
    xw = 2 * n + 3
    dr.rectangle([PADL + xw * C - 5, PADT + L0 * C - 5,
                  PADL + (xw + 3) * C + 2, PADT + L * C],
                 outline=(90, 255, 160), width=2)
    dr.text((PADL + xw * C, PADT + L0 * C - 22), "环B（已下移）", (90, 255, 160))

    for x in range(W):
        dr.text((PADL + x * C + C // 2, PADT - 40), str(x), (190, 190, 200), anchor="ma")
    for z in range(0, L, 2):
        dr.text((8, PADT + z * C + C // 2), f"z{z}", (190, 190, 200), anchor="lm")

    msb = b - 1
    msb_row = 0 if b % 2 == 1 else (b - 1)
    info = [
        f"b = {b}   n = {n}   (b {'奇' if b % 2 else '偶'})",
        f"主结构 z=0..{L0-1}，白环 z={L0}..{L0+2}",
        f"平面 {W} × {L} = {W*L} 格",
        f"容器 n·b = {n*b}",
        "",
        f"MSB = bit{msb} 在 {'顶部红' if b % 2 else '底部'}",
        "  （只有开放端能挂环）",
        "",
        f"主方向 = 西（减法比较器）",
        f"次方向 = 南（逆时针90°）",
        f"环朝向选项 = {orient}",
    ]
    for t, line in enumerate(info):
        dr.text((PADL + W * C + 14, PADT + t * 20), line, (228, 228, 238))
    img.save(fn)
    print(f"  b={b} n={n}: {W}×{L}={W*L} 格, 主结构 L0={L0}, 白环 z={L0}..{L0+2}")
    return W, L


print("=" * 72)
print("修正版：白环下移 + 奇 b 的位序翻转")
print("=" * 72)
a = draw(8, 9, os.path.join(RESULTS, "fix_b9_n8.png"))
c = draw(16, 9, os.path.join(RESULTS, "fix_b9_n16.png"))
print()
print("对照（原样机的 b=8, n=4）")
d = draw(4, 8, os.path.join(RESULTS, "fix_b8_n4.png"))
print()
print("=" * 72)
print("尺寸汇总（含白环下移后的 +3）")
print("=" * 72)
print(f"  {'配置':>10s} {'W':>4s} {'L0':>4s} {'L':>4s} {'面积':>6s} {'容器':>5s} {'体积2格':>8s} {'MSB在':>8s}")
print("  " + "-" * 60)
for (n, b) in ((4, 8), (8, 9), (16, 9)):
    W = 2 * n + 6
    L0 = 3 * b + 2
    L = L0 + 3
    print(f"  {f'{n},{b}':>10s} {W:4d} {L0:4d} {L:4d} {W*L:6d} {n*b:5d} {2*W*L:8,} "
          f"{'底部' if b%2==0 else '顶部':>8s}")

v1 = 2 * (2*4+6) * (3*8+2+3) * 64
v2 = 2 * (2*64+6) * (3*10+2+3) * 32
print()
print("=" * 72)
print("本项目（含白环下移）")
print("=" * 72)
print(f"  层1 (n=4,b=8,m=64)   : {v1:,}")
print(f"  层2 (n=64,b=10,m=32) : {v2:,}")
print(f"  合计                 : {v1+v2:,}")
print(f"  与最早估算 297,984 比 : {(v1+v2)/297984:.2f}×")

json.dump({"shapes": {"4,8": [14, 29], "8,9": [22, 32], "16,9": [38, 32]},
           "v_layer1": v1, "v_layer2": v2, "total": v1 + v2},
          open(os.path.join(RESULTS, "plane_fix.json"), "w", encoding="utf-8"), indent=1)
print("\n  已写入 results/plane_fix.json")
print("FIX_DONE")
