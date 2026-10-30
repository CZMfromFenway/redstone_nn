#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""renn_sr.schem 的环区放大图（z=23..31）。"""
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
g = json.load(open(os.path.join(RESULTS, "renn_sr_grid.json"), encoding="utf-8"))
G, W, L = g["grid"], g["W"], g["L"]

Z0, Z1, X0, X1 = 23, L, 3, W
C, PADL, PADT = 52, 76, 66
img = Image.new("RGB", (PADL + (X1 - X0) * C + 340, PADT + (Z1 - Z0) * C + 16),
                (22, 22, 28))
dr = ImageDraw.Draw(img)

COLM = {
    "minecraft:magenta_concrete": (215, 80, 200),
    "minecraft:white_concrete": (240, 240, 240),
    "minecraft:light_gray_concrete": (120, 120, 130),
}
for z in range(Z0, Z1):
    for x in range(X0, X1):
        v = G[z][x]
        if v == "minecraft:air":
            continue
        x0, y0 = PADL + (x - X0) * C, PADT + (z - Z0) * C
        if v.startswith("minecraft:comparator"):
            fac = v.split("facing=")[1].split(",")[0]
            sub = "subtract" in v
            dr.rectangle([x0 + 2, y0 + 2, x0 + C - 4, y0 + C - 4],
                         fill=(60, 52, 66) if sub else (56, 56, 66),
                         outline=(190, 110, 110) if sub else (120, 120, 135), width=2)
            cx, cy = x0 + C // 2, y0 + C // 2
            vv = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}[fac]
            col = (255, 110, 110) if sub else (255, 205, 95)
            dr.line([cx - vv[0] * 13, cy - vv[1] * 13, cx + vv[0] * 13,
                     cy + vv[1] * 13], fill=col, width=4)
            dr.text((x0 + 4, y0 + 3), ("S" if sub else "") + fac[0].upper(),
                    fill=col)
        elif v.startswith("minecraft:barrel"):
            dr.rectangle([x0 + 5, y0 + 5, x0 + C - 8, y0 + C - 8],
                         fill=(150, 110, 70), outline=(90, 60, 30), width=2)
            dr.text((x0 + C // 2, y0 + C // 2), "B", (255, 240, 210), anchor="mm")
        else:
            dr.rectangle([x0 + 2, y0 + 2, x0 + C - 4, y0 + C - 4],
                         fill=COLM.get(v, (100, 100, 100)))

for x in range(X0, X1):
    dr.text((PADL + (x - X0) * C + C // 2, PADT - 22), str(x),
            (200, 200, 210), anchor="ma")
for z in range(Z0, Z1):
    dr.text((8, PADT + (z - Z0) * C + C // 2), f"z{z}", (200, 200, 210), anchor="lm")

# 品红 = 最高位输入线 / 环内连线
info = [
    "环区放大 (z=23..31)",
    "",
    "品红 = bit7(MSB) 输入线",
    "白  = 总线",
    "灰  = 基座",
    "",
    "S+字母 = 减法比较器",
    "字母   = 比较器朝向",
    "B      = 木桶",
]
for t, s in enumerate(info):
    dr.text((PADL + (X1 - X0) * C + 14, PADT + t * 22), s, (228, 228, 238))

out = os.path.join(RESULTS, "renn_sr_zoom.png")
img.save(out)
print("wrote", out, os.path.getsize(out) // 1024, "KB")
print()
print("=" * 90)
print("环区逐格（z=23..31）")
print("=" * 90)
AB = {"north": "北", "south": "南", "east": "东", "west": "西"}


def nm(v):
    if v.startswith("minecraft:comparator"):
        f = v.split("facing=")[1].split(",")[0]
        return ("减" if "subtract" in v else "比") + AB[f]
    if v.startswith("minecraft:barrel"):
        return "桶"
    return {"minecraft:magenta_concrete": "品", "minecraft:white_concrete": "白",
            "minecraft:light_gray_concrete": "灰"}.get(v, "?")


hdr = "      " + "".join(f"{x%10}" for x in range(X0, X1))
print(hdr)
for z in range(Z0, Z1):
    line = ""
    for x in range(X0, X1):
        v = G[z][x]
        line += " " if v == "minecraft:air" else nm(v)[0]
    print(f"  z={z:2d}  {line}")
print()
print("  图例：品=品红 白=白 灰=灰 桶=木桶 比=compare 减=subtract   数字=朝向(南/北/东/西)")
print("DONE")
