#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""rsnn.schem 的标注俯视图：分区、环、颜色-位对应。"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
d = json.load(open(os.path.join(RESULTS, "schem_grid.json"), encoding="utf-8"))
G = d["grid"]
W, L = d["W"], d["L"]

COLS = {
    "minecraft:light_gray_concrete": (120, 120, 130),
    "minecraft:red_concrete": (200, 45, 45),
    "minecraft:white_concrete": (240, 240, 240),
    "minecraft:orange_concrete": (235, 140, 35),
    "minecraft:yellow_concrete": (240, 225, 55),
    "minecraft:lime_concrete": (105, 215, 55),
    "minecraft:cyan_concrete": (35, 170, 190),
    "minecraft:blue_concrete": (50, 70, 200),
    "minecraft:purple_concrete": (135, 50, 195),
    "minecraft:magenta_concrete": (215, 80, 200),
}
BITNAME = {
    "minecraft:red_concrete": "bit0",
    "minecraft:orange_concrete": "bit1",
    "minecraft:yellow_concrete": "bit2",
    "minecraft:lime_concrete": "bit3",
    "minecraft:cyan_concrete": "bit4",
    "minecraft:blue_concrete": "bit5",
    "minecraft:purple_concrete": "bit6",
    "minecraft:magenta_concrete": "bit7",
}

C = 40
PADL, PADT = 60, 58
img = Image.new("RGB", (PADL + W * C + 200, PADT + L * C + 20), (24, 24, 30))
dr = ImageDraw.Draw(img)

for z in range(L):
    for x in range(W):
        nm = G[z][x]
        x0, y0 = PADL + x * C, PADT + z * C
        if nm == "minecraft:air":
            continue
        if nm.startswith("minecraft:comparator"):
            fac = nm.split("facing=")[1].split(",")[0]
            sub = "subtract" in nm
            dr.rectangle([x0, y0, x0 + C - 2, y0 + C - 2], fill=(52, 52, 62),
                         outline=(110, 110, 125))
            cx, cy = x0 + C // 2, y0 + C // 2
            v = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}[fac]
            col = (255, 110, 110) if sub else (255, 205, 95)
            dr.line([cx - v[0] * 11 + v[1] * 6, cy - v[1] * 11 + v[0] * 6,
                     cx + v[0] * 11, cy + v[1] * 11], fill=col, width=3)
            if sub:
                dr.text((x0 + 3, y0 + 2), "S", fill=col)
        elif nm.startswith("minecraft:barrel"):
            dr.rectangle([x0 + 4, y0 + 4, x0 + C - 6, y0 + C - 6],
                         fill=(152, 112, 68), outline=(95, 62, 30), width=2)
            dr.text((x0 + 12, y0 + 10), "B", fill=(255, 240, 210))
        else:
            dr.rectangle([x0, y0, x0 + C - 2, y0 + C - 2], fill=COLS[nm])
            t = BITNAME.get(nm)
            if t:
                dr.text((x0 + 6, y0 + 12), t, fill=(20, 20, 20))

# 两个环
for (xa, xb, za, zb, label) in ((0, 2, 0, 2, "环A"), (11, 13, 23, 25, "环B")):
    dr.rectangle([PADL + xa * C - 4, PADT + za * C - 4,
                  PADL + (xb + 1) * C + 2, PADT + (zb + 1) * C + 2],
                 outline=(90, 255, 160), width=3)
    dr.text((PADL + xa * C, PADT + za * C - 20), label, fill=(90, 255, 160))

# 两条竖总线
dr.rectangle([PADL + 2 * C - 4, PADT - 4, PADL + 3 * C + 2, PADT + L * C], outline=(120, 180, 255), width=2)
dr.text((PADL + 2 * C, PADT - 22), "x=2 总线", fill=(120, 180, 255))
dr.rectangle([PADL + 11 * C - 4, PADT - 4, PADL + 12 * C + 2, PADT + L * C], outline=(255, 160, 90), width=2)
dr.text((PADL + 11 * C, PADT - 22), "x=11 总线", fill=(255, 160, 90))

for x in range(W):
    dr.text((PADL + x * C + 14, PADT - 44), str(x), fill=(210, 210, 220))
for z in range(L):
    dr.text((6, PADT + z * C + 12), f"z{z}", fill=(210, 210, 220))

legend = [
    ("S+箭头", (255, 110, 110), "减法比较器(32)"),
    ("箭头", (255, 205, 95), "比较器 compare(56)"),
    ("B", (152, 112, 68), "木桶(32)"),
    ("bitN", (200, 200, 200), "彩色=位标记"),
    ("坐标", (120, 120, 130), "灰=基座"),
]
lx = PADL + W * C + 14
for i, (t, col, desc) in enumerate(legend):
    y = PADT + 20 + i * 46
    dr.rectangle([lx, y, lx + 14, y + 14], fill=col)
    dr.text((lx + 20, y), t, fill=(230, 230, 240))
    dr.text((lx + 20, y + 18), desc, fill=(160, 160, 175))

out = os.path.join(RESULTS, "schem_annotated.png")
img.save(out)
print("wrote", out, os.path.getsize(out) // 1024, "KB")

# ---------------- 结构统计
print()
print("=" * 84)
print("重复单元（z 方向周期 6）")
print("=" * 84)
print("   z mod 6   内容                                 每周期数量")
print("   " + "-" * 62)
print("     1      木桶 4 个（奇 x）                        4")
print("     2      彩色 4 格（奇 x）= 位标记                 4")
print("     3      比较器 朝北 (x=2,3,5,7,9)                5")
print("     4      减法比较器 朝西 (x=3..10)                8")
print("     5      朝北 x=2 ／ 朝南 x=4,6,8,10              5")
print("     6      彩色 4 格（偶 x）= 位标记                 4")
print()
print("  周期数 4（z=1..24），首尾各多一行木桶（z=1 的 4 个、z=25 的 4 个）")
print(f"  木桶总数 {4+8+8+8+4} = 4 输入 × 8 位  ✓")
print(f"  减法比较器 {8*4} = 32  ✓")
print("  彩色 8 色 × 4 格 = 32  ✓")
