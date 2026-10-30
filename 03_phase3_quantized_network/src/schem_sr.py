#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""通用 schem 读取 + 俯视图（任意路径）。"""
import gzip
import json
import os
import struct
import sys
from collections import Counter

from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")


class R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def u1(self):
        v = self.b[self.i]; self.i += 1; return v

    def i1(self):
        v = struct.unpack_from(">b", self.b, self.i)[0]; self.i += 1; return v

    def i2(self):
        v = struct.unpack_from(">h", self.b, self.i)[0]; self.i += 2; return v

    def i4(self):
        v = struct.unpack_from(">i", self.b, self.i)[0]; self.i += 4; return v

    def i8(self):
        v = struct.unpack_from(">q", self.b, self.i)[0]; self.i += 8; return v

    def f4(self):
        v = struct.unpack_from(">f", self.b, self.i)[0]; self.i += 4; return v

    def f8(self):
        v = struct.unpack_from(">d", self.b, self.i)[0]; self.i += 8; return v

    def s(self):
        n = struct.unpack_from(">H", self.b, self.i)[0]; self.i += 2
        v = self.b[self.i:self.i + n].decode("utf-8", "replace"); self.i += n
        return v


def payload(r, t):
    if t in (1, 2, 3, 4, 5, 6):
        return {1: r.i1, 2: r.i2, 3: r.i4, 4: r.i8, 5: r.f4, 6: r.f8}[t]()
    if t == 7:
        n = r.i4(); v = list(r.b[r.i:r.i + n]); r.i += n; return v
    if t == 8:
        return r.s()
    if t == 9:
        it = r.u1(); n = r.i4(); return [payload(r, it) for _ in range(n)]
    if t == 10:
        d = {}
        while True:
            tt = r.u1()
            if tt == 0:
                break
            nm = r.s(); d[nm] = payload(r, tt)
        return d
    if t == 11:
        n = r.i4(); v = list(struct.unpack_from(">" + "i" * n, r.b, r.i)); r.i += 4 * n; return v
    if t == 12:
        n = r.i4(); v = list(struct.unpack_from(">" + "q" * n, r.b, r.i)); r.i += 8 * n; return v
    raise ValueError(t)


def load(path):
    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw); r.u1(); r.s()
    s = payload(r, 10)["Schematic"]
    W, H, L = s["Width"], s["Height"], s["Length"]
    pal = s["Blocks"]["Palette"]
    data = s["Blocks"]["Data"]
    inv = {v: k for k, v in pal.items()}
    grid = [[inv[data[x + z * W + y * W * L]] for x in range(W)]
            for z in range(L) for y in range(H)]
    g2 = [[None] * W for _ in range(L)]
    for z in range(L):
        for x in range(W):
            g2[z][x] = inv[data[x + z * W]]
    return s, W, H, L, pal, inv, g2


P = r"D:\czm我的世界\.minecraft\versions\1.21-Fabric 0.15.11\config\worldedit\schematics\renn_sr.schem"
s, W, H, L, pal, inv, G = load(P)

print("=" * 104)
print("一、规模与清单")
print("=" * 104)
print(f"  {W} × {H} × {L}")
cnt = Counter(v for row in G for v in row)
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
    if k == "minecraft:air":
        continue
    print(f"    {v:4d}  {k}")

SHORT = {
    "minecraft:light_gray_concrete": "base",
    "minecraft:red_concrete": "RED",
    "minecraft:white_concrete": "WHITE",
    "minecraft:orange_concrete": "ORANGE",
    "minecraft:yellow_concrete": "YELLOW",
    "minecraft:lime_concrete": "LIME",
    "minecraft:cyan_concrete": "CYAN",
    "minecraft:blue_concrete": "BLUE",
    "minecraft:purple_concrete": "PURPLE",
    "minecraft:magenta_concrete": "MAGENTA",
}


def nm(v):
    if v.startswith("minecraft:comparator"):
        f = v.split("facing=")[1].split(",")[0]
        m = "SUB" if "subtract" in v else "cmp"
        return f"{m}:{f}"
    if v.startswith("minecraft:barrel"):
        return "BARREL"
    return SHORT.get(v, v)


print()
print("=" * 104)
print("二、逐行非空方块")
print("=" * 104)
for z, row in enumerate(G):
    items = [(x, nm(v)) for x, v in enumerate(row) if v != "minecraft:air"]
    if items:
        print(f"z={z:2d}: " + "  ".join(f"{x}:{n}" for x, n in items))

# 出图
COLMAP = {
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
C, PADL, PADT, LEG = 34, 60, 62, 300
img = Image.new("RGB", (PADL + W * C + LEG, PADT + L * C + 14), (22, 22, 28))
dr = ImageDraw.Draw(img)
for z in range(L):
    for x in range(W):
        v = G[z][x]
        if v == "minecraft:air":
            continue
        x0, y0 = PADL + x * C, PADT + z * C
        if v.startswith("minecraft:comparator"):
            fac = v.split("facing=")[1].split(",")[0]
            sub = "subtract" in v
            dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2],
                         fill=(56, 50, 62) if sub else (56, 56, 66),
                         outline=(160, 90, 90) if sub else (110, 110, 125))
            cx, cy = x0 + C // 2, y0 + C // 2
            vv = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}[fac]
            col = (255, 110, 110) if sub else (255, 205, 95)
            dr.line([cx - vv[0] * 9, cy - vv[1] * 9, cx + vv[0] * 9, cy + vv[1] * 9],
                    fill=col, width=3)
            if sub:
                dr.text((x0 + 3, y0 + 2), "S", col)
        elif v.startswith("minecraft:barrel"):
            dr.rectangle([x0 + 4, y0 + 4, x0 + C - 6, y0 + C - 6],
                         fill=(150, 110, 70), outline=(90, 60, 30), width=2)
        else:
            dr.rectangle([x0 + 1, y0 + 1, x0 + C - 2, y0 + C - 2],
                         fill=COLMAP.get(v, (100, 100, 100)))
for x in range(W):
    dr.text((PADL + x * C + C // 2, PADT - 30), str(x), (190, 190, 200), anchor="ma")
for z in range(L):
    dr.text((6, PADT + z * C + C // 2), f"z{z}", (190, 190, 200), anchor="lm")
leg = [("S+箭头", "减法比较器"), ("箭头", "比较器 compare"), ("木桶", "容器"),
       ("白", "走线"), ("灰", "基座")]
for i, (a, b2) in enumerate(leg):
    dr.text((PADL + W * C + 12, PADT + i * 22), f"{a}  {b2}", (225, 225, 235))
out = os.path.join(RESULTS, "renn_sr.png")
img.save(out)
print(f"\n  wrote {out}  ({os.path.getsize(out)//1024} KB)")
json.dump({"W": W, "H": H, "L": L,
           "grid": [[G[z][x] for x in range(W)] for z in range(L)]},
          open(os.path.join(RESULTS, "renn_sr_grid.json"), "w", encoding="utf-8"),
          indent=1)
print("  已写入 results/renn_sr_grid.json")
print("SR_DONE")
