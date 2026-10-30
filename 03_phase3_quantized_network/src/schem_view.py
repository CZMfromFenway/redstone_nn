#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
把 rsnn.schem 画成俯视图 + ASCII 图，并尝试解读信号流。

Sponge v3：索引 = x + z*Width + y*Width*Length
"""
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
P = os.path.join(ROOT, "data", "rsnn.schem")


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


raw = gzip.decompress(open(P, "rb").read())
r = R(raw); r.u1(); r.s()
sch = payload(r, 10)["Schematic"]
W, H, L = sch["Width"], sch["Height"], sch["Length"]
pal = sch["Blocks"]["Palette"]
data = sch["Blocks"]["Data"]
inv = {v: k for k, v in pal.items()}

print("=" * 100)
print("一、规模")
print("=" * 100)
print(f"  {W} × {H} × {L}  = {W*H*L} 格（高度只有 1 —— 完全平面）")
cnt = Counter(data)
print(f"  空气 {cnt.get(pal['minecraft:air'],0)} 格，非空 {W*H*L - cnt.get(pal['minecraft:air'],0)} 格")
print()
print("  非空格子按类型：")
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
    nm = inv[k]
    if nm == "minecraft:air":
        continue
    print(f"    {v:4d}  {nm}")

# ---------------- ASCII 俯视图
SYM = {
    "minecraft:light_gray_concrete": ".",
    "minecraft:air": " ",
    "minecraft:red_concrete": "R",
    "minecraft:white_concrete": "W",
    "minecraft:orange_concrete": "O",
    "minecraft:yellow_concrete": "Y",
    "minecraft:lime_concrete": "L",
    "minecraft:cyan_concrete": "C",
    "minecraft:blue_concrete": "U",
    "minecraft:purple_concrete": "P",
    "minecraft:magenta_concrete": "M",
}


def sym(nm):
    if nm.startswith("minecraft:comparator"):
        fac = nm.split("facing=")[1].split(",")[0]
        mod = "subtract" in nm
        arrow = {"east": ">", "west": "<", "north": "^", "south": "v"}[fac]
        return ("s" if mod else "") + arrow if mod else arrow
    if nm.startswith("minecraft:barrel"):
        return "B"
    return SYM.get(nm, "?")


print()
print("=" * 100)
print("二、俯视图（行 = z，列 = x；箭头 = 比较器输出方向；s=subtract 模式）")
print("=" * 100)
hdr = "      " + "".join(str(x % 10) for x in range(W))
print(hdr)
grid = []
for z in range(L):
    row = []
    for x in range(W):
        idx = x + z * W
        row.append(inv[data[idx]])
    grid.append(row)
    print(f"  z={z:2d}  " + "".join(sym(nm)[-1] if nm.startswith("minecraft:comparator")
                                  else sym(nm) for nm in row))

# ---------------- 图
CELL = 34
PAD = 46
img = Image.new("RGB", (PAD + W * CELL + 10, PAD + L * CELL + 10), (28, 28, 34))
dr = ImageDraw.Draw(img)
COLMAP = {
    "minecraft:light_gray_concrete": (170, 170, 170),
    "minecraft:air": (30, 30, 36),
    "minecraft:red_concrete": (170, 40, 40),
    "minecraft:white_concrete": (235, 235, 235),
    "minecraft:orange_concrete": (230, 140, 40),
    "minecraft:yellow_concrete": (240, 220, 60),
    "minecraft:lime_concrete": (110, 220, 60),
    "minecraft:cyan_concrete": (40, 170, 190),
    "minecraft:blue_concrete": (50, 70, 190),
    "minecraft:purple_concrete": (130, 50, 190),
    "minecraft:magenta_concrete": (210, 80, 200),
}
for z in range(L):
    for x in range(W):
        nm = grid[z][x]
        x0 = PAD + x * CELL
        y0 = PAD + z * CELL
        if nm.startswith("minecraft:comparator"):
            dr.rectangle([x0, y0, x0 + CELL - 2, y0 + CELL - 2], fill=(60, 60, 70),
                         outline=(120, 120, 130))
            fac = nm.split("facing=")[1].split(",")[0]
            sub = "subtract" in nm
            cx, cy = x0 + CELL // 2, y0 + CELL // 2
            d = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}[fac]
            dr.line([cx - d[0] * 9, cy - d[1] * 9, cx + d[0] * 9, cy + d[1] * 9],
                    fill=(255, 200, 90) if not sub else (255, 120, 120), width=3)
            if sub:
                dr.text((x0 + 3, y0 + 2), "s", fill=(255, 120, 120))
        elif nm.startswith("minecraft:barrel"):
            dr.rectangle([x0 + 3, y0 + 3, x0 + CELL - 6, y0 + CELL - 6],
                         fill=(150, 110, 70), outline=(90, 60, 30), width=2)
        else:
            dr.rectangle([x0, y0, x0 + CELL - 2, y0 + CELL - 2],
                         fill=COLMAP.get(nm, (100, 100, 100)))
    dr.text((6, PAD + z * CELL + 8), f"z{z}", fill=(200, 200, 210))
for x in range(W):
    dr.text((PAD + x * CELL + 10, PAD - 20), str(x), fill=(200, 200, 210))
out = os.path.join(RESULTS, "schem_top.png")
img.save(out)
print(f"\n  wrote {out}")

json.dump({"W": W, "H": H, "L": L,
           "grid": [[grid[z][x] for x in range(W)] for z in range(L)]},
          open(os.path.join(RESULTS, "schem_grid.json"), "w", encoding="utf-8"),
          indent=1)
print("  已写入 results/schem_grid.json")
print("SCHEM_DONE")
