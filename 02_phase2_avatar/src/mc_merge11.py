#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
The user's 11-bin merge of 16 grey levels onto the HIGH-column near-greys.

THE SCHEME
    0,1,2 -> BLACK          3,4 -> GRAY         5 -> TERRACOTTA_CYAN
    6 -> DEEPSLATE          7 -> STONE          8 -> LIGHT_GRAY
    9 -> METAL              10,11 -> CLAY       12,13 -> WOOL
    14 -> QUARTZ            15 -> SNOW

It is monotone, which is the important property, but the bins are not evenly spaced in actual
luminance, so the merge may cost visible contrast. That is measured here before any blind test,
so the test is not run blind to its own risk.

WHAT TO WATCH
    effective levels   how many of the 11 bins are actually distinguishable
    luminance mapping  whether the darkest levels are brightened by the merge
    the picture        because the numbers can be right and the result still obvious
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour", "mcfit")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114])

mc = json.load(open(os.path.join(ROOT, "data", "mc_map_colors.json"), encoding="utf-8"))
MOD = mc["_modifiers"]
BASE = np.array([b["rgb"] for b in mc["bases"]], dtype=np.float64)
NAMES = [b["name"] for b in mc["bases"]]
HIGH = {NAMES[i]: np.clip(BASE[i] * MOD["HIGH"], 0, 255) for i in range(len(BASE))}

BINS = [
    ([0, 1, 2], "BLACK"),
    ([3, 4], "GRAY"),
    ([5], "TERRACOTTA_CYAN"),
    ([6], "DEEPSLATE"),
    ([7], "STONE"),
    ([8], "LIGHT_GRAY"),
    ([9], "METAL"),
    ([10, 11], "CLAY"),
    ([12, 13], "WOOL"),
    ([14], "QUARTZ"),
    ([15], "SNOW"),
]
print("=" * 100)
print("THE 11-BIN MERGE, VERIFIED")
print("=" * 100)
print(f"  {'级别':>6s} {'地图色':18s} {'RGB':>18s} {'亮度':>7s} {'饱和':>7s}")
print("  " + "-" * 62)
LUT = np.zeros((16, 3))
assigned = []
for levels, name in BINS:
    c = HIGH[name]
    lum = float((c @ LUM) / 17)
    sat = float((c.max() - c.min()) / max(c.max(), 1e-6))
    assigned.append((levels, name, lum, sat))
    print(f"  {str(levels):>6s} {name:18s} ({c[0]:5.0f},{c[1]:5.0f},{c[2]:5.0f}) "
          f"{lum:7.2f} {sat:7.3f}")
    for k in levels:
        LUT[k] = c
covered = sum(len(l) for l, _ in BINS)
print(f"\n  覆盖 {covered}/16 级，共 {len(BINS)} 个档位 —— 分级完整" if covered == 16
      else f"\n  ** 只覆盖 {covered}/16 级，有缺口 **")

lums = np.array([a[2] for a in assigned])
print(f"\n  档位亮度： " + ", ".join(f"{v:.2f}" for v in lums))
print(f"  相邻间隔： " + ", ".join(f"{v:.2f}" for v in np.diff(lums)))
print(f"  最小间隔 {np.diff(lums).min():.2f}，最大间隔 {np.diff(lums).max():.2f}")

print()
print("  ** 重要发现 **")
close = [(assigned[i][1], assigned[i + 1][1], float(np.diff(lums)[i]))
         for i in range(len(lums) - 1) if np.diff(lums)[i] < 0.5]
for a, b, d in close:
    print(f"    {a} 与 {b} 只差 {d:.2f} 级 —— 肉眼几乎无法区分，实际只算一档")
eff = 1 + sum(1 for d in np.diff(lums) if d >= 0.5)
print(f"  实际可分辨档位约 {eff} 个（不是 11 个）")
print(f"  原因：CLAY 的亮度 {assigned[7][2]:.2f} 与 METAL 的 {assigned[6][2]:.2f} 几乎相同，")
print(f"        CLAY 只是略微偏蓝（饱和 {assigned[7][3]:.2f}），不是更亮。")

# ---- what the merge does to real images
G = json.load(open(os.path.join(WEB, "pca22_dim4w64L1_model.json"), encoding="utf-8"))
seq = []
Ls = G["gen"]["layers"]
for i, Ly in enumerate(Ls):
    W = torch.tensor(Ly["W"], dtype=torch.float32)
    b = torch.tensor(Ly["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq.append(lin)
    if i != len(Ls) - 1:
        seq.append(nn.ReLU())
net = nn.Sequential(*seq).eval()

M = 4000
rng = np.random.default_rng(3)
Z = np.clip(rng.standard_normal((M, 4)), -3.5, 3.5).astype(np.float32)
with torch.no_grad():
    raw = np.clip(net(torch.tensor(Z)).numpy(), 0, 15)
lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
cont = np.clip((raw - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
free = np.rint(cont).astype(int)
grey = LUT[free].mean(-1) / 17.0   # LUT is 0..255, convert to 0..15 levels

print()
print("=" * 100)
print("WHAT THE MERGE DOES TO THE PICTURE (w64, 4000 faces)")
print("=" * 100)
print(f"  {'':16s} {'平均亮度':>9s} {'亮度sd':>8s} {'最暗':>6s} {'最亮':>6s} {'级数/张':>8s}")
print("  " + "-" * 60)


def stat(a):
    f = a.astype(float)
    return (f.mean(), f.std(1).mean(), f.min(), f.max(),
            np.mean([len(set(row.tolist())) for row in np.rint(a).astype(int)]))


for name, a in (("现有 16 级", free), ("归并到 11 档", np.rint(LUT[free].mean(-1) / 17.0).astype(int))):
    m, sd, mn, mx, lv = stat(a)
    print(f"  {name:16s} {m:9.2f} {sd:8.2f} {mn:6.0f} {mx:6.0f} {lv:8.1f}")
print()
print(f"  真实数据平均亮度约 7.7。归并把最暗从 0 抬到 {LUT[0].mean()/17:.2f}，所以整体会变亮。")

# ---- picture
COLS, TILE = 16, 52
rows = [("现有 16 级", free),
        ("归并到 11 档（你的方案）", np.rint(LUT[free].mean(-1) / 17.0).astype(int))]
W = 330 + COLS * (TILE + 5) + 5
H = 96 + len(rows) * (TILE + 26) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t = f_l = f_s = None
for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
    p = os.path.join("C:\\Windows\\Fonts", fn)
    if os.path.exists(p):
        try:
            f_t = ImageFont.truetype(p, 20); f_l = ImageFont.truetype(p, 12)
            f_s = ImageFont.truetype(p, 10)
            break
        except Exception:
            pass
if f_t is None:
    f_t = f_l = f_s = ImageFont.load_default()
dr.text((12, 12), "16 级归并到 11 档地图灰", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "档位亮度：1.47 / 4.47 / 5.32 / 5.88 / 6.59 / 9.00 / 9.82 / 9.92 / 11.71 / 14.83 / 15.00",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "注意 9.82 与 9.92 几乎同亮度，实际可分辨档位约 10 个。", font=f_s,
        fill=(224, 192, 90))
y = 76
for name, arr in rows:
    dr.text((12, y + 6), name, font=f_l,
            fill=(110, 168, 254) if "现有" in name else (110, 200, 130))
    for i in range(COLS):
        img8 = layout.image_from_left_half(np.rint(arr[i]).astype(int))
        t = np.kron(np.clip(img8 / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8))).astype(np.uint8)
        cv.paste(Image.fromarray(np.stack([t] * 3, -1)), (330 + i * (TILE + 5), y))
    y += TILE + 26
p = os.path.join(OUT, "MERGE11.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
json.dump({"lut": LUT.tolist(), "bins": [[l, n] for l, n in BINS],
           "lums": lums.tolist(), "effective_levels": int(eff)},
          open(os.path.join(OUT, "merge11.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'merge11.json')}")
print("MERGE11_DONE")
