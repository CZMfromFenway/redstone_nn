#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Can the grayscale model output be snapped onto real grey Minecraft colours?

The observation: the HIGH column contains roughly ten near-zero-saturation colours, which could
serve as the grey levels for the w64 grayscale model. Then the grayscale output becomes an index
into a grey table, exactly like the colour output -- same mechanism, same buildability.

WHAT IS MEASURED
    the greys       every near-grey colour in the HIGH column, with luminance and saturation
    the ladder      how evenly those luminances cover 0..15, and where the gaps are
    the cost        w64's continuous output snapped onto the ladder, versus its 16 free levels
    the picture     because a level error of 0.5 reads very differently depending on where it is
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


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


mc = json.load(open(os.path.join(ROOT, "data", "mc_map_colors.json"), encoding="utf-8"))
MOD = mc["_modifiers"]
BASE = np.array([b["rgb"] for b in mc["bases"]], dtype=np.float64)
NAMES = [b["name"] for b in mc["bases"]]
HIGH = np.clip(BASE * MOD["HIGH"], 0, 255)

sat = (HIGH.max(1) - HIGH.min(1)) / np.maximum(HIGH.max(1), 1e-6)
lum = HIGH @ LUM

print("=" * 100)
print("NEAR-GREY COLOURS IN THE HIGH COLUMN")
print("=" * 100)
print(f"  {'地图色':22s} {'RGB':>18s} {'亮度(0..15)':>11s} {'饱和':>7s}")
print("  " + "-" * 64)
GREYS = []
for i in np.argsort(lum):
    if sat[i] < 0.12:
        GREYS.append(i)
        print(f"  {NAMES[i]:22s} ({HIGH[i][0]:5.0f},{HIGH[i][1]:5.0f},{HIGH[i][2]:5.0f}) "
              f"{lum[i]/17:11.2f} {sat[i]:7.3f}")
L = np.array([lum[i] for i in GREYS]) / 17.0
L.sort()
print(f"\n  共 {len(L)} 个近灰颜色（饱和 < 0.12），亮度："
      + ", ".join(f"{v:.2f}" for v in L))
gaps = np.diff(L)
print(f"  相邻间隔：" + ", ".join(f"{v:.2f}" for v in gaps))
print(f"  最大间隔 {gaps.max():.2f} 级（出现在 {L[gaps.argmax()]:.1f} 之后）")
print(f"  覆盖范围 {L.min():.2f} .. {L.max():.2f}")
print()
print("  理想情况是 8~12 个等距档位覆盖 0..15；实际分布：")
for a, b in zip(L[:-1], L[1:]):
    if b - a > 1.5:
        print(f"    空档 {a:.1f} → {b:.1f}（缺 {b-a:.1f} 级）")

# fill the gaps with a second shade column? show what that would add
print()
print("  如果允许混用同一基色的不同修饰档位（例如 STONE 的 LOW/NORMAL/HIGH/LOWEST），")
print("  可以补上这些空档，但就不再是单列了。下表显示灰色基色四档能提供的档位：")
GREYBASES = [i for i in range(len(BASE)) if sat[i] < 0.12]
allsh = []
for i in GREYBASES:
    for s, f in (("LOWEST", MOD["LOWEST"]), ("LOW", MOD["LOW"]),
                 ("NORMAL", MOD["NORMAL"]), ("HIGH", MOD["HIGH"])):
        v = np.clip(BASE[i] * f, 0, 255)
        allsh.append((float((v @ LUM) / 17), f"{NAMES[i]}/{s}"))
allsh.sort()
vals = np.array([v for v, _ in allsh])
uniq = np.unique(np.round(vals, 2))
print(f"    四档合计可提供 {len(uniq)} 个不同亮度档位："
      + ", ".join(f"{v:.2f}" for v in uniq))
g2 = np.diff(uniq)
print(f"    最大间隔 {g2.max():.2f} 级 —— 比单列的 {gaps.max():.2f} 级均匀得多")

# ---- the cost: snap w64's output onto the single-column ladder
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
free = np.rint(cont).astype(int)                 # 16 free levels, current behaviour

print()
print("=" * 100)
print("W64 输出吸附到单列灰阶 vs 现有的 16 级")
print("=" * 100)
print(f"  {'方案':26s} {'档位数':>7s} {'平均档位误差':>13s} {'最大误差':>9s} {'级数/张':>9s}")
print("  " + "-" * 70)


def levels_of(a):
    return np.array([len(set(row.tolist())) for row in a])


print(f"  {'现有：16 级自由取值':26s} {16:7d} {0.0:13.3f} {0.0:9.2f} "
      f"{levels_of(free).mean():9.1f}")
res = {"free": {"err": 0.0, "levels": float(levels_of(free).mean())}}
for name, ladder in (("单列 HIGH 灰阶", L),):
    snap = ladder[np.abs(cont[:, :, None] - ladder[None, None, :]).argmin(-1)]
    err = np.abs(snap - cont)
    lv = np.array([len(set(np.round(r, 2).tolist())) for r in snap])
    res[name] = {"err": float(err.mean()), "max": float(err.max()),
                 "levels": float(lv.mean()), "ladder": ladder.tolist()}
    print(f"  {name:26s} {len(ladder):7d} {err.mean():13.3f} {err.max():9.2f} "
          f"{lv.mean():9.1f}")

print()
print("  档位误差 0.3 级左右意味着：原本连续的灰被压到几个固定值，")
print("  同色区域仍然同色（形状不变），但渐变会变成台阶。")

# ---- picture
COLS, TILE = 16, 56
rows = [("现有 16 级", free),
        (f"吸附到 {len(L)} 个 HIGH 灰阶", ladder_used := None)]
snap = L[np.abs(cont[:, :, None] - L[None, None, :]).argmin(-1)]
rows[1] = (f"吸附到 {len(L)} 个 HIGH 灰阶（{res['单列 HIGH 灰阶']['err']:.2f} 级误差）", snap)

W = 330 + COLS * (TILE + 5) + 5
H = 92 + len(rows) * (TILE + 26) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "灰度头像吸附到真实 MC 灰阶", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), f"HIGH 列里有 {len(L)} 个近灰颜色，亮度 {L.min():.1f}~{L.max():.1f}。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "同色区域依然同色，形状不变；代价是渐变变成台阶。", font=f_s,
        fill=(130, 142, 158))
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
p = os.path.join(OUT, "GREY_LADDER.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
json.dump({"greys": {NAMES[i]: float(lum[i] / 17) for i in GREYS},
           "ladder": L.tolist(),
           "four_shade_uniq": [float(v) for v in uniq],
           "cost": res},
          open(os.path.join(OUT, "grey_ladder.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'grey_ladder.json')}")
print("GREY_LADDER_DONE")
