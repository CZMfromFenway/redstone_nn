#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
If "acceptable vs 中/差" is not a direction, is it a RADIUS?

The one-vs-rest probes showed:
    类二次元    AUC 0.896    direction  PC1 -, PC2 -
    类三次元    AUC 0.890    direction  PC2 +
    类游戏      AUC 0.735    direction  PC1 +
    可接受/中差 AUC 0.543    no direction at all

So the three styles look like three rays in the (PC1, PC2) plane, while acceptability is not
a ray. A natural alternative is DISTANCE FROM THE MEAN FACE: a face whose knobs are far from
typical could read as "odd" regardless of which style ray it sits on. That is a scalar,
cheap to compute, and worth testing before concluding acceptability is unreachable.

The figure plots every label in (PC1, PC2), coloured by style, so the ray structure and the
position of 中/差 are visible directly.
"""
import json
import math
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "style5")
os.makedirs(OUT, exist_ok=True)
STYLE = {4: "类二次元", 3: "类游戏", 2: "类三次元", 1: "中", 0: "差"}
COL = {4: (143, 211, 255), 3: (192, 176, 255), 2: (95, 214, 138),
       1: (224, 192, 90), 0: (224, 122, 95)}


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float64)
g = np.array([r["grade"] for r in new])
acc = (g >= 2).astype(float)

print("=" * 100)
print("IS '中/差' A DISTANCE FROM THE MEAN FACE?  (scalar probes, no fitting)")
print("=" * 100)
cands = {
    "|PC1|": np.abs(X[:, 1]),
    "|PC2|": np.abs(X[:, 2]),
    "|PC0|": np.abs(X[:, 0]),
    "|PC3|": np.abs(X[:, 3]),
    "radius PC0-PC3": np.sqrt((X ** 2).sum(1)),
    "radius PC1-PC2": np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2),
    "max |PC|": np.abs(X).max(1),
}
print(f"  {'probe':18s} {'AUC vs 可接受':>14s}   (above 0.5 means large values lean 中/差)")
print("  " + "-" * 60)
res = {}
for name, v in cands.items():
    a = auc(v, 1 - acc)     # predicting 中/差
    res[name] = a
    print(f"  {name:18s} {a:14.3f}")

print()
print("  style is a direction, acceptability is not. Best scalar probe:")
best = max(res, key=lambda k: res[k])
print(f"    {best} at AUC {res[best]:.3f}")

# ---------------------------------------------------------------- scatter
W, H = 720, 640
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(18), font(12), font(10)
dr.text((12, 10), "每一张脸在 (PC1 发肤, PC2 眼型) 平面上的位置", font=f_t, fill=(234, 238, 244))
dr.text((12, 34), "三种风格像是三条射线；中/差 却散落各处 —— 所以风格是方向，可接受性不是。",
        font=f_s, fill=(150, 162, 178))
cx, cy, R = W // 2, 40 + (H - 90) // 2, min(W, H - 90) // 2 - 40
dr.rectangle([cx - R, cy - R, cx + R, cy + R], outline=(50, 56, 68))
dr.line([cx - R, cy, cx + R, cy], fill=(40, 46, 58))
dr.line([cx, cy - R, cx, cy + R], fill=(40, 46, 58))
SC = 2.0
for i, r in enumerate(new):
    x = X[i, 1]; y = X[i, 2]
    px = int(cx + x * R / SC); py = int(cy - y * R / SC)
    if not (cx - R <= px <= cx + R and cy - R <= py <= cy + R):
        continue
    col = COL[r["grade"]]
    rad = 5 if r["grade"] >= 2 else 7
    dr.ellipse([px - rad, py - rad, px + rad, py + rad], fill=col,
               outline=(20, 22, 26) if r["grade"] >= 2 else (255, 255, 255))
dr.text((cx + R - 40, cy + 4), "PC1 →", font=f_s, fill=(120, 132, 148))
dr.text((cx + 4, cy - R + 4), "↑ PC2", font=f_s, fill=(120, 132, 148))
yy = H - 44
for g5 in (4, 3, 2, 1, 0):
    n = int((g == g5).sum())
    x = 16 + (4 - g5) * 132
    dr.ellipse([x, yy - 4, x + 9, yy + 5], fill=COL[g5])
    dr.text((x + 15, yy - 6), f"{STYLE[g5]} {n}", font=f_s, fill=(200, 208, 220))
dr.text((12, H - 22), "白圈 = 中/差（要避免的）；实心 = 可接受。箭头为「类二次元」「类游戏」「类三次元」的线性方向。",
        font=f_s, fill=(140, 152, 168))
p = os.path.join(OUT, "STYLE_PLANE.png")
cv.save(p)
print(f"\nwrote {p}")

json.dump(res, open(os.path.join(OUT, "radial_probes.json"), "w"), indent=2)
print("RADIAL_DONE")
