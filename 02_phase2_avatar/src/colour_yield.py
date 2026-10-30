#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Why is colour yield lower than grayscale, and what separates the faces the user liked from the
ones they rejected?

THE INPUT COUNT
The grayscale generator has 4 independent inputs (4 PCA components of the 8x8 luminance image).
The colour generator has 8 (8 PCA components of the full 192-dim RGB image). That is the one
structural difference that is not about pixels, so it is checked directly: if the rejected faces
sit further from the origin in that 8-dimensional space, then random N(0,1) sampling is spending
effort on regions the model renders badly, and truncating the sampling radius would raise yield
for free.

WHAT IS COMPARED
    per feature   how well it separates 可接受 (grade>=2) from 不可接受
    the montage   the actual faces, best-rated and worst-rated, because numbers lie
"""
import json
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
OUT = os.path.join(ROOT, "results", "colour", "yield")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
os.makedirs(OUT, exist_ok=True)

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


def auc(s, t):
    hi = s[t == 1][:, None]
    lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
# every colour label that carries pixels: rounds 8 and 11 use the five-way style scheme
col = [r for r in recs
       if str(r.get("round") or "") in ("8", "11")
       and len(r.get("predicted_levels") or []) == 96
       and r.get("scheme") == "style"]
print(f"colour labels with pixels: {len(col)}")
print(f"  round 8: {sum(1 for r in col if str(r['round'])=='8')}, "
      f"round 11: {sum(1 for r in col if str(r['round'])=='11')}")

name = {4: "类二次元", 3: "类游戏", 2: "类三次元", 1: "中", 0: "差"}
for g in (4, 3, 2, 1, 0):
    print(f"  {name[g]:6s} {sum(1 for r in col if r['grade']==g):4d}")

FLAT = np.array([r["predicted_levels"] for r in col], dtype=np.float64)
G = np.array([r["grade"] for r in col])
Y = (G >= 2).astype(int)
Z8 = np.array([r["pca"][:8] for r in col], dtype=np.float64)
A = np.array([r["model"] for r in col])
print(f"\n  可接受 {Y.sum()} / {len(Y)} = {100*Y.mean():.1f}%")
print(f"  输入维度：彩色 8（前 8 个 PCA），灰度 4")


def to_img(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


IMG = to_img(FLAT)
PX = IMG.reshape(len(IMG), 64, 3)
A32 = FLAT.reshape(len(FLAT), 32, 3)

mx = PX.max(-1); mn = PX.min(-1)
SAT = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
D = np.linalg.norm(A32[:, :, None, :] - A32[:, None, :, :], axis=-1)
IU = np.triu_indices(32, 1)
DH = np.linalg.norm(IMG[:, :, 1:] - IMG[:, :, :-1], axis=-1)
DV = np.linalg.norm(IMG[:, 1:] - IMG[:, :-1], axis=-1)
ED = np.concatenate([DH.reshape(len(IMG), -1), DV.reshape(len(IMG), -1)], 1)

# area share of the most common colour: the user's 7:2:1 idea, as a single number
top_share = []
for i in range(len(A32)):
    keys = [tuple(np.rint(c).astype(int)) for c in A32[i]]
    vals, cnts = np.unique(keys, axis=0, return_counts=True)
    top_share.append(cnts.max() / 32)
top_share = np.array(top_share)
ncol = np.array([len({tuple(np.rint(c).astype(int)) for c in A32[i]}) for i in range(len(A32))])

FEATS = {
    "饱和均值": SAT.mean(1),
    "饱和标准差": SAT.std(1),
    "深色像素占比(饱和<0.15)": (SAT < 0.15).mean(1),
    "分离度": D[:, IU[0], IU[1]].mean(1),
    "边缘p90": np.percentile(ED, 90, axis=1),
    "边缘均值": ED.mean(1),
    "颜色数": ncol.astype(float),
    "主色占比": top_share,
    "亮度均值": (A32 @ LUM).mean(1),
    "亮度标准差": (A32 @ LUM).std(1),
    "**输入向量长度**": np.linalg.norm(Z8, axis=1),
    "输入最大绝对值": np.abs(Z8).max(1),
}

print()
print("=" * 100)
print("WHICH FEATURES SEPARATE 可接受 FROM 不可接受")
print("=" * 100)
print(f"  {'特征':26s} {'可接受均值':>10s} {'不可接受均值':>12s} {'AUC':>7s} {'强度':>6s}")
print("  " + "-" * 68)
rows = []
for k, v in FEATS.items():
    a = auc(v, Y)
    strength = abs(a - 0.5) * 2
    rows.append((k, v[Y == 1].mean(), v[Y == 0].mean(), a, strength))
for k, m1, m0, a, st in sorted(rows, key=lambda t: -t[4]):
    print(f"  {k:26s} {m1:10.3f} {m0:12.3f} {a:7.3f} {st:6.2f}")

print()
print("  AUC 0.5 = 无区分力；0.5±0.5 = 完美。强度 = |2(AUC-0.5)|。")

# ---- does the input radius matter?
print()
print("=" * 100)
print("DOES THE INPUT RADIUS MATTER?  (8 dimensions, random N(0,1) truncated at |z|<3.5)")
print("=" * 100)
rad = np.linalg.norm(Z8, axis=1)
bins = [(0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 9)]
print(f"  {'输入长度':>12s} {'张数':>6s} {'可接受率':>9s}")
for lo, hi in bins:
    m = (rad >= lo) & (rad < hi)
    if m.sum() < 8:
        continue
    print(f"  {lo:.1f}~{hi:.1f}".rjust(12) + f" {int(m.sum()):6d} {100*Y[m].mean():8.1f}%")
print()
print(f"  8 维 N(0,1) 的典型长度是 sqrt(8)={np.sqrt(8):.2f}，"
      f"这批样本的中位数是 {np.median(rad):.2f}")

# what would truncating at various radii keep?
print()
print(f"  {'半径上限':>10s} {'保留比例':>9s} {'留下的可接受率':>13s} {'总良率':>8s}")
for R in (1.8, 2.0, 2.2, 2.5, 2.8, 3.0):
    keep = rad <= R
    if keep.sum() < 10:
        continue
    print(f"  {R:10.1f} {100*keep.mean():8.1f}% {100*Y[keep].mean():12.1f}% "
          f"{100*Y[keep].mean():7.1f}%")
print("  （最后一列＝只保留这个半径内的样本时的良率；前两列说明牺牲多少多样性）")

# ---- montage
def montage(idx, title, path, cols=16):
    TILE, PAD = 52, 5
    n = len(idx)
    rowsn = (n + cols - 1) // cols
    W = 16 + cols * (TILE + PAD)
    H = 34 + rowsn * (TILE + PAD)
    cv = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    dr.text((12, 9), title, font=font(15), fill=(234, 238, 244))
    for j, i in enumerate(idx):
        r, c = divmod(j, cols)
        t = np.kron(np.clip(IMG[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (16 + c * (TILE + PAD), 34 + r * (TILE + PAD)))
    cv.save(path)
    return cv.size


good = np.where(G == 4)[0]
bad = np.where(G == 0)[0]
mid = np.where(G == 2)[0] + 0
print()
print("=" * 100)
print("MONTAGE")
print("=" * 100)
rng = np.random.default_rng(0)
sel_good = rng.choice(good, min(48, len(good)), replace=False) if len(good) else []
sel_bad = rng.choice(bad, min(48, len(bad)), replace=False) if len(bad) else []
print(f"  类二次元 {len(good)} 张，差 {len(bad)} 张 -> 随机各取 48 张出图")
if len(sel_good):
    print("  wrote", montage(sel_good, "打为「类二次元」（最好）的随机样本",
                             os.path.join(OUT, "GOOD.png")))
if len(sel_bad):
    print("  wrote", montage(sel_bad, "打为「差」（最差）的随机样本",
                             os.path.join(OUT, "BAD.png")))

np.save(os.path.join(OUT, "features.npy"),
        {"FLAT": FLAT, "G": G, "Z8": Z8,
         "feats": {k: v for k, v in FEATS.items()}}, allow_pickle=True)
print(f"\nwrote {os.path.join(OUT, 'features.npy')}")
print("COLOUR_YIELD_DONE")
