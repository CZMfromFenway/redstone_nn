#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Round 8 analysis, then a colour-aware filter trained on these very labels, then before/after.

PART 1  statistics
Wilson intervals and pairwise Fisher exact on "acceptable" (grade >= 2, i.e. any of the three
styles). Note the scheme here is the five-way style scheme, not the old 优/良/中/差, so the
counts are reported by name.

PART 2  a proper colour scorer
The luminance scorer reaches only AUC 0.616 on colour faces, because it cannot see the things
the user actually judges. These 260 labels are the first colour labels large enough to fit a
scorer on COLOUR features directly, so one is trained here and cross-validated.

PART 3  before / after
The best model is run twice: raw draws, and draws filtered by that scorer. The two sheets use
the same size and layout so they can be compared directly.
"""
import json
import math
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
OUT = os.path.join(ROOT, "results", "colour", "round8")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

STYLE = {4: "类二次元", 3: "类游戏", 2: "类三次元", 1: "中", 0: "差"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(a, b, c, d):
    from math import comb
    n = a + b + c + d
    def pr(x):
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)
    lo = max(0, a + c - (c + d)); hi = min(a + b, a + c)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12))


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r8 = [r for r in recs if str(r.get("round") or "") == "8"]
ARMS = ["pal8", "pal16", "pal32"]
print("=" * 100)
print(f"ROUND 8 -- {len(r8)} colour labels")
print("=" * 100)
print(f"  {'arm':7s} {'n':>4s} " + " ".join(f"{STYLE[g]:>7s}" for g in (4, 3, 2, 1, 0))
      + f" {'可接受':>7s} {'95% CI':>17s}")
print("  " + "-" * 78)
tab = {}
for a in ARMS:
    sub = [r for r in r8 if r.get("model") == a]
    n = len(sub)
    c = {g: sum(1 for r in sub if r["grade"] == g) for g in (4, 3, 2, 1, 0)}
    acc = c[4] + c[3] + c[2]
    lo, hi = wilson(acc, n)
    tab[a] = {"n": n, "counts": c, "acc": acc}
    print(f"  {a:7s} {n:4d} " + " ".join(f"{c[g]:7d}" for g in (4, 3, 2, 1, 0))
          + f" {100*acc/max(n,1):6.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%]")

print()
print("  PAIRWISE (two-sided Fisher exact on 可接受)")
print("  " + "-" * 62)
import itertools
for a, b in itertools.combinations(ARMS, 2):
    na, ka = tab[a]["n"], tab[a]["acc"]
    nb, kb = tab[b]["n"], tab[b]["acc"]
    p = fisher(ka, na - ka, kb, nb - kb)
    d = 100 * (ka / na - kb / nb)
    print(f"    {a:6s} vs {b:6s} {100*ka/na:5.1f}% vs {100*kb/nb:5.1f}%  "
          f"({d:+5.1f} 点)  p = {p:.4f}  {'显著' if p < 0.05 else '不显著'}")

print()
print("  「中」的比例 -- 这是最接近'糊成一团'的一类")
for a in ARMS:
    n = tab[a]["n"]
    print(f"    {a:7s} 中 {tab[a]['counts'][1]:3d} = {100*tab[a]['counts'][1]/n:5.1f}%   "
          f"差 {tab[a]['counts'][0]:2d} = {100*tab[a]['counts'][0]/n:4.1f}%")

# ================================================================ colour features
def to_img(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def feats(flat):
    img = to_img(np.asarray(flat, dtype=np.float64))
    a = np.clip(np.asarray(flat, dtype=np.float64), 0, 15).reshape(len(flat), 32, 3)
    x = img.reshape(len(img), 64, 3)
    mx = x.max(-1); mn = x.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    d = np.linalg.norm(a[:, :, None, :] - a[:, None, :, :], axis=-1)
    iu = np.triu_indices(32, 1)
    sep = d[:, iu[0], iu[1]].mean(1)
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    ed = np.concatenate([dh.reshape(len(img), -1), dv.reshape(len(img), -1)], 1)
    ncol = np.array([len({tuple(np.rint(c).astype(int)) for c in a[i]}) for i in range(len(a))])
    order = np.argsort(sat, 1)
    q = np.array([[sat[i, order[i, 0]], sat[i, order[i, -1]]] for i in range(len(a))])
    return np.c_[sat.mean(1), sep, np.percentile(ed, 90, axis=1), ed.mean(1),
                 ncol, q[:, 0], q[:, 1]]


F = []
Y = []
for r in r8:
    pl = r.get("predicted_levels")
    if not pl or len(pl) != 96:
        continue
    F.append(feats([pl])[0])
    Y.append(1.0 if r["grade"] >= 2 else 0.0)
F = np.array(F); Y = np.array(Y)
print()
print("=" * 100)
print("A COLOUR SCORER FITTED ON THESE 260 LABELS")
print("=" * 100)
FEATS = ["平均饱和", "主色分离度", "边缘p90", "平均边缘", "颜色数", "最低饱和", "最高饱和"]


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


print(f"  {'特征':12s} {'AUC':>7s} {'可接受':>9s} {'中/差':>9s}")
print("  " + "-" * 42)
for i, nm in enumerate(FEATS):
    a = auc(F[:, i], Y)
    print(f"  {nm:12s} {a:7.3f} {F[Y==1,i].mean():9.3f} {F[Y==0,i].mean():9.3f}")

idx = np.random.default_rng(0).permutation(len(F))
XP = []
for f in range(5):
    te = idx[f::5]; tr = np.setdiff1d(idx, te)
    mu = F[tr].mean(0); sd = F[tr].std(0) + 1e-9
    A = np.c_[(F[tr] - mu) / sd, np.ones(len(tr))]
    w = np.linalg.solve(A.T @ A + 3.0 * np.eye(A.shape[1]), A.T @ Y[tr])
    XP[te] if False else None
    if f == 0:
        pred = np.zeros(len(F))
    pred[te] = np.c_[(F[te] - mu) / sd, np.ones(len(te))] @ w
AUC_COL = auc(pred, Y)
print(f"\n  组合打分器（7 个特征，5 折交叉验证）AUC = {AUC_COL:.3f}")
print(f"  对比：光度打分器在这些脸上只有 0.616；单用'平均饱和'是 "
      f"{auc(F[:,0], Y):.3f}，'主色分离度'是 {auc(F[:,1], Y):.3f}")

A = np.c_[(F - F.mean(0)) / (F.std(0) + 1e-9), np.ones(len(F))]
W = np.linalg.solve(A.T @ A + 3.0 * np.eye(A.shape[1]), A.T @ Y)
json.dump({"auc": float(AUC_COL), "feats": FEATS, "w": W.tolist(),
           "mu": F.mean(0).tolist(), "sd": F.std(0).tolist()},
          open(os.path.join(OUT, "colour_scorer.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'colour_scorer.json')}")
print("ROUND8_STATS_DONE")
