#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Two measurements the user asked for.

A. STYLE AS A LINEAR SWITCH
The old linear reward turned out to be a 类二次元 detector. The user's reading is fair: that
is not useless, it is a SWITCH -- turn it on and the faces get more anime. So fit one linear
probe per style (that style vs everything else) and report how separable each one is, plus
the direction in knob space. A high AUC means the switch is real; the direction is what you
would dial.

Also fit the thing that actually matters: acceptable (three styles) vs 中/差.

B. THE TWO-STAGE COLOUR DESIGN
Stage 2 would take the produced greyscale map PLUS a few colour principal components and emit
saturation and hue. The earlier measurement showed greyscale ALONE carries almost no colour
information (R^2 0.06-0.11). The question now is whether adding colour components fixes it,
and whether the greyscale map contributes anything on top of them.
"""
import json
import math
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "style5")
COUT = os.path.join(ROOT, "results", "colour")
os.makedirs(OUT, exist_ok=True); os.makedirs(COUT, exist_ok=True)
STYLE = {4: "类二次元", 3: "类游戏", 2: "类三次元", 1: "中", 0: "差"}
KNOB = ["PC0 明度", "PC1 发肤", "PC2 眼型", "PC3 脸型"]


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    if hi.size == 0 or lo.size == 0:
        return float("nan")
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float64)
g = np.array([r["grade"] for r in new])
zm = X.mean(0); zsd = X.std(0) + 1e-9
Xs = (X - zm) / zsd

print("=" * 104)
print(f"A. EACH STYLE AS ITS OWN LINEAR SWITCH  ({len(new)} labels, 5-fold CV)")
print("=" * 104)
print(f"  {'target':28s} {'n pos':>6s} {'cv AUC':>8s}   direction (knob weights)")
print("  " + "-" * 92)
switches = {}
TARGETS = [("类二次元 vs 其他", lambda gg: 1.0 if gg == 4 else 0.0),
           ("类游戏 vs 其他", lambda gg: 1.0 if gg == 3 else 0.0),
           ("类三次元 vs 其他", lambda gg: 1.0 if gg == 2 else 0.0),
           ("可接受 vs 中/差  (关键)", lambda gg: 1.0 if gg >= 2 else 0.0),
           ("中/差 vs 可接受  (反向)", lambda gg: 1.0 if gg <= 1 else 0.0)]
idx = np.random.default_rng(0).permutation(len(Xs))
for name, fn in TARGETS:
    y = np.array([fn(v) for v in g])
    if y.sum() < 5 or (1 - y).sum() < 5:
        print(f"  {name:28s} {int(y.sum()):6d}   (too few)")
        continue
    pred = np.zeros(len(new))
    for f in range(5):
        te = idx[f::5]
        tr = np.setdiff1d(idx, te)
        A = np.c_[Xs[tr], np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + 3.0 * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.c_[Xs[te], np.ones(len(te))] @ w
    a = auc(pred, y)
    # full-data direction for interpretation
    A = np.c_[Xs, np.ones(len(Xs))]
    w = np.linalg.solve(A.T @ A + 3.0 * np.eye(A.shape[1]), A.T @ y)
    d = w[:4] / zsd
    switches[name] = {"auc": a, "n_pos": int(y.sum()), "w4": d.tolist()}
    ds = "  ".join(f"{v:+.2f}" for v in d)
    print(f"  {name:28s} {int(y.sum()):6d} {a:8.3f}   {ds}")
print()
print("  knob order: " + "  ".join(KNOB))
print("  a positive weight means that knob has to go UP to reach the target style")

json.dump(switches, open(os.path.join(OUT, "style_switches.json"), "w"), indent=2,
          ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'style_switches.json')}")
print("STYLE_SWITCHES_DONE")
