#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Test the user's stated criteria against their own 204 colour labels.

WHAT THEY SAID
    "如果某个风格的图能看，我就不打差。打为中，一般都是出现了色彩模糊成一团的情况。
     我评为优良的图一般都有一些共同点，比如鲜明、均一、色彩边界清晰。
     同时一张好图的几种主要颜色遵循一定比例，一般低饱和:中饱和:高饱和 = 7:2:1。"

That is a testable set of claims, and it matters because the current scorer only sees
LUMINANCE. If colour features separate their good faces from 中, then the filter has been
looking at the wrong thing.

Each claim becomes a number per face:

    鲜明       mean saturation, and the share of positions that are strongly saturated
    均一       within-region colour scatter after splitting the face into three colour groups
    边界清晰   mean colour step between neighbouring pixels, i.e. how visible the edges are
    7:2:1      the low/mid/high saturation shares, compared against 0.70/0.20/0.10

None of these are trained. They are computed straight from the 32 stored pixels, so the AUCs
below are honest descriptions of what the user is responding to, not a fitted model.
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
OUT = os.path.join(ROOT, "results", "colour")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r6 = [r for r in recs if str(r.get("round") or "") == "6" and len(r.get("predicted_levels") or []) == 96]
N = len(r6)
print(f"colour labels: {N}")


def to_img(r):
    v = np.clip(np.array(r["predicted_levels"], dtype=np.float64), 0, 15).reshape(32, 3)
    img = np.zeros((8, 8, 3), dtype=np.float64)
    img[layout.POS_ROW, layout.POS_COL] = v
    img[layout.POS_ROW, layout.MIRROR_COL] = v
    return img


def sat_of(x):                      # x: (...,3) 0..15
    mx = x.max(-1); mn = x.min(-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def kmeans3(X, iters=12):
    """cheap 3-means on the 32 colours, deterministic init from the saturation order."""
    s = sat_of(X)
    order = np.argsort(s)
    C = X[[order[0], order[len(order) // 2], order[-1]]].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(3):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    within = 0.0
    for j in range(3):
        m = lab == j
        if m.sum():
            within += ((X[m] - C[j]) ** 2).sum(-1).sum()
    between = 0.0
    cnt = 0
    for i in range(3):
        for j in range(i + 1, 3):
            between += np.linalg.norm(C[i] - C[j]); cnt += 1
    return within / len(X), (between / max(cnt, 1)), C, lab


FEATS = ["sat_mean", "sat_lo", "sat_mid", "sat_hi", "ratio_lo_hi",
         "edge_mean", "edge_p90", "n_colours", "lum_levels",
         "within_scatter", "between_sep", "separation", "sat_entropy"]
rows, Y = [], []
for r in r6:
    img = to_img(r)
    x = img.reshape(-1, 3)
    s = sat_of(x)
    lo = float((s < 0.15).mean()); mid = float(((s >= 0.15) & (s <= 0.45)).mean())
    hi = float((s > 0.45).mean())
    # colour step between neighbouring pixels, both directions
    dh = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    dv = np.linalg.norm(img[1:] - img[:-1], axis=-1)
    ed = np.concatenate([dh.ravel(), dv.ravel()])
    ncol = len({tuple(np.rint(c).astype(int)) for c in x})
    lum = x @ np.array([0.299, 0.587, 0.114])
    lv = len(set(np.rint(lum).astype(int).tolist()))
    within, between, C, lab = kmeans3(x)
    # saturation entropy: concentrated saturation (one dominant band) scores low
    p = np.array([lo, mid, hi])
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())
    rows.append([float(s.mean()), lo, mid, hi, lo / max(hi, 1e-3),
                 float(ed.mean()), float(np.percentile(ed, 90)), ncol, lv,
                 within, between, between / max(within, 1e-6), ent])
    Y.append(1.0 if r["grade"] >= 2 else 0.0)
F = np.array(rows); Y = np.array(Y)


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    if hi.size == 0 or lo.size == 0:
        return float("nan")
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


print()
print("=" * 100)
print("DOES EACH STATED CRITERION ACTUALLY SEPARATE 可接受 FROM 中/差 ?")
print("=" * 100)
print(f"  base rate acceptable: {100*Y.mean():.1f}%   (n={len(Y)})")
print()
print(f"  {'feature':16s} {'AUC':>7s} {'可接受 均值':>13s} {'中/差 均值':>13s}  方向")
print("  " + "-" * 66)
order = sorted(range(len(FEATS)),
               key=lambda i: -abs(auc(F[:, i], Y) - 0.5))

res = {}
for i in order:
    a = auc(F[:, i], Y)
    res[FEATS[i]] = a
    m1 = F[Y == 1, i].mean(); m0 = F[Y == 0, i].mean()
    d = "可接受更高" if m1 > m0 else "中/差更高"
    print(f"  {FEATS[i]:16s} {a:7.3f} {m1:13.3f} {m0:13.3f}  {d}")

print()
print("=" * 100)
print("THE 7:2:1 CLAIM, MEASURED")
print("=" * 100)
lo_ = F[:, 1]; mid_ = F[:, 2]; hi_ = F[:, 3]
print(f"  {'group':12s} {'低饱和':>8s} {'中饱和':>8s} {'高饱和':>8s}")
print("  " + "-" * 42)
print(f"  {'可接受':12s} {lo_[Y==1].mean():8.3f} {mid_[Y==1].mean():8.3f} {hi_[Y==1].mean():8.3f}")
print(f"  {'中/差':12s} {lo_[Y==0].mean():8.3f} {mid_[Y==0].mean():8.3f} {hi_[Y==0].mean():8.3f}")
print(f"  {'你说的 7:2:1':12s} {0.700:8.3f} {0.200:8.3f} {0.100:8.3f}")
# how close is each face to 7:2:1?
dist = np.abs(np.stack([lo_, mid_, hi_], 1) - np.array([0.7, 0.2, 0.1])).sum(1)
print()
print(f"  distance to 7:2:1: 可接受 {dist[Y==1].mean():.3f}   中/差 {dist[Y==0].mean():.3f}")
print(f"  AUC of (-distance) = {auc(-dist, Y):.3f}")

json.dump({"auc": res, "n": int(len(Y)), "base": float(Y.mean()),
           "sat_shares": {"good": [float(lo_[Y==1].mean()), float(mid_[Y==1].mean()),
                                   float(hi_[Y==1].mean())],
                          "bad": [float(lo_[Y==0].mean()), float(mid_[Y==0].mean()),
                                  float(hi_[Y==0].mean())]}},
          open(os.path.join(OUT, "colour_criteria.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'colour_criteria.json')}")
print("COLOUR_CRITERIA_DONE")
