#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
After a min-max stretch, is the face still too dark or too bright -- and does it matter?

WHAT A MIN-MAX STRETCH DOES AND DOES NOT DO
It maps the face's own [lo,hi] onto [0,15]. That fixes the RANGE. It does not touch the
SHAPE of the distribution: if 28 of 32 pixels sat in the upper half and four dragged the
bottom down, the stretch keeps exactly that imbalance and merely re-labels it 0..15. So a
face can be full-range and still be mostly bright or mostly dark. The question is whether
that residual bias is harmless variation or a defect.

HOW TO ANSWER IT WITHOUT GUESSING
The labelling records the exact 32 levels that were shown (`predicted_levels`), so for every
labelled dim4x face the mean brightness and the range are known, together with the grade the
user gave. If brightness bias damages quality it must show up as a lower grade at the
brightness extremes, WITHIN the same arm. That is a direct test on real judgements.

Reported
  1. distribution of mean brightness per arm, against real faces
  2. within dim4x: grade as a function of mean brightness and of range
  3. how much of the grade variance brightness alone can explain
"""
import collections
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
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "dimmix")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------- real faces
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
real = full[:, layout.POS_ROW, layout.MIRROR_COL].astype(float)
real_mean = real.mean(1)
real_con = real.max(1) - real.min(1)

print("=" * 100)
print("1. MEAN BRIGHTNESS AFTER THE STRETCH")
print("=" * 100)
print(f"  real faces: mean brightness {real_mean.mean():.2f}, "
      f"share outside 5.5..9.5 = {100*np.mean((real_mean<5.5)|(real_mean>9.5)):.1f}%")
print(f"  (a perfectly centred face would sit at 7.5; min-max does not force that)")

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if r.get("predicted_levels") and r.get("grade") in (0, 1, 2, 3)]
print()
print(f"  {'arm':8s} {'n':>4s} {'mean bright':>12s} {'outside 5.5-9.5':>17s} {'mean range':>11s}")
print("  " + "-" * 58)
by_arm = collections.defaultdict(list)
for r in lab:
    by_arm[r["model"]].append(r)
summ = {}
for k in sorted(by_arm):
    rows = by_arm[k]
    if len(rows) < 15:
        continue
    b = np.array([np.mean(r["predicted_levels"]) for r in rows])
    c = np.array([max(r["predicted_levels"]) - min(r["predicted_levels"]) for r in rows])
    out = 100 * np.mean((b < 5.5) | (b > 9.5))
    summ[k] = {"n": len(rows), "bright": float(b.mean()), "outside": float(out),
               "range": float(c.mean())}
    print(f"  {k:8s} {len(rows):4d} {b.mean():12.2f} {out:16.1f}% {c.mean():11.2f}")

# ---------------------------------------------------------------- does brightness matter
print()
print("=" * 100)
print("2. WITHIN dim4x: DOES RESIDUAL BRIGHTNESS OR RANGE PREDICT YOUR GRADE?")
print("=" * 100)
for k in ("dim4x", "dim4"):
    rows = by_arm.get(k, [])
    if len(rows) < 30:
        continue
    b = np.array([np.mean(r["predicted_levels"]) for r in rows])
    c = np.array([max(r["predicted_levels"]) - min(r["predicted_levels"]) for r in rows])
    g = np.array([r["grade"] for r in rows], float)
    good = (g >= 2).astype(float)
    print()
    print(f"  {k}  (n={len(rows)}, 良+优 {100*good.mean():.1f}%)")
    print(f"    corr(brightness, grade) {np.corrcoef(b, g)[0,1]:+.3f}   "
          f"corr(range, grade) {np.corrcoef(c, g)[0,1]:+.3f}")
    # quartiles of brightness
    qs = np.quantile(b, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"    {'brightness bin':>26s} {'n':>4s} {'良+优':>7s} {'mean grade':>11s}")
    for i in range(4):
        m = (b >= qs[i]) & (b <= qs[i + 1] if i == 3 else b < qs[i + 1])
        if m.sum() < 3:
            continue
        print(f"    {qs[i]:6.2f} .. {qs[i+1]:6.2f} {m.sum():14d} "
              f"{100*good[m].mean():6.1f}% {g[m].mean():11.2f}")
    # is the extreme tail worse?
    lo_tail = b < 4.5
    hi_tail = b > 10.5
    mid = ~lo_tail & ~hi_tail
    for name, m in (("dark tail (<4.5)", lo_tail), ("middle", mid), ("bright tail (>10.5)", hi_tail)):
        if m.sum() >= 3:
            print(f"    {name:>26s} {m.sum():14d} {100*good[m].mean():6.1f}% {g[m].mean():11.2f}")

# ---------------------------------------------------------------- histogram picture
fig_rows = []
def arm_faces(tag, jf, post):
    p = os.path.join(WEB, jf)
    j = json.load(open(p, encoding="utf-8"))
    layers = j["gen"]["layers"]
    import torch
    import torch.nn as nn
    seq = []
    for i, L in enumerate(layers):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(layers) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.eval()
    k = int(len(layers[0]["W"][0]))
    Z = np.clip(np.random.default_rng(9).standard_normal((4000, k)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        v = np.clip(m(torch.tensor(Z)).numpy(), 0, 15)
    if post == "stretch":
        lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
        v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
    return v

ARMS = [("dim4 raw", "pca22_dim4_model.json", "none"),
        ("dim4x stretch", "pca22_dim4x_model.json", "stretch"),
        ("dim4st baked", "pca22_dim4st_model.json", "none"),
        ("dim22", "pca22_dim22_model.json", "none")]
print()
print("=" * 100)
print("3. BRIGHTNESS HISTOGRAM OF THE GENERATED LEVELS")
print("=" * 100)
print(f"  {'arm':16s} {'mean level':>11s} {'share of pixels in 0-3':>23s} {'in 12-15':>9s}")
print("  " + "-" * 62)
for name, jf, post in ARMS:
    try:
        v = arm_faces(None, jf, post)
    except Exception as e:
        print(f"  {name:16s} failed: {e}")
        continue
    print(f"  {name:16s} {v.mean():11.2f} {100*np.mean(v<=3):22.1f}% {100*np.mean(v>=12):8.1f}%")
    fig_rows.append((name, v))
print(f"  {'real faces':16s} {real.mean():11.2f} {100*np.mean(real<=3):22.1f}% "
      f"{100*np.mean(real>=12):8.1f}%")

json.dump(summ, open(os.path.join(OUT, "brightness_after_stretch.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'brightness_after_stretch.json')}")
print("BRIGHTNESS_STUDY_DONE")
