#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
What actually separates the liked faces from the rejected ones: face structure, not colour.

THE OBSERVATION FROM THE MONTAGES
Good faces read as faces because they have three things: a hair or hat band across the top, a
pair of dark eyes set into a lighter skin field, and shading below. Bad faces are mostly flat --
one or two colour fields with no eyes at all, or eyes floating in a field with no boundary.

Global statistics barely see this. The strongest single feature in the first pass was edge p90 at
AUC 0.613, because many bad faces also have plenty of edges; the edges are just in the wrong
place. This script measures structure-specific features instead, and then asks how much of the
user's verdict any function of the 8 inputs could possibly explain.
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour", "yield")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114])


def auc(s, t):
    hi = s[t == 1][:, None]
    lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
col = [r for r in recs if str(r.get("round") or "") in ("8", "11")
       and len(r.get("predicted_levels") or []) == 96 and r.get("scheme") == "style"]
FLAT = np.array([r["predicted_levels"] for r in col], dtype=np.float64)
G = np.array([r["grade"] for r in col])
Y = (G >= 2).astype(int)
A = np.clip(FLAT, 0, 15).reshape(len(FLAT), 32, 3)

# rows 0..7, cols 0..3 (the mirrored half is implied). Position p -> row p%8, col p//8
ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])
LUMP = A @ LUM          # (n, 32) luminance per position

def col_of(r, c):
    return c * 8 + r

# the eye band: rows 3-4, the two inner columns
eye_idx = [col_of(r, c) for r in (3, 4) for c in (1, 2)]
inner = [col_of(r, c) for r in (3, 4) for c in (1, 2)]
skin_idx = [col_of(r, c) for r in (5, 6) for c in (0, 1, 2, 3)]     # lower face = skin
hair_idx = [col_of(r, c) for r in (0, 1) for c in (0, 1, 2, 3)]     # top band = hair

eye_min = LUMP[:, eye_idx].min(1)
skin_max = LUMP[:, skin_idx].max(1)
skin_med = np.median(LUMP[:, skin_idx], 1)
eye_contrast = skin_med - eye_min
hair_minus_skin = np.abs(LUMP[:, hair_idx].mean(1) - LUMP[:, skin_idx].mean(1))

# how many distinct colours in how many rows: a flat face has identical rows
row_means = np.array([[LUMP[i, [col_of(r, c) for c in range(4)]].mean()
                       for r in range(8)] for i in range(len(LUMP))])
row_variation = row_means.std(1)
# vertical banding: is there a boundary between the hair band and the skin?
band_jump = np.abs(row_means[:, :2].mean(1) - row_means[:, 4:].mean(1))

# connected colour regions: flat faces have very few
def n_regions(a):
    seen = np.zeros((8, 8), bool)
    img = np.zeros((8, 8, 3))
    img[:, layout.POS_COL[:]] = np.clip(a, 0, 15)
    img[:, layout.MIRROR_COL[:]] = np.clip(a, 0, 15)
    key = np.rint(img).astype(int)
    n = 0
    for r in range(8):
        for c in range(8):
            if seen[r, c]:
                continue
            n += 1
            stack = [(r, c)]
            seen[r, c] = True
            k = key[r, c]
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and not seen[nr, nc]:
                        if (key[nr, nc] == k).all():
                            seen[nr, nc] = True
                            stack.append((nr, nc))
    return n


regions = np.array([n_regions(A[i]) for i in range(len(A))])

FEATS = {
    "眼睛最暗值": eye_min,
    "眼睛与皮肤的对比": eye_contrast,
    "上下色带差(发vs肤)": hair_minus_skin,
    "行亮度变化": row_variation,
    "行均值带跳变": band_jump,
    "连通色块数": regions.astype(float),
}

print("=" * 100)
print(f"STRUCTURAL FEATURES  ({len(Y)} colour labels, 可接受 {100*Y.mean():.1f}%)")
print("=" * 100)
print(f"  {'特征':22s} {'可接受':>9s} {'不可接受':>9s} {'AUC':>7s} {'强度':>6s}")
print("  " + "-" * 62)
for k, v in sorted(FEATS.items(), key=lambda t: -abs(auc(t[1], Y) - 0.5)):
    print(f"  {k:22s} {v[Y==1].mean():9.3f} {v[Y==0].mean():9.3f} {auc(v,Y):7.3f} "
          f"{abs(auc(v,Y)-0.5)*2:6.2f}")

# a simple composite: does it have eyes AND a hair band AND some structure?
has_eyes = (eye_contrast > 4.0).astype(float)
has_hair = (hair_minus_skin > 2.0).astype(float)
has_struct = (row_variation > 1.5).astype(float)
comp = has_eyes + has_hair + has_struct
print()
print(f"  三个结构条件同时看（眼睛对比>4 + 上下色带差>2 + 行变化>1.5）：")
for s in (0, 1, 2, 3):
    m = comp == s
    if m.sum() < 5:
        continue
    print(f"    满足 {s} 个条件: {int(m.sum()):4d} 张，可接受率 {100*Y[m].mean():5.1f}%")
print(f"  复合分 AUC = {auc(comp, Y):.3f}")

# ---- how much is explainable at all? a probe on the 8 inputs, cross-validated
Z8 = np.array([r["pca"][:8] for r in col], dtype=np.float64)
def probe_cv(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    scores = np.zeros(len(y))
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(k) if j != f])
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        A_ = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
        w = np.linalg.solve(A_.T @ A_ + 3 * np.eye(A_.shape[1]), A_.T @ y[tr])
        scores[te] = np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ w
    return scores

print()
print("=" * 100)
print("CEILING: how well can ANY linear function of the 8 inputs predict the verdict?")
print("=" * 100)
s_in = probe_cv(Z8, Y.astype(float))
print(f"  从 8 个输入做 5 折交叉验证线性探针  AUC = {auc(s_in, Y):.3f}")
feat_mat = np.c_[eye_contrast, hair_minus_skin, row_variation, regions,
                 np.percentile(np.abs(np.diff(LUMP, axis=1)), 90, axis=1)]
s_feat = probe_cv(feat_mat, Y.astype(float))
print(f"  从手工特征做探针                    AUC = {auc(s_feat, Y):.3f}")
s_both = probe_cv(np.c_[Z8, feat_mat], Y.astype(float))
print(f"  两者合并                            AUC = {auc(s_both, Y):.3f}")
print()
print("  这就是筛选能达到的上限。若筛选按 best-of-n 挑，良率大致是")
for n in (2, 4):
    for label, s in (("输入探针", s_in), ("特征探针", s_feat)):
        keep = np.argsort(-s)[:len(s)//n]
        print(f"    {label} {n} 选 1: {100*Y[keep].mean():5.1f}%")

json.dump({"auc_eye": auc(eye_contrast, Y), "auc_regions": auc(regions, Y),
           "auc_rowvar": auc(row_variation, Y), "auc_in": auc(s_in, Y),
           "auc_feat": auc(s_feat, Y), "auc_both": auc(s_both, Y)},
          open(os.path.join(OUT, "structure.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\nCOLOUR_STRUCTURE_DONE")
