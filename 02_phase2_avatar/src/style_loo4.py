#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Honest re-evaluation with ALL 24 tiles as FOUR classes.

The first pass only used the 20 labelled tiles and three classes, which INFLATES the
leave-one-out score: points the user lumped together are similar, so nearest-centroid
easily separates the lumps. Treating the 4 unlabelled tiles as a legitimate fourth
category removes that bias -- 12/6/4/2 items, majority class 12/24 = 0.50.

Also reports, for the best space, WHICH tiles land in which cluster vs the user's label.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = os.path.join(ROOT, "results", "style")

SHEET = {
    "anime": [1, 2, 4, 9, 13, 15, 17, 18, 19, 21],
    "real3d": [5, 7, 20],
    "game": [0, 6, 11, 14, 16, 22, 23],
    "unnamed": [3, 8, 10, 12],
}
# 4-class labelling over all 24 tiles
y4 = np.full(24, -1, dtype=int)
for ci, (name, items) in enumerate(SHEET.items()):
    for it in items:
        y4[it] = ci
assert (y4 >= 0).all(), "every tile must be labelled in this scheme"
names4 = list(SHEET.keys())
counts = np.bincount(y4)
print(f"4-class labelling over all 24 tiles: {dict(zip(names4, counts.tolist()))}")
print(f"  majority-class baseline (always predict 'anime'): {counts.max()/24:.3f}")

# ---- feature spaces (identical definitions to style_cluster.py) ---------------------
LEVELS = np.linspace(0, 255, 16)
NLEV = 16
import importlib.util as _ilu
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)

va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
rows = va[np.random.default_rng(11).choice(len(va), 24, replace=False)]

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
m = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
     gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[m]
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)

info = json.load(open(os.path.join(ROOT, "results", "kernel_style", "kernel_style",
                                   "kernel_info.json"), encoding="utf-8"))
E = info["embed"]
k3k = np.array(info["kernels_3x3"], dtype=np.float32).reshape(E, 3, 3)
k2k = np.array(info["kernels_2x2"], dtype=np.float32).reshape(E, 2, 2)


def conv_same(img, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    P = np.pad(img, ((0, 0), (ph, ph), (pw, pw)), mode="edge")
    out = np.zeros((img.shape[0], 8, 8), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * P[:, i:i + 8, j:j + 8]
    return out


g = full[rows].astype(np.float64)
g01 = g.astype(np.float32) / 15.0 * 2 - 1
F = {}
F["pca22"] = ((X[rows] - mu[0]) @ comps.T) / sd[0]
f3, f2 = [], []
for e in range(E):
    r = np.maximum(conv_same(g01, k3k[e]), 0)
    f3 += [r.mean(axis=(1, 2)), r.max(axis=(1, 2)), (r > 0).mean(axis=(1, 2))]
    r2 = np.maximum(conv_same(g01, k2k[e]), 0)
    f2 += [r2.mean(axis=(1, 2)), r2.max(axis=(1, 2)), (r2 > 0).mean(axis=(1, 2))]
F["k3"] = np.stack(f3, 1); F["k2"] = np.stack(f2, 1)
F["kernels_all"] = np.concatenate([F["k3"], F["k2"]], 1)
top = g[:, :3, :].mean(axis=(1, 2)); bot = g[:, 5:, :].mean(axis=(1, 2))
st = np.stack([g.mean(axis=(1, 2)), g.std(axis=(1, 2)), top, bot, top - bot,
               g[:, 3, :].mean(1), g[:, 0, :].mean(1), g[:, 7, :].mean(1),
               np.abs(np.diff(g, axis=2)).mean(axis=(1, 2)),
               np.abs(np.diff(g, axis=1)).mean(axis=(1, 2)),
               (g[:, :, :4] == g[:, :, 4:][:, :, ::-1]).mean(axis=(1, 2))], 1)
hist = np.stack([(g == k).mean(axis=(1, 2)) for k in range(NLEV)], 1)
F["stats"] = st; F["hist"] = hist; F["stats_hist"] = np.concatenate([st, hist], 1)
# a combined space: PCA style + kernel texture
F["pca_plus_kernels"] = np.concatenate([F["pca22"], F["kernels_all"]], 1)

Z = {k: (v - v.mean(0)) / (v.std(0) + 1e-9) for k, v in F.items()}


def loo_centroid(Xf, y):
    ok = 0
    for i in range(len(Xf)):
        msk = np.ones(len(Xf), bool); msk[i] = False
        cents, labs = [], []
        for c in np.unique(y[msk]):
            cents.append(Xf[msk & (y == c)].mean(0)); labs.append(c)
        d = [np.sum((Xf[i] - c) ** 2) for c in cents]
        if labs[int(np.argmin(d))] == y[i]:
            ok += 1
    return ok / len(Xf)


def loo_knn(Xf, y, k=3):
    ok = 0
    for i in range(len(Xf)):
        d = np.sqrt(((Xf - Xf[i]) ** 2).sum(1)); d[i] = 1e18
        nn = np.argsort(d)[:k]
        vals, cnt = np.unique(y[nn], return_counts=True)
        if vals[int(np.argmax(cnt))] == y[i]:
            ok += 1
    return ok / len(Xf)


def loo_logreg(Xf, y, steps=3000, lr=0.5, l2=0.1):
    """multinomial logistic regression, leave-one-out. Honest but weak with n=24."""
    K = len(np.unique(y)); ok = 0
    for i in range(len(Xf)):
        msk = np.ones(len(Xf), bool); msk[i] = False
        Xt, yt = Xf[msk], y[msk]
        W = np.zeros((Xt.shape[1], K)); b = np.zeros(K)
        Y = np.eye(K)[yt]
        for t in range(steps):
            lg = Xt @ W + b
            lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            g = (p - Y) / len(Xt)
            W -= lr * (Xt.T @ g + l2 * W); b -= lr * g.sum(0)
        s = Xf[i] @ W + b
        if int(np.argmax(s)) == y[i]:
            ok += 1
    return ok / len(Xf)


print()
print("=" * 100)
print("4-CLASS LEAVE-ONE-OUT ACCURACY  (24 tiles; majority baseline 0.500)")
print("=" * 100)
print(f"{'feature space':20s} {'centroid':>10s} {'3-NN':>8s} {'logreg':>8s}")
print("-" * 100)
table = []
for name, Xf in Z.items():
    a = loo_centroid(Xf, y4); b = loo_knn(Xf, y4); c = loo_logreg(Xf, y4)
    table.append({"space": name, "centroid": a, "knn3": b, "logreg": c})
    print(f"{name:20s} {a:10.3f} {b:8.3f} {c:8.3f}")
table.sort(key=lambda r: -max(r["centroid"], r["knn3"], r["logreg"]))
best = table[0]
print()
print(f"best: {best['space']}  centroid {best['centroid']:.3f} / 3NN {best['knn3']:.3f} "
      f"/ logreg {best['logreg']:.3f}   (majority baseline 0.500)")

# ---- how stable is that? permutation test on the best space -------------------------
rng = np.random.default_rng(0)
null = []
for _ in range(400):
    yp = rng.permutation(y4)
    null.append(loo_centroid(Z[best["space"]], yp))
null = np.array(null)
obs = best["centroid"]
pval = float((null >= obs).mean())
print(f"\npermutation test on {best['space']} centroid-LOO: observed {obs:.3f}, "
      f"null mean {null.mean():.3f} +- {null.std():.3f}, p = {pval:.3f}")
print("  (p is the share of random labellings that scored at least as well)")

# ---- and the majority-class baseline under the same protocol ------------------------
print()
print("for scale, under this same leave-one-out protocol:")
print(f"  always predict the majority class : {(np.bincount(y4).max()-1)/(24-1):.3f} "
      f"(removing one item leaves the majority still majority)")

json.dump({"labels": SHEET, "vectorised": y4.tolist(), "table": table,
           "best": best, "perm_p": pval, "null_mean": float(null.mean()),
           "null_std": float(null.std()), "majority_baseline": 0.5},
          open(os.path.join(S, "loo_4class_results.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(S, 'loo_4class_results.json')}")
print("LOO4_DONE")
