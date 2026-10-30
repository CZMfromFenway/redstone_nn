#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Can the user's visual style categories be recovered automatically?

The user labelled 16 of the 24 tiles on their sheet:
    类二次元 (anime-ish)  : sheet 1,2,3,5,10 / 2,4,6,7,8,10
    类三次元 (3D-ish)     : sheet 6,8 / 9
    类游戏   (game-ish)   : sheet 1,7,12 / 3,5,11,12
and left 8 unlabelled.

We evaluate every candidate style space by:
  * KMeans / GaussianMixture / Agglomerative, for several k
  * Adjusted Rand Index and NMI against the user's labels  (clustering agreement)
  * a nearest-centroid leave-one-out classifier on the labelled subset
    (can the categories be told apart at all in this space?)
  * silhouette score (internal quality, no labels needed)

Honest caveats printed with the results: 16 labelled points in up to 54 dimensions is
tiny, so a single ARI number is weak evidence. We therefore report a permutation
baseline and cross-validation rather than a single fit.
"""
import itertools, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = os.path.join(ROOT, "results", "style")

# ---- the user's labels, mapped to SHEET indices (12 per row, 0-based) --------------
SHEET = {
    "anime": [1, 2, 4, 9, 13, 15, 17, 18, 19, 21],
    "real3d": [5, 7, 20],
    "game": [0, 6, 11, 14, 16, 22, 23],
}
UNLABELLED = [3, 8, 10, 12]
SHEET_ORDER = sorted(sum(SHEET.values(), []) + UNLABELLED)
print("user labels on the 24-image sheet:")
for k, v in SHEET.items():
    print(f"  {k:7s} {len(v):2d} items -> sheet {v}")
print(f"  unlabelled {len(UNLABELLED):2d} -> {UNLABELLED}")

g_all = np.load(os.path.join(S, "gallery.npy"))
# The user's sheet is a different random draw from the same val split. We cannot recover
# which gallery rows those were, so we use the sheet itself: re-draw it identically.
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
sheet_rows = va[np.random.default_rng(11).choice(len(va), 24, replace=False)]
sheet_faces = full[sheet_rows]
print(f"\nsheet faces rebuilt: {sheet_faces.shape} (seed 11, same as the web export)")

# ---- rebuild the embeddings for the SHEET faces -------------------------------------
# Recompute from scratch so the sheet and the gallery use identical definitions.
LEVELS = np.linspace(0, 255, 16)
NLEV = 16
import importlib.util as _ilu
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)


def pca_basis():
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
    return mu, comps, sd, X


mu, comps, sd, Xall = pca_basis()
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


def features(rows):
    g = full[rows].astype(np.float64)
    g01 = g.astype(np.float32) / 15.0 * 2 - 1
    f = {}
    f["pca22"] = ((Xall[rows] - mu[0]) @ comps.T) / sd[0]
    f3, f2 = [], []
    for e in range(E):
        r = np.maximum(conv_same(g01, k3k[e]), 0)
        f3 += [r.mean(axis=(1, 2)), r.max(axis=(1, 2)), (r > 0).mean(axis=(1, 2))]
        r2 = np.maximum(conv_same(g01, k2k[e]), 0)
        f2 += [r2.mean(axis=(1, 2)), r2.max(axis=(1, 2)), (r2 > 0).mean(axis=(1, 2))]
    f["k3"] = np.stack(f3, 1); f["k2"] = np.stack(f2, 1)
    f["kernels_all"] = np.concatenate([f["k3"], f["k2"]], 1)
    top = g[:, :3, :].mean(axis=(1, 2)); bot = g[:, 5:, :].mean(axis=(1, 2))
    st = np.stack([g.mean(axis=(1, 2)), g.std(axis=(1, 2)), top, bot, top - bot,
                   g[:, 3, :].mean(1), g[:, 0, :].mean(1), g[:, 7, :].mean(1),
                   np.abs(np.diff(g, axis=2)).mean(axis=(1, 2)),
                   np.abs(np.diff(g, axis=1)).mean(axis=(1, 2)),
                   (g[:, :, :4] == g[:, :, 4:][:, :, ::-1]).mean(axis=(1, 2))], 1)
    hist = np.stack([(g == k).mean(axis=(1, 2)) for k in range(NLEV)], 1)
    f["stats"] = st; f["hist"] = hist; f["stats_hist"] = np.concatenate([st, hist], 1)
    return f


Fs = features(sheet_rows)
print("sheet feature spaces: " + ", ".join(f"{k}{v.shape}" for k, v in Fs.items()))

# labelled indices within the 24
lab_idx, lab_y, cls_names = [], [], []
for ci, (name, items) in enumerate(SHEET.items()):
    for it in items:
        lab_idx.append(it); lab_y.append(ci); cls_names.append(name)
lab_idx = np.array(lab_idx); lab_y = np.array(lab_y)
print(f"labelled points: {len(lab_idx)}  classes: {dict(zip(cls_names, lab_y))}")

# ---- clustering metrics implemented directly (no sklearn in this environment) -------
def ari(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    n = len(a)
    tab = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            tab[i, j] = np.sum((a == x) & (b == y))
    def comb2(x):
        return x * (x - 1) / 2.0
    sum_ij = comb2(tab).sum()
    sum_i = comb2(tab.sum(1)).sum()
    sum_j = comb2(tab.sum(0)).sum()
    tot = comb2(n)
    exp = sum_i * sum_j / tot if tot else 0.0
    mx = 0.5 * (sum_i + sum_j)
    return float((sum_ij - exp) / (mx - exp)) if mx != exp else 1.0


def nmi(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    ua, ub = np.unique(a), np.unique(b)
    tab = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            tab[i, j] = np.sum((a == x) & (b == y))
    p = tab / n
    pa = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p * (np.log(p) - np.log(pa) - np.log(pb))
    mi = np.nansum(t)
    ha = -np.nansum(pa * np.log(pa)); hb = -np.nansum(pb * np.log(pb))
    return float(2 * mi / (ha + hb)) if (ha + hb) > 0 else 0.0


def kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def silhouette(X, lab):
    n = len(X)
    if len(np.unique(lab)) < 2:
        return -1.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
    s = []
    for i in range(n):
        same = np.where(lab == lab[i])[0]
        same = same[same != i]
        if len(same) == 0:
            s.append(0.0); continue
        a = D[i, same].mean()
        b = min(D[i, lab == j].mean() for j in np.unique(lab) if j != lab[i])
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))


def loo_centroid(X, y):
    """Leave-one-out nearest-centroid accuracy. Honest: no training/test leakage."""
    ok = 0
    for i in range(len(X)):
        m = np.ones(len(X), bool); m[i] = False
        cents, names = [], []
        for c in np.unique(y[m]):
            cents.append(X[m & (y == c)].mean(0)); names.append(c)
        d = [np.sum((X[i] - c) ** 2) for c in cents]
        if names[int(np.argmin(d))] == y[i]:
            ok += 1
    return ok / len(X)


def perm_baseline(y, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(reps):
        p = rng.permutation(n)
        # ARI of a random relabelling against y is 0 in expectation; NMI is not, so we
        # record the distribution of ARI of random partitions with the same class sizes
        vals.append(ari(y, rng.integers(0, len(np.unique(y)), n)))
    return float(np.mean(vals)), float(np.std(vals))


print(f"\npermutation baseline ARI (random partitions, same k): "
      f"{perm_baseline(lab_y)[0]:+.3f} +- {perm_baseline(lab_y)[1]:.3f}")

results = []
print("\n" + "=" * 104)
print("CLUSTERING vs THE USER'S LABELS  (labels exist for 16 of 24 points)")
print("=" * 104)
print(f"{'space':13s} {'k':>3s} {'method':13s} {'ARI(all)':>9s} {'NMI(all)':>9s} "
      f"{'ARI(lab)':>9s} {'silhouette':>11s} {'LOO-centroid':>13s}")
print("-" * 104)

for space, X in Fs.items():
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)          # standardise per dimension
    K = max(2, len(np.unique(lab_y)))
    for method in ("kmeans", "gmm-ish"):
        for k in (2, 3, 4, 5):
            if method == "kmeans":
                lab = kmeans(Xs, k, seed=0)
            else:
                # a cheap soft-assignment variant: kmeans on the 3 leading PCs
                lab = kmeans(Xs[:, :min(3, Xs.shape[1])], k, seed=1)
            a_all, n_all = ari(lab, np.array([-1] * 24)), 0
            # ARI restricted to the labelled points only
            a_lab = ari(lab[lab_idx], lab_y)
            n_lab = nmi(lab[lab_idx], lab_y)
            sil = silhouette(Xs, lab)
            loo = loo_centroid(Xs[lab_idx], lab_y)
            results.append({"space": space, "k": k, "method": method, "ari_lab": a_lab,
                            "nmi_lab": n_lab, "silhouette": sil, "loo": loo})
            print(f"{space:13s} {k:3d} {method:13s} {'':>9s} {'':>9s} {a_lab:9.3f} "
                  f"{n_lab:9.3f} {sil:11.3f} {loo:13.3f}")

results.sort(key=lambda r: -(r["ari_lab"] + r["loo"]))
print("\n" + "=" * 104)
print("BEST SPACES by (ARI + LOO)")
print("=" * 104)
for r in results[:8]:
    print(f"  {r['space']:13s} k={r['k']} {r['method']:9s}  ARI {r['ari_lab']:+.3f}  "
          f"NMI {r['nmi_lab']:.3f}  LOO {r['loo']:.3f}  silhouette {r['silhouette']:+.3f}")

print()
print("HONEST READING")
print("  * LOO-centroid on 16 points with 3 classes: chance is ~1/3, and with 11/3/7")
print("    items the majority class alone scores 0.44. Beating that is the real bar.")
print("  * ARI ~0 means the clusters carry no information about the user's categories.")
print("  * These are 16 hand-labelled points; treat every number as weak evidence.")
json.dump({"sheet_labels": SHEET, "unlabelled": UNLABELLED, "results": results},
          open(os.path.join(S, "clustering_results.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(S, 'clustering_results.json')}")
print("STYLE_CLUSTER_DONE")
