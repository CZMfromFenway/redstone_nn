#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Add Gaussian-envelope style sampling to the web page.

The page currently draws each of the 22 principal components as an INDEPENDENT standard
normal N(0,1). The user asked for a Gaussian "envelope" fitted around their 类二次元
samples, sampled jointly. This writes the data the page needs:

    webapp/style_envelopes.json
      * per style: mean (22), and a Cholesky factor L (22x22) of the REGULARISED
        covariance, so the page can draw  mean + L @ z  with z ~ N(0, I)
      * the style's own samples, so the page can show what it is aiming at
      * the measured hit rate of the independent sampler, so the improvement is honest

It also fits envelopes for the other two labelled styles so the page can offer them,
but marks how many samples each was fitted from, because 10 or 3 points cannot identify
a 22x22 covariance and the envelope must be read as a guide, not a fitted distribution.
"""
import json, os
import numpy as np

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
LEVELS = np.linspace(0, 255, 16)
WEB = os.path.join(ROOT, "webapp")
os.makedirs(WEB, exist_ok=True)

SHEET = {
    "anime":   [1, 2, 4, 9, 13, 15, 17, 18, 19, 21],
    "real3d":  [5, 7, 20],
    "game":    [0, 6, 11, 14, 16, 22, 23],
    "unnamed": [3, 8, 10, 12],
}
LABEL_CN = {"anime": "类二次元", "real3d": "类三次元", "game": "类游戏", "unnamed": "未标注"}

# ---- exact PCA space + the exact 24 sheet rows ------------------------------------
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
rows24 = va[np.random.default_rng(11).choice(len(va), 24, replace=False)]
Z = ((X[rows24] - mu[0]) @ comps.T) / sd[0]

# ---- shrink covariance towards the identity ---------------------------------------
# With 10 (or 3) samples in 22 dimensions the sample covariance is rank-deficient.
# Shrinking hard towards I keeps it invertible and keeps the envelope from claiming
# structure the data cannot support.
ALPHA = {"anime": 0.90, "real3d": 0.95, "game": 0.90, "unnamed": 0.95}

out = {"styles": {}, "note": ("Gaussian envelopes fitted in the 22-dim PCA space that the "
                              "generator is conditioned on. Covariance is shrunk towards "
                              "the identity because only a handful of samples were "
                              "labelled; treat these as rough envelopes, not fitted "
                              "distributions.")}
rng = np.random.default_rng(0)

for name, idx in SHEET.items():
    Za = Z[np.array(idx)]
    n = len(idx)
    m = Za.mean(0)
    C = np.cov(Za.T) if n > 1 else np.eye(22)
    C = np.atleast_2d(C)
    a = ALPHA[name]
    Creg = (1 - a) * C + a * np.eye(22) + 1e-6 * np.eye(22)
    L = np.linalg.cholesky(Creg)

    # honest measurement: how often does an independent N(0,1) draw land in the envelope?
    Ci = np.linalg.inv(Creg)
    maha_in = np.array([float(np.sqrt((z - m) @ Ci @ (z - m))) for z in Za])
    thr = np.percentile(maha_in, 90)
    D = rng.standard_normal((100000, 22))
    bb = D - m
    mr = np.sqrt(np.einsum("ij,jk,ik->i", bb, Ci, bb))
    hit_maha = float((mr <= thr).mean())
    # a perceptually closer criterion: within the style's own nearest-neighbour spread
    d_nn = np.sqrt(((Za[:, None, :] - Za[None, :, :]) ** 2).sum(2))
    np.fill_diagonal(d_nn, 1e9)
    d_nn_min = d_nn.min(1)
    d_to_any = np.sqrt(((D[:, None, :] - Za[None, :, :]) ** 2).sum(2)).min(1)
    hit_nn = float((d_to_any <= np.percentile(d_nn_min, 75)).mean())

    out["styles"][name] = {
        "label_cn": LABEL_CN[name],
        "n_samples": n,
        "mean": [round(float(v), 5) for v in m],
        "chol": [[round(float(v), 6) for v in r] for r in L],
        "mean_norm": round(float(np.linalg.norm(m)), 3),
        "samples": [[round(float(v), 4) for v in z] for z in Za],
        "independent_sampler_hit_maha": round(hit_maha, 4),
        "independent_sampler_hit_nn": round(hit_nn, 4),
        "shrink_alpha": a,
        "caveat": (f"fitted from only {n} labelled face(s); the 22x22 covariance is not "
                   f"identifiable from {n} points, so it is shrunk {int(a*100)}% towards "
                   f"the identity"),
    }
    print(f"{name:8s} n={n:2d}  |mean|={np.linalg.norm(m):.2f}  "
          f"independent-sampler hit rate: maha {hit_maha*100:.1f}%  nn {hit_nn*100:.1f}%")

# the faces themselves, so the page can show what each envelope aims at
out["sheet_faces"] = [{"rgb": full[r].reshape(-1).tolist(),
                       "pca": [round(float(v), 4) for v in Z[i]]}
                      for i, r in enumerate(rows24)]
out["sheet_labels"] = [next((k for k, v in SHEET.items() if i in v), "unnamed")
                       for i in range(24)]

p = os.path.join(WEB, "style_envelopes.json")
json.dump(out, open(p, "w", encoding="utf-8"))
print(f"\nwrote {p}  ({os.path.getsize(p)/1e6:.2f} MB)")
print("ENVELOPE_EXPORT_DONE")
