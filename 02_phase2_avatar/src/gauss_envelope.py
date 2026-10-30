#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Fit a GAUSSIAN ENVELOPE around the user's 类二次元 samples in the 22-dim PCA space.

The page currently samples each principal component as an INDEPENDENT standard normal
N(0,1). Two things can be wrong with that:
  1. the anime-style faces may not sit near the ORIGIN -- they may occupy a corner of
     the space, so their mean is not 0
  2. the components may be CORRELATED within that style, which an independent draw
     cannot reproduce

This script measures both and fits a full multivariate normal (mean + covariance) to
the style's samples, then reports how much of the style's own probability mass the
current independent sampler would actually hit.

Caveat stated up front: the user labelled only 10 anime samples. Fitting a 22x22
covariance from 10 points is degenerate (rank <= 9), so the covariance is regularised
and the result must be treated as a rough envelope, not a fitted distribution.
"""
import json, os
import numpy as np

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
LEVELS = np.linspace(0, 255, 16)
OUT = os.path.join(ROOT, "results", "gauss_envelope")
os.makedirs(OUT, exist_ok=True)

# the user's labels on their 24-tile sheet (12 per row, 0-based within the 24)
SHEET = {
    "anime":  [1, 2, 4, 9, 13, 15, 17, 18, 19, 21],
    "real3d": [5, 7, 20],
    "game":   [0, 6, 11, 14, 16, 22, 23],
    "unnamed": [3, 8, 10, 12],
}

# rebuild the exact PCA space and the exact 24 sheet rows
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

Z = ((X[rows24] - mu[0]) @ comps.T) / sd[0]          # (24,22) the sheet in model units
print("the user's 24 tiles, in PCA units (first 4 components):")
print(f"  {'idx':>4} {'label':>8} {'PC0':>7} {'PC1':>7} {'PC2':>7} {'PC3':>7} {'|z|':>7}")
for i in range(24):
    lab = next((k for k, v in SHEET.items() if i in v), "?")
    print(f"  {i:4d} {lab:>8} {Z[i,0]:7.2f} {Z[i,1]:7.2f} {Z[i,2]:7.2f} {Z[i,3]:7.2f} "
          f"{np.linalg.norm(Z[i]):7.2f}")

print()
print("=" * 92)
print("1. IS THE ANIME SET CENTRED ON THE ORIGIN?")
print("=" * 92)
idx = np.array(SHEET["anime"])
Za = Z[idx]
allm = Z.mean(0)
print(f"  anime mean |z| = {np.linalg.norm(Za.mean(0)):.3f}   all-24 mean |z| = {np.linalg.norm(allm):.3f}")
print(f"  anime mean distance from origin per component: "
      f"{np.round(Za.mean(0)[:6], 2).tolist()} ...")
print(f"  mean norms: anime {np.linalg.norm(Za,axis=1).mean():.3f}  "
      f"real3d {np.linalg.norm(Z[SHEET['real3d']],axis=1).mean():.3f}  "
      f"game {np.linalg.norm(Z[SHEET['game']],axis=1).mean():.3f}  "
      f"unnamed {np.linalg.norm(Z[SHEET['unnamed']],axis=1).mean():.3f}")
print("  (an independent N(0,1)^22 draw has expected norm sqrt(22) = %.2f)" % np.sqrt(22))

print()
print("=" * 92)
print("2. ARE THE COMPONENTS CORRELATED WITHIN THE STYLE?")
print("=" * 92)
C = np.cov(Za.T)
off = C[~np.eye(22, dtype=bool)]
print(f"  anime: mean |off-diagonal correlation| {np.abs(off).mean():.3f}, "
      f"max {np.abs(off).max():.3f}")
Cg = np.cov(Z[SHEET['game']].T)
offg = Cg[~np.eye(22, dtype=bool)]
print(f"  game : mean |off-diagonal correlation| {np.abs(offg).mean():.3f}, "
      f"max {np.abs(offg).max():.3f}")
print(f"  all 24 faces: mean |off-diagonal| "
      f"{np.abs(np.cov(Z.T)[~np.eye(22,dtype=bool)]).mean():.3f}")

print()
print("=" * 92)
print("3. HOW MUCH OF THE STYLE DOES THE CURRENT SAMPLER HIT?")
print("=" * 92)
# Does an independent N(0,1) draw land inside the anime envelope?
# Envelope: Mahalanobis distance to the anime mean, with a REGULARISED covariance
# (10 points cannot determine a 22-dim covariance, so shrink hard towards identity).
alpha = 0.9
Creg = (1 - alpha) * C + alpha * np.eye(22)
Creg += 1e-6 * np.eye(22)
Ci = np.linalg.inv(Creg)

def maha(v, mean, Ci):
    d = v - mean
    return float(np.sqrt(d @ Ci @ d))

Za_m = Za.mean(0)
m_in = np.array([maha(z, Za_m, Ci) for z in Za])
print(f"  anime samples' own Mahalanobis radii: mean {m_in.mean():.2f}, "
      f"max {m_in.max():.2f}   (chi with 22 dof: mean ~4.7)")
# distribution of that radius under the CURRENT sampler (independent N(0,1))
rng = np.random.default_rng(0)
Ndraw = 200000
D = rng.standard_normal((Ndraw, 22))
bb = D - Za_m
mr = np.sqrt(np.einsum("ij,jk,ik->i", bb, Ci, bb))
thresh = np.percentile(m_in, 90)
frac_inside = float((mr <= thresh).mean())
# also: distance from a draw to the NEAREST anime sample, to compare "style hit rate"
d_near = np.sqrt(((D[:, None, :] - Za[None, :, :]) ** 2).sum(2)).min(1)
# the anime samples' own nearest-neighbour distances to each other
d_nn_anime = np.sqrt(((Za[:, None, :] - Za[None, :, :]) ** 2).sum(2))
np.fill_diagonal(d_nn_anime, 1e9)
d_nn_anime = d_nn_anime.min(1)
for q in (50, 75, 90):
    thr = np.percentile(d_nn_anime, q)
    print(f"  a random N(0,1)^22 draw is within the anime's own {q}th-percentile "
          f"nearest-neighbour distance ({thr:.2f}) in {float((d_near<=thr).mean())*100:.3f}% of draws")
print(f"  -> by the Mahalanobis envelope (90th pct of the anime radii), the current "
      f"sampler hits the style in {frac_inside*100:.3f}% of draws")

print()
print("=" * 92)
print("4. THE FITTED ENVELOPE (what the page should sample from)")
print("=" * 92)
# regularised mean and covariance, exported for the page
mean_out = Za_m
sd_out = np.sqrt(np.diag(Creg))
print(f"  anime mean (first 6): {np.round(mean_out[:6],3).tolist()}")
print(f"  anime sd   (first 6): {np.round(sd_out[:6],3).tolist()}")
print(f"  current sampler uses mean 0 and sd 1 for every component")
print(f"  -> the style is offset from the origin by |mean| = {np.linalg.norm(mean_out):.3f}")
# a Cholesky factor so the page can draw correlated samples
L = np.linalg.cholesky(Creg)
np.save(os.path.join(OUT, "anime_mean.npy"), mean_out.astype(np.float32))
np.save(os.path.join(OUT, "anime_chol.npy"), L.astype(np.float32))
json.dump({"style": "anime", "n_samples": int(len(idx)), "sheet_indices": SHEET["anime"],
           "mean": [float(v) for v in mean_out], "sd": [float(v) for v in sd_out],
           "chol": [[float(v) for v in r] for r in L],
           "regularisation_alpha": alpha,
           "sampler_hit_rate_maha": frac_inside,
           "caveat": "only 10 labelled samples; a 22-dim covariance is not identifiable "
                     "from 10 points, so the covariance is heavily shrunk towards the "
                     "identity and the envelope is a rough guide, not a fitted model"},
          open(os.path.join(OUT, "anime_envelope.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'anime_envelope.json')}")
print("ENVELOPE_DONE")
