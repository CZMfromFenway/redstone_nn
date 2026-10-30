#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Two things the user asked for.

1. IS THE ANIME DOMINANCE IN THE DATA, RATHER THAN IN THE MODEL?
The user's point: the source skin set is itself mostly anime-styled, so a sampler reproducing
it faithfully will produce mostly anime, and blaming the model is wrong. This is checkable:
project every REAL face onto the 类二次元 probe direction and see where the mass sits. If real
faces already sit on the anime side, the dominance is inherited.

2. AFTER LINEARLY TARGETING A STYLE, HOW MUCH DOES A NONLINEAR FILTER STILL ADD?
Inside a style region most faces are acceptable by definition, but the user notes 中/差 also
occur there. So: fix a style region with the linear probe, measure its acceptable rate, then
apply the quadratic acceptability score INSIDE that region and see how much it lifts.
That is the headroom left for the second stage.
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
os.makedirs(OUT, exist_ok=True)
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
acc = (g >= 2).astype(float)
zm = X.mean(0); zsd = X.std(0) + 1e-9

# ---- probes on the labelled batches
Xs = (X - zm) / zsd
idx = np.random.default_rng(0).permutation(len(Xs))


def cv_score(y, lam=3.0):
    pr = np.zeros(len(Xs))
    for f in range(5):
        te = idx[f::5]
        tr = np.setdiff1d(idx, te)
        A = np.c_[Xs[tr], np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pr[te] = np.c_[Xs[te], np.ones(len(te))] @ w
    return pr


def probe(y, lam=3.0):
    A = np.c_[Xs, np.ones(len(Xs))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return w


w_anime = probe((g == 4).astype(float))
w_play = probe((g == 3).astype(float))
w_real = probe((g == 2).astype(float))
print("=" * 104)
print("1. IS THE ANIME DOMINANCE INHERITED FROM THE DATA?")
print("=" * 104)


def proj(w, Z):
    return (((Z - zm) / zsd) @ w[:4] + w[4])


sc_gen = proj(w_anime, X)
gen_anime_rate = float((g == 4).mean())

# find the score at which the generated faces are 50/50 anime vs rest
thr = float(np.median(sc_gen[g == 4]))

# real faces: their own 4 knobs are the first 4 PCA coefficients, already standardised
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
Xr = full.reshape(len(full), -1).astype(np.float64)
mu = Xr.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(Xr), min(60000, len(Xr)), replace=False)
_, _, Vt = np.linalg.svd(Xr[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((Xr - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
nr = 200000
br = np.random.default_rng(1).choice(len(Xr), nr, replace=False)
Zreal = (((Xr[br] - mu) @ comps.T) / sd[0])[:, :4]
sc_real = proj(w_anime, Zreal)

print(f"  generated faces: {100*gen_anime_rate:.1f}% read as 类二次元")
print(f"  the anime probe score on GENERATED faces: "
      f"类二次元 {sc_gen[g==4].mean():+.3f}  vs  others {sc_gen[g!=4].mean():+.3f}")
print()
print(f"  REAL faces (n={nr:,}), projected on the same axis:")
print(f"    mean {sc_real.mean():+.3f}   sd {sc_real.std():.3f}")
q = np.percentile(sc_real, [5, 25, 50, 75, 95])
print("    percentiles 5/25/50/75/95: " + "  ".join(f"{v:+.2f}" for v in q))
# what fraction of real faces sit where the anime-labelled generated ones sit?
hi_real = float(np.mean(sc_real > thr))
print(f"    share above the anime median threshold ({thr:+.3f}): {100*hi_real:.1f}%")
print()
print("  If a large share of REAL faces project onto the anime side, the dominance is")
print("  inherited from the source data, not invented by the sampler.")

# per-knob position of real faces vs the anime direction
print()
print(f"  {'knob':10s} {'real mean':>10s} {'anime weight':>13s} {'real sd':>8s}")
print("  " + "-" * 44)
for i in range(4):
    print(f"  {KNOB[i]:10s} {Zreal[:,i].mean():+10.3f} {w_anime[:4][i]/zsd[i]:+13.3f} "
          f"{Zreal[:,i].std():8.3f}")

# ---------------------------------------------------------------- 2. headroom inside a style
print()
print("=" * 104)
print("2. INSIDE A TARGETED STYLE, HOW MUCH DOES A NONLINEAR FILTER STILL ADD?")
print("=" * 104)
gp = cv_score(acc)


def quad_features(P):
    d = P.shape[1]
    F = [P]
    for i in range(d):
        F.append((P[:, i] ** 2)[:, None])
        for j in range(i + 1, d):
            F.append((P[:, i] * P[:, j])[:, None])
    return np.hstack(F)


Zc = X - X.mean(0)
F = quad_features(Zc)
Aq = np.c_[F, np.ones(len(F))]
wq = np.linalg.solve(Aq.T @ Aq + 30.0 * np.eye(Aq.shape[1]), Aq.T @ acc)
q_pred = np.zeros(len(Xs))
for f in range(5):
    te = idx[f::5]
    tr = np.setdiff1d(idx, te)
    A = np.c_[F[tr], np.ones(len(tr))]
    w = np.linalg.solve(A.T @ A + 30.0 * np.eye(A.shape[1]), A.T @ acc[tr])
    q_pred[te] = np.c_[F[te], np.ones(len(te))] @ w

print(f"  the nonlinear acceptability score has cv AUC {auc(q_pred, acc):.3f}")
print()
print("  step 1: take the top 50% by the style probe (this models 'user asked for this style')")
print("  step 2: inside that, keep the top half by the acceptability score")
print()
print(f"  {'style':10s} {'region n':>9s} {'region acc':>11s} {'after filter':>13s} "
      f"{'lift':>7s} {'中差 left':>10s}")
print("  " + "-" * 66)
out = {}
for t, nm in ((4, "类二次元"), (3, "类游戏"), (2, "类三次元")):
    sp = cv_score((g == t).astype(float))
    k = int(len(Xs) * 0.5)
    region = np.argsort(-sp)[:k]
    base = float(acc[region].mean())
    k2 = max(1, k // 2)
    keep = region[np.argsort(-q_pred[region])[:k2]]
    after = float(acc[keep].mean())
    bad = 1 - after
    out[nm] = {"region_n": int(k), "base": base, "after": after, "lift": after - base}
    print(f"  {nm:10s} {k:9d} {100*base:10.1f}% {100*after:12.1f}% "
          f"{100*(after-base):+6.1f} {100*bad:9.1f}%")

print()
print("  reading: 'region acc' is what the style probe alone already gives. 'after filter' is")
print("  what the acceptability model adds on top. The gap is the headroom of the second stage.")

json.dump({"anime_dominance": {"generated_rate": gen_anime_rate,
                               "real_mean_score": float(sc_real.mean()),
                               "real_share_above_thr": hi_real},
           "headroom": out},
          open(os.path.join(OUT, "headroom.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'headroom.json')}")
print("HEADROOM_DONE")
