#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Export the quadratic scorer in the form the hardware needs.

WHY A QUADRATIC
The marginals from the 4096-point label map show PC1 is an inverted U: 15% bad at the low end,
51% in the middle, 31% at the top. No linear function can represent a hump, which is exactly why
the current 4 -> 16 -> 1 scorer stalls at AUC 0.694 while a 14-term quadratic reaches 0.853.

THE EXPORTED FORM
    score = c + sum_i a_i x_i + sum_i b_i x_i^2 + sum_{i<j} d_ij x_i x_j
with the feature standardisation folded into the coefficients, so the page feeds raw knobs.

COST
    4 squares, 6 products, 14 multiply-accumulates plus a bias. Cheaper than the 80 multiplies of
    the current 4-16-1 MLP, and 15 numbers fit in a tiny lookup table.
"""
import itertools
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
OUT = os.path.join(ROOT, "results", "pca4map")
WEB = os.path.join(ROOT, "webapp")
LEVELS = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])

j = json.load(open(os.path.join(OUT, "pca4_plane_marks.json"), encoding="utf-8"))
bad = np.zeros((8, 8, 8, 8), dtype=bool)
for m in j["marks"]:
    idx = [int(np.argmin(np.abs(LEVELS - v))) for v in m["pca"]]
    bad[tuple(idx)] = True
X = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, LEVELS, indexing="ij"), -1).reshape(-1, 4)
y = bad.reshape(-1).astype(float)
print(f"{len(y)} points, {100*y.mean():.1f}% bad")

PAIRS = list(itertools.combinations(range(4), 2))


def feats(x):
    return np.concatenate([x, x ** 2, np.stack([x[:, i] * x[:, k] for i, k in PAIRS], 1)], 1)


F = feats(X)
mu = F.mean(0); sd = F.std(0) + 1e-9
A = np.c_[(F - mu) / sd, np.ones(len(F))]
w = np.linalg.solve(A.T @ A + 3.0 * np.eye(A.shape[1]), A.T @ y)

# fold the standardisation back into raw-feature coefficients
raw = w[:-1] / sd
c = float(w[-1] - (w[:-1] * mu / sd).sum())
print(f"bias after folding: {c:+.5f}")

lin = raw[:4]
sq = raw[4:8]
cr = raw[8:]
coef = {"c": c,
        "linear": [float(v) for v in lin],
        "square": [float(v) for v in sq],
        "cross": {f"{i}{k}": float(v) for (i, k), v in zip(PAIRS, cr)}}

# verify the folded form reproduces the fitted score
def score(x):
    v = np.full(len(x), c)
    for i in range(4):
        v += lin[i] * x[:, i] + sq[i] * x[:, i] ** 2
    for (i, k), d in zip(PAIRS, cr):
        v += d * x[:, i] * x[:, k]
    return v


ref = A @ w
print(f"fold check: max |folded - fitted| = {np.abs(score(X) - ref).max():.2e}")


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


print(f"in-sample AUC  {auc(score(X), y):.3f}   (the honest number is the 5-fold 0.853)")
thr = float(np.quantile(score(X), 1 - y.mean()))
print(f"median-equivalent threshold {thr:+.4f} (keeps the same bad rate as the whole cube)")

out = {"type": "scorer_quad", "in_dim": 4, "target": "bad",
       "note": "higher score = more likely 中 (bad). Trained on the 4096-point lattice map.",
       "coef": coef, "thresholds": {"0.5": thr},
       "meta": {"n": int(len(y)), "bad_rate": float(y.mean()),
                "auc_cv": 0.853, "auc_insample": float(auc(score(X), y)),
                "mults": 14, "squares": 4, "cross": 6,
                "source": "results/pca4map/pca4_plane_marks.json"}}
fp = os.path.join(WEB, "scorer_quad.json")
json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nwrote {fp}  ({os.path.getsize(fp)/1e3:.1f} KB)")
print("QUAD_EXPORT_DONE")
