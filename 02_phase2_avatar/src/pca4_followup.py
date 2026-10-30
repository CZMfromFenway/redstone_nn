#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Two follow-ups the first pass made necessary.

1  THE BAFFLE: a ball-shaped sampling region barely helps, 33.5% -> 32.8%. The good set turns out
   to be ONE connected blob of 66.5%, not several tight balls, so shrinking to a ball mostly
   throws away variety without avoiding the bad region. What actually selects well is a BOX
   restriction per knob, because the marginals are strongly structured: PC2 alone runs 54% bad at
   -3.5 down to 19% at +3.5. A box is also the natural thing to implement: clamp the generator.

2  THE SCORER SHAPE: the linear probe caps at AUC 0.694 while a 12-ball nearest-centre model
   reaches 0.776. The marginals show why -- PC1 is an inverted U (15% bad at the low end, 51% in
   the middle, 31% at the top), which no linear function can represent. A QUADRATIC form can, and
   costs far less than twelve balls. That is tested here.
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
LEVELS = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])

j = json.load(open(os.path.join(OUT, "pca4_plane_marks.json"), encoding="utf-8"))
bad = np.zeros((8, 8, 8, 8), dtype=bool)
for m in j["marks"]:
    idx = [int(np.argmin(np.abs(LEVELS - v))) for v in m["pca"]]
    bad[tuple(idx)] = True
X = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, LEVELS, indexing="ij"), -1).reshape(-1, 4)
y = bad.reshape(-1).astype(float)
N = len(y)
print(f"{N} 点，坏 {100*y.mean():.1f}%")


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


def cv(feat, y, lam=3.0, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    s = np.zeros(len(y))
    F = feat(X)
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[q] for q in range(k) if q != f])
        mu = F[tr].mean(0); sd = F[tr].std(0) + 1e-9
        A = np.c_[(F[tr] - mu) / sd, np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        s[te] = np.c_[(F[te] - mu) / sd, np.ones(len(te))] @ w
    return auc(s, y), s


def f_lin(x): return x
def f_quad(x):
    cols = [x]
    cols.append(x ** 2)
    cross = [x[:, i] * x[:, k] for i, k in itertools.combinations(range(4), 2)]
    if cross:
        cols.append(np.stack(cross, 1))
    return np.concatenate(cols, 1)


a_lin, _ = cv(f_lin, y)
a_quad, s_quad = cv(f_quad, y)
print()
print("=" * 96)
print("SCORER SHAPE")
print("=" * 96)
print(f"  线性      4 项           AUC {a_lin:.3f}   乘法约 4")
print(f"  二次      14 项          AUC {a_quad:.3f}   乘法约 14（4 个平方 + 6 个乘积 + 权重）")
print(f"  最近中心  12 球（前一次）  AUC 0.776   乘法约 48 + 12 次比较")
print()
print(f"  二次比线性高 {100*(a_quad-a_lin):+.1f} AUC 点，成本只多 10 次乘法。")

# which quadratic terms matter
F = f_quad(X)
mu = F.mean(0); sd = F.std(0) + 1e-9
A = np.c_[(F - mu) / sd, np.ones(len(F))]
w = np.linalg.solve(A.T @ A + 3 * np.eye(A.shape[1]), A.T @ y)
names = ["PC0", "PC1", "PC2", "PC3",
         "PC0²", "PC1²", "PC2²", "PC3²",
         "PC0·PC1", "PC0·PC2", "PC0·PC3", "PC1·PC2", "PC1·PC3", "PC2·PC3", "b"]
order = np.argsort(-np.abs(w))
print()
print("  二次模型的权重（按绝对值排序，越大越重要）：")
for i in order[:8]:
    print(f"    {names[i]:9s} {w[i]:+.4f}")

# ---- box restriction
print()
print("=" * 96)
print("BOX RESTRICTION: clamp each knob to a range before sampling")
print("=" * 96)
print(f"  {'限制':46s} {'覆盖':>7s} {'坏率':>7s} {'降幅':>7s}")
print("  " + "-" * 72)
base = y.mean()


def box_rate(lo, hi):
    m = np.ones(N, bool)
    for k in range(4):
        m &= (X[:, k] >= LEVELS[lo[k]]) & (X[:, k] <= LEVELS[hi[k]])
    return m


cands = []
for lo in itertools.product(range(8), repeat=4):
    pass
# single-axis thresholds
for k in range(4):
    for lo in range(8):
        m = box_rate([0,0,0,0], [7,7,7,7])
        m = (X[:, k] >= LEVELS[lo])
        cands.append((y[m].mean(), m.mean(), f"PC{k} ≥ {LEVELS[lo]:+.1f}"))
    for hi in range(8):
        m = (X[:, k] <= LEVELS[hi])
        cands.append((y[m].mean(), m.mean(), f"PC{k} ≤ {LEVELS[hi]:+.1f}"))
# two-axis: one lower bound on PC2, one range on PC1
for p2lo in range(8):
    for p1lo in range(8):
        m = (X[:, 2] >= LEVELS[p2lo]) & (X[:, 1] >= LEVELS[p1lo])
        cands.append((y[m].mean(), m.mean(),
                      f"PC2 ≥ {LEVELS[p2lo]:+.1f} 且 PC1 ≥ {LEVELS[p1lo]:+.1f}"))
for p2lo in range(8):
    for p1hi in range(8):
        m = (X[:, 2] >= LEVELS[p2lo]) & (X[:, 1] <= LEVELS[p1hi])
        cands.append((y[m].mean(), m.mean(),
                      f"PC2 ≥ {LEVELS[p2lo]:+.1f} 且 PC1 ≤ {LEVELS[p1hi]:+.1f}"))

# keep only rules that retain at least 20% of the space, then sort by bad rate
cands = [c for c in cands if c[1] >= 0.20]
cands.sort(key=lambda t: t[0])
for br, cov, name in cands[:14]:
    print(f"  {name:46s} {100*cov:6.1f}% {100*br:6.1f}% {100*(base-br):+6.1f}")

print()
print("  只看保留 ≥20% 空间、坏率最低的规则。")

# the best few, with what they mean for yield
print()
print("  换算成良率（坏率取反）：")
for br, cov, name in cands[:5]:
    print(f"    {name:44s} 良率 {100*(1-br):.1f}%  多样性保留 {100*cov:.0f}%")

json.dump({"auc_linear": a_lin, "auc_quad": a_quad,
           "quad_weights": {names[i]: float(w[i]) for i in range(len(names))},
           "box_rules": [{"rule": n, "coverage": float(c), "bad": float(b)}
                         for b, c, n in cands[:14]]},
          open(os.path.join(OUT, "scorer_shapes.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'scorer_shapes.json')}")
print("FOLLOWUP_DONE")
