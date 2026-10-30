#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
What the secondary filter actually did, and a fair comparison at the same budget.

WHAT THE PREVIOUS HEADROOM RUN DID -- stated plainly
It did NOT generate new faces. It re-selected subsets of the 153 already-labelled faces:

    1. take the 153 recorded 4-knob coordinates (exactly the ones used to generate those faces)
    2. give each a STYLE score, computed by 5-fold cross-validation: split the 153 into five
       folds, fit the linear style probe on four, score the fifth. Every face gets a score
       from a model that never saw its label.
    3. keep the top 50% by that style score   (76 faces) --> this models "user asked for style X"
    4. give each a NONLINEAR acceptability score, again 5-fold cross-validated
    5. inside the 76, keep the top half by that score   (38 faces)
    6. count how many of those 38 the user had labelled acceptable

Because the 153 ARE a random draw of what the generator produces, re-selecting from them is a
faithful simulation of "generate a pool, score it, keep the best" -- at the same cost, since
keeping 38 of 153 costs the same number of generations either way.

THE CATCH THIS RUN FIXES
The previous comparison was not at equal budget: the global filter kept 35% (53 faces) while
the two-stage kept 38. This repeats everything at exactly 38 kept, so the strategies are
directly comparable.
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
STYLE = {4: "类二次元", 3: "类游戏", 2: "类三次元", 1: "中", 0: "差"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float64)
g = np.array([r["grade"] for r in new])
acc = (g >= 2).astype(float)
zm = X.mean(0); zsd = X.std(0) + 1e-9
Xs = (X - zm) / zsd
idx = np.random.default_rng(0).permutation(len(Xs))
N = len(Xs)


def cv_linear(y, lam=3.0):
    pr = np.zeros(N)
    for f in range(5):
        te = idx[f::5]
        tr = np.setdiff1d(idx, te)
        A = np.c_[Xs[tr], np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pr[te] = np.c_[Xs[te], np.ones(len(te))] @ w
    return pr


def quad(P):
    d = P.shape[1]
    F = [P]
    for i in range(d):
        F.append((P[:, i] ** 2)[:, None])
        for j in range(i + 1, d):
            F.append((P[:, i] * P[:, j])[:, None])
    return np.hstack(F)


Zc = X - X.mean(0)
Fq = quad(Zc)
Fz = (Fq - Fq.mean(0)) / (Fq.std(0) + 1e-12)      # standardising the quadratic features
Aq = np.c_[Fz, np.ones(len(Fz))]
q_pred = np.zeros(N)
for f in range(5):
    te = idx[f::5]
    tr = np.setdiff1d(idx, te)
    A = np.c_[Fz[tr], np.ones(len(tr))]
    w = np.linalg.solve(A.T @ A + 30.0 * np.eye(A.shape[1]), A.T @ acc[tr])
    q_pred[te] = np.c_[Fz[te], np.ones(len(te))] @ w

hi = q_pred[acc == 1][:, None]; lo = q_pred[acc == 0][None, :]
print("=" * 100)
print("THE TWO SCORES")
print("=" * 100)
print(f"  acceptability score (quadratic, standardised features): cv AUC "
      f"{(hi>lo).mean()+0.5*(hi==lo).mean():.3f}")

KEEP = 38          # equal budget for every strategy
print()
print("=" * 100)
print(f"EQUAL BUDGET: every strategy keeps exactly {KEEP} faces out of {N}")
print("=" * 100)
print(f"  {'strategy':34s} {'可接受':>8s} {'95% CI':>17s} {'类二次元':>9s} {'中差':>7s}")
print("  " + "-" * 80)


def report(name, sel):
    a = float(acc[sel].mean())
    lo_, hi_ = wilson(int(acc[sel].sum()), len(sel))
    c4 = float((g[sel] == 4).mean())
    bad = 1 - a
    print(f"  {name:34s} {100*a:7.1f}% [{lo_*100:5.1f}%,{hi_*100:5.1f}%] "
          f"{100*c4:8.1f}% {100*bad:6.1f}%")
    return a


base = float(acc.mean())
print(f"  {'no filtering (keep all 153)':34s} {100*base:7.1f}%")
report("GLOBAL: top by acceptability only",
       np.argsort(-q_pred)[:KEEP])
for t, nm in ((4, "类二次元"), (3, "类游戏"), (2, "类三次元")):
    sp = cv_linear((g == t).astype(float))
    region = np.argsort(-sp)[:int(N * 0.5)]
    sel = region[np.argsort(-q_pred[region])[:KEEP]]
    report(f"STYLE {nm} then acceptability", sel)
print()
print("  Also, for reference: each style probe on its own, no acceptability filter")
for t, nm in ((4, "类二次元"), (3, "类游戏"), (2, "类三次元")):
    sp = cv_linear((g == t).astype(float))
    report(f"STYLE {nm} only", np.argsort(-sp)[:KEEP])

print()
print("=" * 100)
print("WHAT THIS MEANS FOR THE COST")
print("=" * 100)
print(f"  every row above spends the same generation budget: {N} faces generated to keep {KEEP},")
print(f"  i.e. about {N/KEEP:.1f} generated per kept face. Filters do not make generation cheaper;")
print("  they decide WHICH of the generated faces survive.")
print()
print("  The user's criterion was: the style that comes out does not matter as long as the")
print("  acceptable rate improves. Under that criterion, compare the 可接受 column only.")
json.dump({"keep": KEEP, "base": base},
          open(os.path.join(OUT, "equal_budget.json"), "w"), indent=2)
print("EQUAL_BUDGET_DONE")
