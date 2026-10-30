#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Style-conditioned purification: is asking for one style better than purifying in general?

THE STRATEGY
Rather than one global "acceptable" filter, let the user pick a style before generating and
push the sampler toward that style. The user's reasoning: if a global purifier yields 0.6 and
a style-specific one yields 0.7-0.8, the latter wins -- and that may matter more than the
choice between a linear and a quadratic reward.

WHY IT IS TESTABLE WITH WHAT WE HAVE
The five categories are mutually exclusive: a face is 类二次元, OR 类游戏, OR 类三次元, OR 中,
OR 差. So "the user asked for 类二次元" is satisfiable only by faces labelled 类二次元, and
those are acceptable by definition. The question therefore has two parts:
    style hit rate   of the faces the sampler produces, how many land in the requested class
    purity           within the targeted region, how much 中/差 still leaks in

Each style already has a linear probe (类二次元 AUC 0.896, 类三次元 0.890, 类游戏 0.735), so
selecting by that probe models "push the sampler toward this style". Everything is measured
with 5-fold cross-validated scores so no label is scored by a model that saw it.
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float64)
g = np.array([r["grade"] for r in new])
acc = (g >= 2).astype(float)
zm = X.mean(0); zsd = X.std(0) + 1e-9
Xs = (X - zm) / zsd
idx = np.random.default_rng(0).permutation(len(Xs))


def cv_score(y, lam=3.0):
    pred = np.zeros(len(Xs))
    for f in range(5):
        te = idx[f::5]
        tr = np.setdiff1d(idx, te)
        A = np.c_[Xs[tr], np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.c_[Xs[te], np.ones(len(te))] @ w
    return pred


print("=" * 104)
print(f"STYLE-CONDITIONED SELECTION  ({len(new)} labels, 5-fold CV scores)")
print("=" * 104)
print(f"  overall: {100*acc.mean():.1f}% acceptable, "
      f"类二次元 {100*(g==4).mean():.1f}% / 类游戏 {100*(g==3).mean():.1f}% / "
      f"类三次元 {100*(g==2).mean():.1f}% / 中差 {100*(g<=1).mean():.1f}%")

styles = [(4, "类二次元"), (3, "类游戏"), (2, "类三次元")]
global_pred = cv_score(acc)
res = {}

for target, name in styles:
    ys = (g == target).astype(float)
    if ys.sum() < 5:
        print(f"\n  {name}: only {int(ys.sum())} positives, skipped")
        continue
    sp = cv_score(ys)
    a_style = auc(sp, ys)
    print()
    print("=" * 104)
    print(f"ASKING FOR {name}  (linear probe cv AUC {a_style:.3f}, "
          f"{int(ys.sum())} positives of {len(ys)})")
    print("=" * 104)
    print(f"  {'keep':>6s} {'n':>4s} {'命中该风格':>10s} {'可接受':>8s} {'中差':>7s} "
          f"{'风格纯度':>8s}")
    print("  " + "-" * 52)
    rows = {}
    for keep in (1.0, 0.5, 0.35, 0.25, 0.15):
        k = max(1, int(len(ys) * keep))
        top = np.argsort(-sp)[:k]
        hit = float(ys[top].mean())
        a = float(acc[top].mean())
        bad = 1 - a
        rows[keep] = {"n": k, "hit": hit, "acc": a, "bad": bad}
        print(f"  {keep*100:5.0f}% {k:4d} {100*hit:9.1f}% {100*a:7.1f}% {100*bad:6.1f}% "
              f"{100*hit/max(float((g==target).mean()),1e-9):7.1f}%")
    res[name] = {"auc": a_style, "rows": {str(k2): v for k2, v in rows.items()}}

# ---------------------------------------------------------------- global for comparison
print()
print("=" * 104)
print("FOR COMPARISON: ONE GLOBAL 'ACCEPTABLE' FILTER")
print("=" * 104)
print(f"  {'keep':>6s} {'n':>4s} {'可接受':>8s} {'类二次元':>9s} {'类游戏':>8s} "
      f"{'类三次元':>9s} {'中差':>7s}")
print("  " + "-" * 60)
glob = {}
for keep in (1.0, 0.5, 0.35, 0.25):
    k = max(1, int(len(acc) * keep))
    top = np.argsort(-global_pred)[:k]
    c = {gg: float((g[top] == gg).mean()) for gg in (4, 3, 2, 1, 0)}
    glob[keep] = {"n": k, "acc": float(acc[top].mean()), **{STYLE[gg]: c[gg] for gg in c}}
    print(f"  {keep*100:5.0f}% {k:4d} {100*acc[top].mean():7.1f}% "
          f"{100*c[4]:8.1f}% {100*c[3]:7.1f}% {100*c[2]:8.1f}% {100*(c[1]+c[0]):6.1f}%")

print()
print("=" * 104)
print("VERDICT")
print("=" * 104)
for name, v in res.items():
    k35 = v["rows"]["0.35"]
    base = float((g == [k for k, n2 in styles if n2 == name][0]).mean())
    print(f"  ask for {name}: at keep 35%, {100*k35['hit']:.0f}% of what you get is that "
          f"style (base {100*base:.0f}%), and {100*k35['acc']:.0f}% is acceptable")
print()
print("  A global filter cannot choose the style: at keep 35% it returns a fixed mix.")
print("  Style conditioning makes the MIX controllable, which is a different kind of win")
print("  from a higher acceptable rate.")

json.dump({"styles": res, "global": {str(k): v for k, v in glob.items()}},
          open(os.path.join(OUT, "style_conditioned.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'style_conditioned.json')}")
print("STYLE_CONDITIONED_DONE")
