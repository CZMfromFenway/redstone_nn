#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Round 11 statistics: the five-way style scheme, so "acceptable" means any of the three styles.

A NOTE ON THE SCHEME
The colour page labels with 类二次元 / 类游戏 / 类三次元 / 中 / 差, stored as grades 4/3/2/1/0.
Acceptable is grade >= 2. A first pass at these numbers counted only grades 2 and 3 and reported
50.5 / 30.1 / 23.4, which was wrong; the correct figures are below.

Fisher exact is implemented from the hypergeometric pmf directly, because scipy is not installed.
"""
import json
import math
import os
import sys
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]], summing the hypergeometric support."""
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    lo = max(0, r1 + c1 - n)
    hi = min(r1, c1)

    def logpmf(x):
        return (math.lgamma(r1 + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
                + math.lgamma(r2 + 1) - math.lgamma(c1 - x + 1) - math.lgamma(r2 - c1 + x + 1)
                - math.lgamma(n + 1) + math.lgamma(c1 + 1) + math.lgamma(n - c1 + 1))

    obs = logpmf(a)
    tot = 0.0
    for x in range(lo, hi + 1):
        lp = logpmf(x)
        if lp <= obs + 1e-9:
            tot += math.exp(lp)
    return min(1.0, tot)


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r = [x for x in recs if str(x.get("round") or "") == "11"]
ARMS = ("k8free", "k8snap", "k8mc")

print("=" * 96)
print(f"ROUND 11 -- {len(r)} blind labels, five-way style scheme")
print("=" * 96)
print(f"  {'arm':9s} {'n':>4s} {'差':>4s} {'中':>4s} {'三次元':>6s} {'游戏':>5s} "
      f"{'二次元':>6s} {'可接受':>8s} {'95% CI':>18s}")
D = {}
for m in ARMS:
    s = [x for x in r if x.get("model") == m]
    n = len(s)
    cnt = {g: sum(1 for x in s if x["grade"] == g) for g in range(5)}
    acc = cnt[2] + cnt[3] + cnt[4]
    lo, hi = wilson(acc, n)
    D[m] = (acc, n - acc, n)
    print(f"  {m:9s} {n:4d} {cnt[0]:4d} {cnt[1]:4d} {cnt[2]:6d} {cnt[3]:5d} "
          f"{cnt[4]:6d} {100*acc/n:7.1f}%  [{100*lo:5.1f}%, {100*hi:5.1f}%]")

print()
print("  PAIRWISE (two-sided Fisher exact on 可接受)")
print("  " + "-" * 74)
for x, y in combinations(ARMS, 2):
    a, b, _ = D[x]
    c, d, _ = D[y]
    pv = fisher(a, b, c, d)
    diff = 100 * (a / (a + b) - c / (c + d))
    print(f"    {x:8s} vs {y:8s}  {100*a/(a+b):5.1f}% vs {100*c/(c+d):5.1f}%  "
          f"({diff:+5.1f} 点)  p = {pv:.4f}  -> {'显著' if pv < 0.05 else '不显著'}")

print()
print("  STYLE MIX (share of all labels)")
print("  " + "-" * 74)
print(f"  {'arm':9s} {'类二次元':>8s} {'类游戏':>7s} {'类三次元':>8s} {'中':>6s} {'差':>6s}")
for m in ARMS:
    s = [x for x in r if x.get("model") == m]
    n = len(s)
    print(f"  {m:9s} " + " ".join(
        f"{100*sum(1 for x in s if x['grade']==g)/n:6.1f}%" for g in (4, 3, 2, 1, 0)))

print()
print("  ROUND 8, same scheme, for context only (cross-round comparison is not valid)")
print("  " + "-" * 74)
r8 = [x for x in recs if str(x.get("round") or "") == "8"]
for m in ("pal8", "pal16", "pal32"):
    s = [x for x in r8 if x.get("model") == m]
    if s:
        acc = sum(1 for x in s if x["grade"] >= 2)
        print(f"    {m:8s} {100*acc/len(s):5.1f}%  (n={len(s)})")

out = {"n": len(r), "arms": {m: {"acc": D[m][0], "n": D[m][2]} for m in ARMS},
       "pairwise": {f"{x}|{y}": fisher(D[x][0], D[x][1], D[y][0], D[y][1])
                    for x, y in combinations(ARMS, 2)}}
json.dump(out, open(os.path.join(ROOT, "results", "colour", "k8schemes", "round11.json"),
                    "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nROUND11_STATS_DONE")
