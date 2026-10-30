#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Round 12 statistics and the honest read.

Both attempts lost to the untouched baseline by a wide margin, even though each of them improved
the metrics they were designed to improve. That is worth stating plainly: the metrics are not a
proxy for the verdict, and every intervention that chased a metric has now been judged worse.

Fisher exact is implemented from the hypergeometric pmf because scipy is not installed.
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(a, b, c, d):
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)

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


recs = [json.loads(l) for l in open(os.path.join(ROOT, "results", "preference", "labels.jsonl"),
                                    encoding="utf-8") if l.strip()]
r = [x for x in recs if str(x.get("round") or "") == "12"]
ARMS = ("k8snap", "c2adv", "c1struct")
NAMES = {4: "类二次元", 3: "类游戏", 2: "类三次元", 1: "中", 0: "差"}

print("=" * 96)
print(f"ROUND 12 -- {len(r)} blind labels, three arms with the SAME palette")
print("=" * 96)
print(f"  {'arm':10s} {'n':>4s} " + " ".join(f"{NAMES[g]:>7s}" for g in (4, 3, 2, 1, 0))
      + f" {'可接受':>8s} {'95% CI':>17s}")
D = {}
for m in ARMS:
    s = [x for x in r if x.get("model") == m]
    n = len(s)
    c = {g: sum(1 for x in s if x["grade"] == g) for g in range(5)}
    acc = c[2] + c[3] + c[4]
    lo, hi = wilson(acc, n)
    D[m] = (acc, n - acc, n)
    print(f"  {m:10s} {n:4d} " + " ".join(f"{100*c[g]/n:6.1f}%" for g in (4, 3, 2, 1, 0))
          + f" {100*acc/n:7.1f}%  [{100*lo:5.1f}%, {100*hi:5.1f}%]")

print()
print("  PAIRWISE (two-sided Fisher exact on 可接受)")
print("  " + "-" * 76)
for x, y in combinations(ARMS, 2):
    a, b, _ = D[x]
    c, d, _ = D[y]
    pv = fisher(a, b, c, d)
    diff = 100 * (a / (a + b) - c / (c + d))
    print(f"    {x:9s} vs {y:9s}  {100*a/(a+b):5.1f}% vs {100*c/(c+d):5.1f}%  "
          f"({diff:+6.1f} 点)  p = {pv:.4g}  -> {'显著' if pv < 0.05 else '不显著'}")

# what each attempt bought on the metrics, versus what it cost in verdict
print()
print("=" * 96)
print("METRICS SAID ONE THING, THE VERDICT SAID THE OPPOSITE")
print("=" * 96)
met = {"k8snap": {"edge": 13.4, "ncol": 4.2, "eye": 3.8, "acc": 61.1},
       "c2adv": {"edge": 15.8, "ncol": 4.8, "eye": 4.7, "acc": 43.2},
       "c1struct": {"edge": 13.8, "ncol": 4.2, "eye": 9.2, "acc": 41.0}}
print(f"  {'arm':10s} {'边缘':>7s} {'颜色数':>7s} {'眼睛对比':>8s} {'可接受':>8s}")
print("  " + "-" * 46)
for m in ARMS:
    k = met[m]
    print(f"  {m:10s} {k['edge']:7.1f} {k['ncol']:7.1f} {k['eye']:8.2f} {k['acc']:7.1f}%")
print()
print("  c2adv 的边缘和颜色数都是最高的，眼睛对比也高于基线，可接受率却低 17.9 点。")
print("  c1struct 的眼睛对比是基线的 2.4 倍，可接受率低 20.1 点。")

json.dump({"n": len(r), "arms": {m: {"acc": D[m][0], "n": D[m][2]} for m in ARMS}},
          open(os.path.join(ROOT, "results", "colour", "yield", "round12.json"), "w",
               encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nROUND12_STATS_DONE")
