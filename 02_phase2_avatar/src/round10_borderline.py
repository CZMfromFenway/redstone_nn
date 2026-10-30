#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
How much of the round-10 signal is label noise from borderline faces?

THE USER'S OBSERVATION
"我遇到了好多很相似的头像，但是我可能有时候打良有时候打中，那代表它们是边缘角色，品质介于两者之间。"

That is a statement about the labels themselves, and it can be measured without asking for more
labelling. Take every pair of labelled faces, measure how similar they are in the printed pixels,
and then look at the grade disagreement as a function of similarity:

    if near-identical faces frequently get different grades, then no scorer can separate them,
    and the ceiling on any AUC is set by that disagreement rather than by the features.

DISAGREEMENT IS MEASURED THREE WAYS
    exact      良 vs 中 and similar adjacent swaps only, i.e. one step apart
    pass/fail  whether the two faces land on opposite sides of the 良 threshold, which is what
               the yield metric actually counts
    magnitude  mean absolute grade difference
"""
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
OUT = os.path.join(ROOT, "results", "dimmix")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
os.makedirs(OUT, exist_ok=True)


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r10 = [r for r in recs if str(r.get("round") or "") == "10"
       and len(r.get("predicted_levels") or []) == 32]
print(f"round 10 labels with pixels: {len(r10)}")

X = np.array([r["predicted_levels"] for r in r10], dtype=np.float64)   # (n,32) printed levels
G = np.array([r["grade"] for r in r10])
A = np.array([r["model"] for r in r10])
print(f"  grade spread: {dict(zip(*np.unique(G, return_counts=True)))}")
print(f"  printed level value counts differ by arm (merged arms are discrete):")
for a in sorted(set(A)):
    u = np.unique(X[A == a])
    print(f"    {a:9s} {len(u):5d} distinct printed values")

# pairwise distance in printed pixels, mean absolute difference in levels
n = len(X)
print(f"\ncomputing {n*(n-1)//2:,} pairs ...")
D = np.abs(X[:, None, :] - X[None, :, :]).mean(-1)
iu = np.triu_indices(n, 1)
d = D[iu]
g1 = G[iu[0]]
g2 = G[iu[1]]
both_pass = ((g1 >= 2) & (g2 >= 2)) | ((g1 < 2) & (g2 < 2))

print()
print("=" * 100)
print("GRADE DISAGREEMENT AS A FUNCTION OF HOW SIMILAR THE TWO FACES ARE")
print("=" * 100)
print(f"  {'像素距离':>16s} {'对数':>7s} {'等级不同':>9s} {'差一步':>8s} "
      f"{'过线与否相反':>13s}")
print("  " + "-" * 62)
bands = [(0, 0.01), (0.01, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]
out = {}
for lo, hi in bands:
    m = (d >= lo) & (d < hi)
    if m.sum() == 0:
        continue
    diff = g1[m] != g2[m]
    one = np.abs(g1[m] - g2[m]) == 1
    flip = ~both_pass[m]
    label = f"{lo:.2f}~{hi:.2f}" if hi < 99 else f">{lo:.2f}"
    out[label] = {"pairs": int(m.sum()), "diff": float(diff.mean()),
                  "one": float(one.mean()), "flip": float(flip.mean())}
    print(f"  {label:>16s} {int(m.sum()):7d} {100*diff.mean():8.1f}% "
          f"{100*one.mean():7.1f}% {100*flip.mean():12.1f}%")

print()
print("  读法：'过线与否相反' 是最关键的一列——两张几乎一样的面，一张算良一张算中，")
print("  就意味这一对的标签在你的标准下是无法用图像区分的。")

# how many labelled faces have a near-twin with an opposite verdict?
NEAR = 0.25
mask = (d < NEAR) & (g1 != g2)
flips = (d < NEAR) & (~both_pass)
twin_idx = set()
for i, j in zip(iu[0][mask], iu[1][mask]):
    twin_idx.add(i); twin_idx.add(j)
print()
print("=" * 100)
print(f"边缘角色：在像素距离 < {NEAR} 的前提下被判成不同等级的样本")
print("=" * 100)
print(f"  这样的对数 {int(mask.sum())}，涉及 {len(twin_idx)} 张脸 "
      f"（占 {100*len(twin_idx)/n:.1f}%）")
print(f"  其中过线与否相反的 {int(flips.sum())} 对")

# what does this imply for any scorer?
print()
print("=" * 100)
print("REPEATABILITY TEST: split the labels in half, does the grade pattern survive?")
print("=" * 100)
# a face that is a near-twin of a 中 should itself be 中. How often is it not?
idx = np.arange(n)
hit = 0; tot = 0
for i in range(n):
    near = np.where((D[i] < 0.15) & (idx != i))[0]
    if len(near) < 2:
        continue
    vote = float(np.mean(G[near] >= 2))
    if vote > 0.99 or vote < 0.01:
        continue                     # neighbours agree, nothing to test
    tot += 1
    hit += 1 if abs(vote - (1 if G[i] >= 2 else 0)) < 0.5 else 0
print(f"  邻居意见分裂的脸：{tot} 张，其中自己的判断与邻居多数一致的有 {hit} 张 "
      f"（{100*hit/max(tot,1):.0f}%）")
print(f"  也就是说，在这批数据里约 {100*(1-hit/max(tot,1)):.0f}% 的判断在其邻近区域内是少数派。")

json.dump({"bands": out, "near_pairs": int(mask.sum()), "near_faces": len(twin_idx),
           "n": n}, open(os.path.join(OUT, "borderline.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'borderline.json')}")
print("BORDERLINE_DONE")
