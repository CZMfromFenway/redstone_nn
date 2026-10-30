#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Did the old 优/良 reward model quietly prefer one visual style?

THE SUSPICION
The user labelled a fresh batch with STYLE categories instead of quality grades, because they
suspect their earlier 优/良 grading pushed the reward model toward a specific look. This is a
testable claim now: the old reward was fitted on quality grades, and the new batch carries
style labels for faces from the same generator. If the old reward systematically scores
类二次元 higher, the suspicion is confirmed.

WHAT IS COMPARED
    old reward score   the linear model fitted on the 274 earlier dim4x labels (优+良 = good)
    style             类二次元 / 类游戏 / 类三次元 (all acceptable) / 中 / 差
If the old score rises monotonically from 差 up to 类二次元, then "quality" as the old labels
defined it was partly a proxy for "looks anime", exactly what the user feared.
"""
import collections
import json
import math
import os
import sys
import numpy as np
from datetime import datetime

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
ORDER = [4, 3, 2, 1, 0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]

# ---------------------------------------------------------------- the new style batch
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
print("=" * 100)
print(f"STYLE BATCH -- {len(new)} labels")
print("=" * 100)
c = collections.Counter(r["grade"] for r in new)
print(f"  {'category':10s} {'n':>4s} {'share':>7s} {'95% CI':>17s}   verdict")
print("  " + "-" * 58)
tot = len(new)
for g in ORDER:
    k = c[g]
    lo, hi = wilson(k, tot)
    verdict = "可接受" if g >= 2 else "要避免"
    print(f"  {STYLE[g]:10s} {k:4d} {100*k/tot:6.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%]   {verdict}")
acc = sum(c[g] for g in (2, 3, 4))
lo, hi = wilson(acc, tot)
print("  " + "-" * 58)
print(f"  可接受合计   {acc:4d} {100*acc/tot:6.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%]")

print()
print("  the generator's output is dominated by ONE style:")
print(f"    类二次元 {100*c[4]/tot:.1f}%   类游戏 {100*c[3]/tot:.1f}%   "
      f"类三次元 {100*c[2]/tot:.1f}%")
print(f"    ratio 类二次元 : 类三次元 = {c[4]/max(c[2],1):.1f} : 1")

# ---------------------------------------------------------------- old reward vs style
old = [r for r in recs if r.get("model") == "dim4x" and str(r.get("round") or "") in ("2", "3", "4")
       and r.get("grade") in (0, 1, 2, 3)]
Xo = np.array([r["pca"][:4] for r in old], dtype=np.float64)
yo = np.array([r["grade"] for r in old], dtype=np.float64)
zm = Xo.mean(0); zsd = Xo.std(0) + 1e-9
Xos = (Xo - zm) / zsd
A = np.c_[Xos, np.ones(len(Xos))]
LAM = 30.0
wz = np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ yo)
w4 = wz[:4] / zsd
b4 = float(wz[4] - (w4 * zm).sum())
print()
print("=" * 100)
print("THE OLD 优/良 REWARD, SCORING THE NEW STYLE LABELS")
print("=" * 100)
print(f"  fitted on {len(old)} earlier labels, weights "
      f"{', '.join(f'{v:+.3f}' for v in w4)}")

Zs = np.array([r["pca"][:4] for r in new], dtype=np.float64)
sc = Zs @ w4 + b4
print()
print(f"  {'category':10s} {'n':>4s} {'mean old score':>15s} {'sd':>7s}")
print("  " + "-" * 42)
by = {}
for g in ORDER:
    m = np.array([r["grade"] for r in new]) == g
    if m.sum() == 0:
        continue
    by[g] = {"n": int(m.sum()), "mean": float(sc[m].mean()), "sd": float(sc[m].std())}
    print(f"  {STYLE[g]:10s} {m.sum():4d} {sc[m].mean():15.3f} {sc[m].std():7.3f}")

# does the old score rise with "anime-ness"?
anime = np.array([r["grade"] for r in new]) == 4
other_ok = np.isin(np.array([r["grade"] for r in new]), [2, 3])
if anime.sum() >= 5 and other_ok.sum() >= 5:
    d = sc[anime].mean() - sc[other_ok].mean()
    # Welch t
    va, vb = sc[anime].var(ddof=1), sc[other_ok].var(ddof=1)
    se = math.sqrt(va / anime.sum() + vb / other_ok.sum())
    t = d / se if se > 0 else 0.0
    p = math.erfc(abs(t) / math.sqrt(2))
    print()
    print(f"  类二次元 vs 其他可接受风格: 旧分差 {d:+.3f}  (t = {t:.2f}, p = {p:.4f})")
    if p < 0.05 and d > 0:
        print("  -> 旧奖励模型给类二次元更高的分。你的怀疑成立：")
        print("     「优」在旧标签里有一部分就是「像二次元」。")
    elif p < 0.05 and d < 0:
        print("  -> 旧奖励模型反而给类二次元更低的分，与怀疑相反。")
    else:
        print("  -> 没有显著差异，旧奖励模型不是单纯在追某种风格。")

# ---------------------------------------------------------------- is 中 a separate thing?
print()
print("=" * 100)
print("WHERE DOES 中 SIT?")
print("=" * 100)
print("  中 is the ambiguous bucket. If its old score sits between the styles and 差, it is")
print("  really 'meh quality'. If it looks like a style, it is a style the user does not have")
print("  a name for and should stay acceptable.")
for g in ORDER:
    if g in by:
        print(f"    {STYLE[g]:10s} old score {by[g]['mean']:+.3f}")

json.dump({"counts": {STYLE[g]: c[g] for g in ORDER}, "total": tot,
           "acceptable": acc, "old_reward_by_style": {STYLE[g]: by[g] for g in by}},
          open(os.path.join(OUT, "style5_summary.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'style5_summary.json')}")
print("STYLE5_DONE")
