#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
The clean test: guided vs explore INSIDE round 2.

Why this is the right comparison and round-1-vs-round-2 is not:

The user says their standard rose between rounds ("我相对保守，标准有了一个整体的提高，
几乎没打优"). That is visible in the data: round 1's plain samples scored 40.5% good, while
round 2's explore samples scored only 21.1%. Comparing round 2's guided faces against
round 1's plain faces would therefore mix TWO changes -- the sampler got better AND the
standard got stricter -- and the second one hides the first.

But round 2 contains a randomised control. The page draws 25% of faces from pure noise
(`explore`) and 75% from the purified sampler (`guided`), interleaved in one sitting with
one standard. So within round 2:

    guided vs explore  =  the effect of purification, holding the judge constant

That comparison is immune to the standard shift. This script computes it properly:
Fisher exact test on the good/bad counts, a Welch t-test on the mean grade, the input
spread of each subset (the diversity question), and a drift check (is the guided region
moving outward toward the +-3.5 clip?).
"""
import collections
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
REWARD = os.path.join(ROOT, "webapp", "preference_reward.json")
OUT = os.path.join(ROOT, "results", "preference")
GRADE = {0: "差", 1: "中", 2: "良", 3: "优"}

MODEL = "w128"
recs = []
for line in open(LABELS, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    if not (isinstance(r.get("pca"), list) and len(r["pca"]) == 22
            and r.get("grade") in (0, 1, 2, 3) and r.get("model") == MODEL):
        continue
    r["_sampler"] = (r.get("note") or "plain(r1)").strip() or "plain(r1)"
    recs.append(r)

R2 = [r for r in recs if r["_sampler"] in ("guided", "explore")]


# ---------------------------------------------------------------- statistics helpers
def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test for a 2x2 table [[a,b],[c,d]]."""
    def logfact(n):
        return math.lgamma(n + 1)
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    def prob(a_):
        return math.exp(logfact(r1) + logfact(r2) + logfact(c1) + logfact(c2)
                        - logfact(n) - logfact(a_) - logfact(r1 - a_)
                        - logfact(c1 - a_) - logfact(c2 - r1 + a_))
    p0 = prob(a)
    total = 0.0
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    for a_ in range(lo, hi + 1):
        pa = prob(a_)
        if pa <= p0 * (1 + 1e-7):
            total += pa
    return min(1.0, total)


def welch_t(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0:
        return float("nan"), float("nan")
    t = (x.mean() - y.mean()) / se
    # Welch-Satterthwaite
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    # two-sided p from a normal approximation (df is large enough here)
    p = math.erfc(abs(t) / math.sqrt(2))
    return float(t), float(p)


print("=" * 96)
print("THE CLEAN TEST: guided vs explore within round 2 (same session, same standard)")
print("=" * 96)
G = [r for r in R2 if r["_sampler"] == "guided"]
E = [r for r in R2 if r["_sampler"] == "explore"]
print(f"  guided : n={len(G)}")
print(f"  explore: n={len(E)}")

if len(G) < 5 or len(E) < 5:
    print("  not enough of either to test")
    raise SystemExit(0)

gG = sum(1 for r in G if r["grade"] >= 2)
gE = sum(1 for r in E if r["grade"] >= 2)
print()
print(f"  {'subset':10s} {'n':>4s} {'差':>4s} {'中':>4s} {'良':>4s} {'优':>4s} "
      f"{'mean grade':>11s} {'优+良':>7s}")
print("  " + "-" * 54)
for name, rs, good in (("explore", E, gE), ("guided", G, gG)):
    c = collections.Counter(r["grade"] for r in rs)
    print(f"  {name:10s} {len(rs):4d} {c[0]:4d} {c[1]:4d} {c[2]:4d} {c[3]:4d} "
          f"{np.mean([r['grade'] for r in rs]):11.2f} {good/len(rs)*100:6.1f}%")

p_fisher = fisher_exact(gG, len(G) - gG, gE, len(E) - gE)
t, p_t = welch_t([r["grade"] for r in G], [r["grade"] for r in E])
print()
print(f"  Fisher exact on 优+良 counts : p = {p_fisher:.5f}")
print(f"  Welch t on mean grade       : t = {t:.2f}, p = {p_t:.5f}")
verdict = "SIGNIFICANT" if p_fisher < 0.01 and p_t < 0.01 else "not significant"
print(f"  -> {verdict}")
ratio = (gG / len(G)) / max(gE / len(E), 1e-9)
print(f"  odds: a guided face is {ratio:.2f}x as likely to be rated 优+良 as a random one")

# ---------------------------------------------------------------- diversity
print()
print("=" * 96)
print("DIVERSITY: did purifying shrink the region that is being explored?")
print("=" * 96)


def spread(rs):
    Z = np.array([r["pca"] for r in rs], dtype=np.float64)
    return float(Z.std(0).mean()), float(np.linalg.norm(Z.mean(0), axis=0) if Z.ndim > 1 else np.linalg.norm(Z.mean(0)))


def mean_norm(rs):
    Z = np.array([r["pca"] for r in rs], dtype=np.float64)
    return float(np.linalg.norm(Z, axis=1).mean())


print(f"  {'subset':10s} {'n':>4s} {'per-dim spread':>15s} {'mean |z|':>10s} "
      f"{'max |z| seen':>13s}")
print("  " + "-" * 58)
for name, rs in (("explore", E), ("guided", G)):
    Z = np.array([r["pca"] for r in rs], dtype=np.float64)
    print(f"  {name:10s} {len(rs):4d} {Z.std(0).mean():15.3f} "
          f"{np.linalg.norm(Z, axis=1).mean():10.2f} {np.abs(Z).max():13.2f}")
print(f"  reference: plain N(0,1)^22 has per-dim spread 1.000 and mean |z| ~4.69")
r1 = [r for r in recs if r["_sampler"] == "plain(r1)"]
if r1:
    Z1 = np.array([r["pca"] for r in r1], dtype=np.float64)
    print(f"  round1   {len(r1):4d} {Z1.std(0).mean():15.3f} "
          f"{np.linalg.norm(Z1, axis=1).mean():10.2f} {np.abs(Z1).max():13.2f}")

ZG = np.array([r["pca"] for r in G], dtype=np.float64)
print()
print(f"  guided subset per-dim spread {ZG.std(0).mean():.3f} vs plain 1.000 "
      f"-> ratio {ZG.std(0).mean():.3f}")
print(f"  guided mean |z| = {np.linalg.norm(ZG, axis=1).mean():.2f} vs plain 4.69 "
      f"({'outward drift' if np.linalg.norm(ZG,axis=1).mean() > 5.0 else 'no outward drift'})")
print(f"  largest single coordinate seen in the guided set: {np.abs(ZG).max():.2f} "
      f"(the sampler clips at 3.50)")

# how much of the guided set sits near the clip?
near_clip = float((np.abs(ZG) > 3.0).mean())
print(f"  share of guided coordinates beyond |z|>3.0: {near_clip*100:.1f}%")

# ---------------------------------------------------------------- refit value
print()
print("=" * 96)
print("IS THE NEW MODEL BETTER? (cross-validated AUC, round 1 vs all w128 labels)")
print("=" * 96)


def ridge(Ztr, ytr, Zte, lam=10.0):
    A_ = np.c_[Ztr, np.ones(len(Ztr))]
    B_ = np.c_[Zte, np.ones(len(Zte))]
    w = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ ytr)
    return B_ @ w


def cv_auc(rs, folds=5):
    Z = np.array([r["pca"] for r in rs], dtype=np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    y = np.array([r["grade"] for r in rs], dtype=np.float64)
    idx = np.random.default_rng(0).permutation(len(Z))
    pred = np.zeros(len(Z))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        pred[te] = ridge(Z[tr], y[tr], Z[te])
    hi = pred[y >= 2][:, None]; lo = pred[y <= 1][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


a1 = cv_auc(r1) if len(r1) >= 12 else float("nan")
a_all = cv_auc(recs)
# the honest extra check: train on round 1, test on round 2's RANDOM subset only, so the
# test set is drawn from the same distribution the training set came from
if len(r1) >= 12 and len(E) >= 8:
    Z1 = (np.array([r["pca"] for r in r1]) - np.mean([r["pca"] for r in r1], 0)) / (np.std([r["pca"] for r in r1], 0) + 1e-9)
    y1 = np.array([r["grade"] for r in r1], float)
    ZE = (np.array([r["pca"] for r in E]) - np.mean([r["pca"] for r in r1], 0)) / (np.std([r["pca"] for r in r1], 0) + 1e-9)
    yE = np.array([r["grade"] for r in E], float)
    pE = ridge(Z1, y1, ZE)
    hi = pE[yE >= 2][:, None]; lo = pE[yE <= 1][None, :]
    a_tr = float((hi > lo).mean() + 0.5 * (hi == lo).mean()) if hi.size and lo.size else float("nan")
    print(f"  AUC trained on round 1, tested on round 2's RANDOM faces : {a_tr:.3f} "
          f"(n={len(E)})")
print(f"  AUC (cross-validated) round 1 only   : {a1:.3f}  (n={len(r1)})")
print(f"  AUC (cross-validated) all w128 labels: {a_all:.3f}  (n={len(recs)})")

json.dump({"guided": {"n": len(G), "good": gG}, "explore": {"n": len(E), "good": gE},
           "fisher_p": p_fisher, "welch_t": t, "welch_p": p_t, "ratio": ratio,
           "guided_spread": float(ZG.std(0).mean()),
           "guided_mean_norm": float(np.linalg.norm(ZG, axis=1).mean()),
           "auc_round1": a1, "auc_all": a_all},
          open(os.path.join(OUT, "round2_clean_test.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'round2_clean_test.json')}")
print("CLEAN_TEST_DONE")
