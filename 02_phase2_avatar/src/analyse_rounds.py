#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Did the second preference round IMPROVE things, or COLLAPSE the diversity?

A preference loop is a feedback loop: round 2's faces are drawn from the distribution that
round 1's labels produced. Two things can happen and they look similar in the grade
average alone:

  it works      the new faces really are better, and the grades shift upward while the
                sampled region stays wide enough to keep producing varied faces
  it collapses  the sampler has narrowed onto a small region; the grades may look fine or
                even better, but every face is nearly the same, and the model can no
                longer discover anything outside that region

So this reports, per round, BOTH the quality signal and the diversity signal:

  * grade distribution and mean grade
  * mean predicted score under the reward model that produced the round
  * SPREAD of the sampled 22-dim inputs, relative to the base N(0,1) sampler (which is 1.0)
  * number of near-duplicate input vectors
  * how often a face was identical to another in the same round

Rounds are detected from the `note` field: labels written before the sampler was recorded
have an empty note and are round 1 (plain sampling); afterwards each label says
plain / explore / guided.
"""
import collections
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
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
REWARD = os.path.join(ROOT, "webapp", "preference_reward.json")
OUT = os.path.join(ROOT, "results", "preference")
GRADE = {0: "差", 1: "中", 2: "良", 3: "优"}

MODEL = ""
for i, a in enumerate(sys.argv):
    if a == "--model" and i + 1 < len(sys.argv):
        MODEL = sys.argv[i + 1]
if not MODEL:
    MODEL = "w128"


def load():
    out = []
    for line in open(LABELS, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not (isinstance(r.get("pca"), list) and len(r["pca"]) == 22
                and r.get("grade") in (0, 1, 2, 3)):
            continue
        if r.get("model") != MODEL:
            continue
        out.append(r)
    return out


recs = load()
print(f"model '{MODEL}': {len(recs)} labels")
if not recs:
    raise SystemExit("no labels for this model")

# round assignment: an empty note means the sampler was not yet recorded = round 1
for r in recs:
    note = (r.get("note") or "").strip()
    r["_sampler"] = note if note else "plain(r1)"
    r["_round"] = 1 if not note else 2

R1 = [r for r in recs if r["_round"] == 1]
R2 = [r for r in recs if r["_round"] == 2]

print()
print("=" * 100)
print("ROUND COMPARISON")
print("=" * 100)


def spread(rs):
    """Per-component standard deviation of the 22-dim inputs, averaged."""
    if len(rs) < 2:
        return float("nan"), float("nan")
    Z = np.array([r["pca"] for r in rs], dtype=np.float64)
    sd = Z.std(0)
    return float(sd.mean()), float(np.linalg.norm(Z.mean(0)))


def dupes(rs, tol=0.25):
    """How many pairs of input vectors are nearly identical?"""
    if len(rs) < 2:
        return 0, 0
    Z = np.array([r["pca"] for r in rs], dtype=np.float64)
    # quantise to `tol` units and count collisions
    key = np.round(Z / tol).astype(np.int64)
    seen = collections.Counter(map(tuple, key))
    colliding = sum(c for c in seen.values() if c > 1)
    return len(seen), colliding


def rw_score(rs):
    """Score the inputs with the CURRENT exported reward model, if it matches."""
    if not os.path.exists(REWARD):
        return float("nan")
    R = json.load(open(REWARD, encoding="utf-8"))
    if R.get("model") != MODEL:
        return float("nan")
    w = np.array(R["w"]); b = float(R["b"])
    zm = np.array(R["zmean"]); zsd = np.array(R["zsd"])
    Z = np.array([r["pca"] for r in rs], dtype=np.float64)
    return float((((Z - zm) / zsd) @ w + b).mean())


print(f"  {'round':22s} {'n':>4s} {'差':>4s} {'中':>4s} {'良':>4s} {'优':>4s} "
      f"{'mean grade':>11s} {'优+良':>7s} {'input spread':>13s} {'distinct':>9s}")
print("  " + "-" * 92)
rows = []
for name, rs in (("round 1 (plain)", R1), ("round 2 (guided)", R2)):
    if not rs:
        print(f"  {name:22s} {'--':>4s}   (no labels yet)")
        continue
    g = collections.Counter(r["grade"] for r in rs)
    sp, mn = spread(rs)
    nd, coll = dupes(rs)
    rows.append({"round": name, "n": len(rs), "counts": dict(g),
                 "mean_grade": float(np.mean([r["grade"] for r in rs])),
                 "good": float(np.mean([r["grade"] >= 2 for r in rs])),
                 "spread": sp, "distinct": nd, "colliding": coll})
    print(f"  {name:22s} {len(rs):4d} {g[0]:4d} {g[1]:4d} {g[2]:4d} {g[3]:4d} "
          f"{np.mean([r['grade'] for r in rs]):11.2f} "
          f"{np.mean([r['grade'] >= 2 for r in rs])*100:6.1f}% {sp:13.3f} {nd:9d}")

print()
print("  reference: a pure N(0,1)^22 sampler has per-component spread 1.000")
if R1 and R2:
    sp1 = spread(R1)[0]
    sp2 = spread(R2)[0]
    print(f"  spread ratio round2/round1 = {sp2/sp1:.3f}  "
          f"({'WIDE, no collapse' if sp2/sp1 > 0.85 else 'NARROWER' if sp2/sp1 > 0.6 else 'COLLAPSING'})")

# ---------------------------------------------------------------- sampler breakdown
print()
print("=" * 100)
print("ROUND 2 BROKEN DOWN BY HOW EACH FACE WAS DRAWN")
print("=" * 100)
by_s = collections.defaultdict(list)
for r in R2:
    by_s[r["_sampler"]].append(r)
if by_s:
    print(f"  {'sampler':12s} {'n':>4s} {'mean grade':>11s} {'优+良':>7s} {'mean score':>11s}")
    print("  " + "-" * 52)
    for s in sorted(by_s):
        rs = by_s[s]
        print(f"  {s:12s} {len(rs):4d} "
              f"{np.mean([r['grade'] for r in rs]):11.2f} "
              f"{np.mean([r['grade'] >= 2 for r in rs])*100:6.1f}% "
              f"{rw_score(rs):11.3f}")
    print()
    print("  The guided subset is the one the loop is optimising. If its mean grade is")
    print("  clearly above the explore subset, the loop is doing what it claims. If they")
    print("  are equal, the reward model has stopped separating anything.")
else:
    print("  (no round 2 labels yet)")
    print("  -> run the labelling page; round 2 labels carry a `note` field naming the")
    print("     sampler that drew them (plain / explore / guided)")

# ---------------------------------------------------------------- does refitting help?
print()
print("=" * 100)
print("WOULD A REFIT ON ALL ROUNDS HELP? (cross-validated, so it is honest)")
print("=" * 100)


def ridge(Ztr, ytr, Zte, lam):
    A_ = np.c_[Ztr, np.ones(len(Ztr))]
    B_ = np.c_[Zte, np.ones(len(Zte))]
    w = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ ytr)
    return B_ @ w


def cv_auc(rs, lam=10.0, folds=5):
    Z = np.array([r["pca"] for r in rs], dtype=np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    y = np.array([r["grade"] for r in rs], dtype=np.float64)
    idx = np.random.default_rng(0).permutation(len(Z))
    pred = np.zeros(len(Z))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        pred[te] = ridge(Z[tr], y[tr], Z[te], lam)
    hi = pred[y >= 2][:, None]
    lo = pred[y <= 1][None, :]
    if hi.size == 0 or lo.size == 0:
        return float("nan")
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


a1 = cv_auc(R1) if len(R1) >= 12 else float("nan")
a12 = cv_auc(recs) if len(recs) >= 12 else float("nan")
print(f"  AUC using round 1 only      : {a1:.3f}  (n={len(R1)})")
print(f"  AUC using both rounds       : {a12:.3f}  (n={len(recs)})")
print(f"  change: {a12-a1:+.3f}")
print()
print("  Caveat: round 2 labels come from a NARROWER input distribution than round 1, so")
print("  the two rounds are not a random sample of the same population. Pooling them is")
print("  still valid for learning a ranking, but the AUC is measured on a different mix, so")
print("  a small change should not be over-interpreted.")

json.dump({"model": MODEL, "rows": rows,
           "samplers": {s: {"n": len(v), "mean_grade": float(np.mean([r["grade"] for r in v]))}
                        for s, v in by_s.items()},
           "auc_round1": a1, "auc_both": a12},
          open(os.path.join(OUT, "rounds.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'rounds.json')}")
print("ROUNDS_DONE")
