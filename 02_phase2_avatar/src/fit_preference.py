#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Fit a preference distribution over the 22-dim style space from the user's manual labels.

WHY CEM AND NOT "FIT A NORMAL PER CLASS"
----------------------------------------
The label space is 22-dimensional. A full 22x22 covariance needs thousands of samples;
100 labels cannot identify it. So the sampler is a DIAGONAL Gaussian (22 means + 22
stds) and it is updated the way the Cross-Entropy Method does:

    keep the elite (the 优 and 良 labels), refit mean/std to them,
    shrink the result towards the base N(0,1) prior according to how few labels exist,
    keep an exploration floor so the search cannot collapse onto one point.

This is exactly "每多一个标签就把分布形状往优良推、远离差" but with the two safeguards
that make it work at 100 labels: shrinkage (early labels cannot lock the search) and an
exploration floor (the sampler keeps trying new regions).

DIAGNOSTICS, printed before anything is fitted, because they decide whether the whole
exercise can work at all:
  * what fraction of random samples are 优+良  -- if this is under ~2%, no method can
    find the good region from 100 samples
  * a held-out estimate: fit on part of the labels, predict the good rate on the rest
  * a novelty check: if the 优 faces are much CLOSER to real training faces than the 差
    ones, the "optimisation" is really "drift towards memorised faces", which the user
    should know before trusting it

Outputs webapp/preference_dist.json for the labelling page and the viewer.
"""
import argparse, json, os
import numpy as np

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
LEVELS = np.linspace(0, 255, 16)
WEB = os.path.join(ROOT, "webapp")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
os.makedirs(os.path.dirname(LABELS), exist_ok=True)

AP = argparse.ArgumentParser()
AP.add_argument("--min-labels", type=int, default=12,
                help="below this the fit is meaningless and is refused")
AP.add_argument("--prior", type=float, default=24.0,
                help="pseudo-count of the N(0,1) prior; larger = more shrinkage")
AP.add_argument("--eps", type=float, default=0.15, help="exploration floor")
AP.add_argument("--sd-floor", type=float, default=0.55,
                help="never let a component's std fall below this (prevents collapse)")
AP.add_argument("--clip", type=float, default=3.5)
AP.add_argument("--write", action="store_true", help="write preference_dist.json")
AP.add_argument("--crossval", action="store_true", help="run the held-out check")
A = AP.parse_args()

GRADE_NAME = {0: "差", 1: "中", 2: "良", 3: "优"}


def load():
    if not os.path.exists(LABELS):
        return []
    out = []
    for line in open(LABELS, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not isinstance(r.get("pca"), list) or len(r["pca"]) != 22:
            continue
        if r.get("grade") not in (0, 1, 2, 3):
            continue
        out.append(r)
    return out


def fit(recs, prior, eps, sd_floor):
    """Diagonal Gaussian fitted to the elite (grade >= 2), shrunk to the N(0,1) prior."""
    Z = np.array([r["pca"] for r in recs], dtype=np.float64)
    g = np.array([r["grade"] for r in recs], dtype=np.int64)
    elite = Z[g >= 2]
    if len(elite) < 2:
        return None, "not enough 优良 labels to fit"
    n = len(elite)
    m = elite.mean(0)
    s = elite.std(0) if n > 1 else np.ones(22)
    # shrinkage towards the N(0,1) prior
    w = n / (n + prior)
    m = w * m + (1 - w) * 0.0
    s = np.sqrt(w * (s ** 2) + (1 - w) * 1.0)
    s = np.maximum(s, sd_floor)
    return {"mean": m, "sd": s, "n_elite": n}, None


def good_rate_under(dist, recs, eps):
    """Empirical check: draw from the fitted sampler and score with a simple k-NN model
    of the labels. Used for the held-out estimate, NOT as ground truth."""
    Z = np.array([r["pca"] for r in recs], dtype=np.float64)
    g = np.array([r["grade"] for r in recs], dtype=np.int64)
    if dist is None:
        return float((g >= 2).mean())
    rng = np.random.default_rng(0)
    N = 4000
    Zs = dist["mean"][None, :] + dist["sd"][None, :] * rng.standard_normal((N, 22))
    use_base = rng.random(N) < eps
    Zs[use_base] = rng.standard_normal((use_base.sum(), 22))
    Zs = np.clip(Zs, -3.5, 3.5)
    # k-NN vote with k=5 over the labels
    d = ((Zs[:, None, :] - Z[None, :, :]) ** 2).sum(2)
    k = min(5, len(Z))
    nn = np.argsort(d, 1)[:, :k]
    votes = (g[nn] >= 2).mean(1)
    return float((votes >= 0.5).mean())


def novelty_check(recs, model_tag):
    """Are the 优 faces closer to the training data than the 差 ones?
    If yes, 'good' may mean 'looks like a memorised face'."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    run = os.path.join(ROOT, "results", "coherence", model_tag)
    p = os.path.join(run, "final.pt")
    if not os.path.exists(p):
        return None

    class Gen(nn.Module):
        def __init__(self, hid=256, cdim=22):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                     nn.Linear(hid, hid), nn.ReLU(),
                                     nn.Linear(hid, 32 * 16))
        def logits(self, c):
            return self.net(c).view(-1, 32, 16)

    ck = torch.load(p, weights_only=False)
    Gm = Gen(ck["args"]["hid"], 22); Gm.load_state_dict(ck["G"]); Gm.eval()
    Z = torch.tensor(np.array([r["pca"] for r in recs]), dtype=torch.float32)
    with torch.no_grad():
        dec = Gm.logits(Z).argmax(-1).numpy()
    img = layout.image_from_left_half(dec)                     # (n,8,8)
    va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
    full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
    pool = full[va[:2000]].reshape(2000, -1).astype(np.int16)
    cur = img.reshape(len(img), -1).astype(np.int16)
    # Hamming distance from each generated face to the nearest real face
    d = np.empty(len(cur))
    for s in range(0, len(cur), 64):
        blk = cur[s:s + 64]
        d[s:s + 64] = (blk[:, None, :] != pool[None, :, :]).sum(2).min(1)
    g = np.array([r["grade"] for r in recs])
    out = {}
    for gv in (0, 1, 2, 3):
        m = g == gv
        if m.sum():
            out[GRADE_NAME[gv]] = {"n": int(m.sum()),
                                   "mean_nearest_real_distance": float(d[m].mean()),
                                   "min": int(d[m].min())}
    return out


recs = load()
print(f"labels loaded: {len(recs)}")
if len(recs) < A.min_labels:
    print(f"\nREFUSED: need at least {A.min_labels} labels to fit anything meaningful; "
          f"have {len(recs)}.")
    print("Collect more labels with webapp/label.html first.")
    raise SystemExit(0)

g = np.array([r["grade"] for r in recs])
counts = np.bincount(g, minlength=4)
tot = len(g)
print()
print("=" * 88)
print("0. THE DIAGNOSTIC THAT DECIDES WHETHER THIS CAN WORK")
print("=" * 88)
for k in (3, 2, 1, 0):
    print(f"  {GRADE_NAME[k]} : {counts[k]:4d}  ({100*counts[k]/tot:5.1f}%)")
good = (g >= 2).mean()
print(f"  ---> 优+良 占 {100*good:.1f}%")
if good < 0.02:
    print("  !! under 2% good: from ~100 random samples only ~2 are good, so an elite")
    print("     selection would be fitting noise. More labels (or a wider search) first.")
elif good < 0.10:
    print("  !  5-10% good: workable but expect slow progress; 200+ labels advised.")
else:
    print("  OK  >=10% good: CEM has a real signal to climb.")

print()
print("=" * 88)
print("1. NOVELTY CHECK  (does 'good' secretly mean 'looks like a training face'?)")
print("=" * 88)
nov = novelty_check(recs, "final_lam5_20k")
if nov:
    for k, v in nov.items():
        print(f"  {k}: n={v['n']:3d}  mean distance to nearest real face "
              f"{v['mean_nearest_real_distance']:6.2f}  (min {v['min']})")
    if "优" in nov and "差" in nov:
        diff = nov["差"]["mean_nearest_real_distance"] - nov["优"]["mean_nearest_real_distance"]
        print(f"  -> 优 is {diff:+.2f} closer than 差 "
              f"({'good faces look MORE like memorised faces' if diff > 1 else 'no strong memorisation drift'})")
else:
    print("  (model checkpoint not found; skipped)")

print()
print("=" * 88)
print("2. FIT (diagonal Gaussian on the elite, shrunk to the N(0,1) prior)")
print("=" * 88)
dist, err = fit(recs, A.prior, A.eps, A.sd_floor)
if err:
    print(f"  cannot fit: {err}")
    raise SystemExit(0)
print(f"  elite size {dist['n_elite']}   prior strength {A.prior}   "
      f"shrinkage weight w = {dist['n_elite']/(dist['n_elite']+A.prior):.3f}")
print(f"  mean (first 6): {np.round(dist['mean'][:6], 3).tolist()}")
print(f"  sd   (first 6): {np.round(dist['sd'][:6], 3).tolist()}")
print(f"  sd range: {dist['sd'].min():.3f} .. {dist['sd'].max():.3f}  "
      f"(floor {A.sd_floor})")

if A.crossval and len(recs) >= 30:
    print()
    print("=" * 88)
    print("3. HELD-OUT CHECK (fit on half the labels, score on the other half)")
    print("=" * 88)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(recs))
    h1 = [recs[i] for i in perm[:len(perm) // 2]]
    h2 = [recs[i] for i in perm[len(perm) // 2:]]
    d1, e1 = fit(h1, A.prior, A.eps, A.sd_floor)
    base_rate = float((np.array([r["grade"] for r in h2]) >= 2).mean())
    if d1:
        new_rate = good_rate_under(d1, h2, A.eps)
        print(f"  base sampler good rate on the held-out half : {base_rate*100:5.1f}%")
        print(f"  fitted sampler good rate (k-NN proxy)       : {new_rate*100:5.1f}%")
        print(f"  -> apparent change {100*(new_rate-base_rate):+.1f} points")
        print("  (the k-NN proxy is weak with so few labels; treat the sign, not the size)")

if A.write:
    out = {
        "mean": [float(v) for v in dist["mean"]],
        "sd": [float(v) for v in dist["sd"]],
        "eps": float(A.eps),
        "clip": [-float(A.clip), float(A.clip)],
        "n_labels": int(len(recs)),
        "n_elite": int(dist["n_elite"]),
        "prior_strength": float(A.prior),
        "sd_floor": float(A.sd_floor),
        "good_rate_observed": float(good),
        "counts": {"优": int(counts[3]), "良": int(counts[2]),
                   "中": int(counts[1]), "差": int(counts[0])},
        "predicted_good_rate": float(good_rate_under(dist, recs, A.eps)),
        "method": "CEM: diagonal Gaussian fitted to grade>=2, shrunk to N(0,1) prior, "
                  "with an exploration floor",
    }
    p = os.path.join(WEB, "preference_dist.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {p}  (the labelling page and the viewer will pick it up on refresh)")
print("\nFIT_PREFERENCE_DONE")
