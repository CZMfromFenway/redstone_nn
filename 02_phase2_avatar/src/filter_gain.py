#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Estimate the ACTUAL gain from filtering, using the user's own held-out labels.

The reward-model approach is only worth using if selecting its top candidates raises the
rate of 优+良 candidates. This is measured directly, without any proxy:

  * split the 159 labels into two halves
  * fit the reward model on one half
  * rank the OTHER half by the model's prediction
  * report the true 优+良 rate among the top X% versus the overall rate

That is exactly what the sampler does at run time (rank a pool, keep the best), except
the "pool" here is real labelled faces whose true grade we know.

Two models are compared because the earlier analysis found something counter-intuitive:
ridge beat the MLP on held-out data (AUC 0.617 vs 0.553). With 159 labels the MLP
overfits, so the exported reward model should probably be the SIMPLER one.

Repeated over many random splits so the answer is not one lucky partition.
"""
import json, os, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "preference")
GRADE = {0: "差", 1: "中", 2: "良", 3: "优"}


def load(model_filter=""):
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
        if model_filter and r.get("model") != model_filter:
            continue
        out.append(r)
    return out


MODEL = ""
for i, a in enumerate(sys.argv):
    if a == "--model" and i + 1 < len(sys.argv):
        MODEL = sys.argv[i + 1]
recs = load(MODEL)
if MODEL:
    print(f"using ONLY labels from model '{MODEL}'")
X = np.array([r["pca"] for r in recs], dtype=np.float64)
y = np.array([r["grade"] for r in recs], dtype=np.int64)
n = len(recs)
base_good = float((y >= 2).mean())
base_excellent = float((y == 3).mean())
print(f"{n} labels   base rate 优+良 = {base_good*100:.1f}%   优 = {base_excellent*100:.1f}%")

try:
    import torch
    import torch.nn as nn
    HAVE_T = True
except Exception:
    HAVE_T = False


def ridge_fit(Xtr, ytr, lam):
    """Standardise inside the fit so the penalty is meaningful across components."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return lambda Xte: (np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ w)


def mlp_fit(Xtr, ytr, hid=32, steps=1500, lr=3e-3, l2=3e-3, seed=0):
    torch.manual_seed(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    Ztr = (Xtr - mu) / sd
    m = nn.Sequential(nn.Linear(22, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                      nn.Linear(hid, 1))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=l2)
    Xt = torch.tensor(Ztr, dtype=torch.float32)
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-9)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    for _ in range(steps):
        loss = ((m(Xt).squeeze(-1) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def pred(Xte):
        with torch.no_grad():
            p = m(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
        return p * ys + ym
    return pred


KEEPS = (1.0, 0.5, 0.35, 0.25, 0.15)
print()
print("=" * 100)
print("TRUE 优+良 RATE AMONG THE MODEL'S TOP-RANKED HELD-OUT FACES")
print("=" * 100)
print("  keep=1.0 is the untouched pool. 'keep' is the fraction the sampler would retain.")
print()

results = {}
for name, fitfn in (("ridge lam=10", lambda a, b: ridge_fit(a, b, 10.0)),
                    ("ridge lam=30", lambda a, b: ridge_fit(a, b, 30.0)),
                    ("MLP 32x32", lambda a, b: mlp_fit(a, b))):
    if name.startswith("MLP") and not HAVE_T:
        continue
    agg = {k: [] for k in KEEPS}
    for split in range(40):
        rng = np.random.default_rng(split)
        idx = rng.permutation(n)
        h = n // 2
        tr, te = idx[:h], idx[h:]
        pred = fitfn(X[tr], y[tr])
        s = pred(X[te])
        yt = y[te]
        order = np.argsort(-s)
        for keep in KEEPS:
            m = max(1, int(len(order) * keep))
            sel = order[:m]
            agg[keep].append(float((yt[sel] >= 2).mean()))
    results[name] = {str(k): float(np.mean(v)) for k, v in agg.items()}
    print(f"  {name}")
    for keep in KEEPS:
        r = float(np.mean(agg[keep]))
        sd = float(np.std(agg[keep]))
        tag = "  <- untouched pool" if keep == 1.0 else ""
        delta = "" if keep == 1.0 else f"   ({100*(r-base_good):+.1f} points vs pool)"
        print(f"    keep {keep*100:4.0f}%  ->  优+良 rate {r*100:5.1f}%  "
              f"(+-{sd*100:.1f}){delta}{tag}")
    print()

print("=" * 100)
print("READING")
print("=" * 100)
best_name = max(results, key=lambda k: results[k]["0.35"])
best = results[best_name]["0.35"]
print(f"  best at keep=35%: {best_name}  ->  优+良 {best*100:.1f}% "
      f"(pool was {base_good*100:.1f}%, so {100*(best-base_good):+.1f} points)")
if best - base_good < 0.03:
    print("  VERDICT: the gain is small. Filtering does not clearly beat random drawing,")
    print("           so the labels as they stand do not support a useful sampler.")
elif best - base_good < 0.10:
    print("  VERDICT: a real but modest gain. Filtering helps somewhat; more labels and a")
    print("           cleaner judging standard would help more.")
else:
    print("  VERDICT: a substantial gain. Preference-guided filtering is worth using.")

json.dump({"n": n, "base_good": base_good, "results": results},
          open(os.path.join(OUT, "filter_gain.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'filter_gain.json')}")
print("FILTER_GAIN_DONE")
