#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Analyse the user's 159 manual labels properly, before trusting any reward model.

Three things must be true before preference-guided sampling can help, and each is tested:

  1. THE LABELS ARE LEARNABLE. A model trained on some of the labels must predict the
     grades of held-out labels better than chance. Measured by k-fold cross-validation
     on the 22 PCA coordinates, against a shuffled-label control.

  2. THE LABELS ARE CONSISTENT. If the user's standard drifted during the session, the
     same region of style space would carry contradictory grades and no model could fit.
     Checked by looking at the grade distribution over time and by measuring how much
     the model's accuracy degrades when trained on the first half only.

  3. THE GRADES SEPARATE IN STYLE SPACE. If 优 and 差 occupy overlapping regions, the
     choice is about something the 22 PCA numbers do not capture, and no amount of
     resampling will help. Measured by nearest-neighbour grade agreement and by the
     distance between class centroids relative to the spread.

Output feeds the decision: whether to trust the reward model, and how much improvement
to expect from it.
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
        if isinstance(r.get("pca"), list) and len(r["pca"]) == 22 and r.get("grade") in (0, 1, 2, 3):
            out.append(r)
    return out


recs = load()
X = np.array([r["pca"] for r in recs], dtype=np.float64)
y = np.array([r["grade"] for r in recs], dtype=np.int64)
ts = [r["ts"] for r in recs]
n = len(recs)
print(f"loaded {n} labels")
counts = np.bincount(y, minlength=4)
print("grade counts (差/中/良/优): " + " / ".join(
    f"{GRADE[k]}={counts[k]}" for k in (0, 1, 2, 3)))
print(f"优+良 = {100*(y>=2).mean():.1f}%   (this is the base rate a sampler must beat)")

# =====================================================================================
print()
print("=" * 96)
print("1. ARE THE LABELS LEARNABLE? (k-fold cross-validation vs a shuffled control)")
print("=" * 96)

try:
    import torch
    import torch.nn as nn
    HAVE_T = True
except Exception:
    HAVE_T = False
    print("  torch unavailable")


def fit_predict_mlp(Xtr, ytr, Xte, hid=32, steps=1200, lr=3e-3, l2=3e-3, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(22, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                      nn.Linear(hid, 1))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=l2)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    ym, ys = float(yt.mean()), float(yt.std() + 1e-9)
    yz = (yt - ym) / ys
    for _ in range(steps):
        loss = ((m(Xt).squeeze(-1) - yz) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = m(torch.tensor(Xte, dtype=torch.float32)).squeeze(-1).numpy()
    return p * ys + ym


def ridge_predict(Xtr, ytr, Xte, lam=1.0):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    B = np.c_[Xte, np.ones(len(Xte))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return B @ w


def knn_predict(Xtr, ytr, Xte, k=7):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(2)
    idx = np.argsort(d, 1)[:, :k]
    return ytr[idx].mean(1)


def cv_score(fn, X, y, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    pred = np.zeros(len(X))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        pred[te] = fn(X[tr], y[tr], X[te])
    # Spearman rank correlation is the right metric: we only need the ORDERING
    rk = np.argsort(np.argsort(pred))
    rt = np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rk, rt)[0, 1])
    # and: does it rank 优+良 above 中+差?
    hi = pred[y >= 2].mean()
    lo = pred[y <= 1].mean()
    # AUC-style: probability a random good label scores above a random poor one
    if (y >= 2).any() and (y <= 1).any():
        a = pred[y >= 2][:, None]
        b = pred[y <= 1][None, :]
        auc = float((a > b).mean() + 0.5 * (a == b).mean())
    else:
        auc = float("nan")
    return rho, float(hi - lo), auc


print(f"  {'model':22s} {'Spearman':>10s} {'gap(优+良 - 中+差)':>20s} {'AUC':>7s}")
print("  " + "-" * 64)
rows = []
if HAVE_T:
    for name, fn in (("MLP 32x32", lambda a, b, c: fit_predict_mlp(a, b, c)),
                     ("ridge", ridge_predict),
                     ("kNN k=7", knn_predict)):
        rho, gap, auc = cv_score(fn, X, y)
        rows.append({"model": name, "spearman": rho, "gap": gap, "auc": auc})
        print(f"  {name:22s} {rho:10.3f} {gap:20.3f} {auc:7.3f}")

# control: shuffle the labels, everything should collapse to chance
rng = np.random.default_rng(3)
y_sh = rng.permutation(y)
rho_c, gap_c, auc_c = cv_score(lambda a, b, c: fit_predict_mlp(a, b, c), X, y_sh)
print(f"  {'CONTROL (shuffled)':22s} {rho_c:10.3f} {gap_c:20.3f} {auc_c:7.3f}")
print()
best = max(rows, key=lambda r: r["auc"]) if rows else None
if best:
    print(f"  best model: {best['model']}  AUC {best['auc']:.3f}  Spearman "
          f"{best['spearman']:.3f}")
    print(f"  (AUC 0.5 = no signal, 1.0 = perfect; control sits at {auc_c:.3f})")

# =====================================================================================
print()
print("=" * 96)
print("2. ARE THE LABELS CONSISTENT OVER TIME?")
print("=" * 96)
print(f"  session spans {ts[0]} .. {ts[-1]}")
# split into 4 chunks in time and compare grade mixes
q = np.array_split(np.arange(n), 4)
print(f"  {'chunk':>6} {'n':>4} {'差':>4} {'中':>4} {'良':>4} {'优':>4} {'mean grade':>11}")
for i, ix in enumerate(q):
    c = np.bincount(y[ix], minlength=4)
    print(f"  {i+1:6d} {len(ix):4d} {c[0]:4d} {c[1]:4d} {c[2]:4d} {c[3]:4d} "
          f"{y[ix].mean():11.2f}")
# does a model trained on the first half work on the second?
h = n // 2
if HAVE_T:
    p2 = fit_predict_mlp(X[:h], y[:h], X[h:])
    rk = np.argsort(np.argsort(p2)); rt = np.argsort(np.argsort(y[h:]))
    rho_h = float(np.corrcoef(rk, rt)[0, 1])
    a = p2[y[h:] >= 2][:, None]; b = p2[y[h:] <= 1][None, :]
    auc_h = float((a > b).mean()) if a.size and b.size else float("nan")
    print(f"  trained on the FIRST half, tested on the SECOND: "
          f"Spearman {rho_h:.3f}, AUC {auc_h:.3f}")
    print(f"  (if this is much worse than the k-fold number, the standard drifted)")

# =====================================================================================
print()
print("=" * 96)
print("3. DO THE GRADES SEPARATE IN STYLE SPACE?")
print("=" * 96)
mu_all = X.mean(0)
sd_all = X.std(0) + 1e-9
Z = (X - mu_all) / sd_all
cent = {k: Z[y == k].mean(0) for k in np.unique(y)}
print(f"  {'grade':>6} {'n':>4} {'|centroid|':>11} {'mean dist to own centroid':>26}")
for k in sorted(cent):
    d = np.linalg.norm(Z[y == k] - cent[k], axis=1)
    print(f"  {GRADE[k]:>6} {int((y==k).sum()):4d} {np.linalg.norm(cent[k]):11.2f} "
          f"{d.mean():26.2f}")
print()
print(f"  distance between class centroids:")
for a_ in sorted(cent):
    for b_ in sorted(cent):
        if a_ < b_:
            print(f"    {GRADE[a_]} <-> {GRADE[b_]} : "
                  f"{np.linalg.norm(cent[a_]-cent[b_]):.2f}")
# nearest-neighbour grade agreement: if neighbours disagree, grades are not a function
# of position in this space
d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(2)
np.fill_diagonal(d, 1e18)
nn = d.argmin(1)
agree = float((y[nn] == y).mean())
agree_pm1 = float((np.abs(y[nn] - y) <= 1).mean())
# chance level for exact agreement, given the grade distribution
p = np.bincount(y, minlength=4) / n
chance = float((p ** 2).sum())
print()
print(f"  nearest-neighbour EXACT grade agreement: {agree*100:.1f}%  "
      f"(chance {chance*100:.1f}%)")
print(f"  nearest-neighbour agreement within +-1 grade: {agree_pm1*100:.1f}%")

json.dump({"n": n, "counts": counts.tolist(),
           "cv": rows, "control": {"spearman": rho_c, "auc": auc_c},
           "nn_agree": agree, "nn_chance": chance, "nn_agree_pm1": agree_pm1,
           "time_chunks": [np.bincount(y[ix], minlength=4).tolist() for ix in q]},
          open(os.path.join(OUT, "label_analysis.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'label_analysis.json')}")
print("LABEL_ANALYSIS_DONE")
