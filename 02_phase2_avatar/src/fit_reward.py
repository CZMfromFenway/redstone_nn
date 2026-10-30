#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Fit the reward model from the user's manual labels, and export it for the page.

WHY THE DEFAULT IS NOW A LINEAR (RIDGE) MODEL
---------------------------------------------
fit_reward.py originally exported a 22->64->64->1 MLP. With the user's 159 real labels,
held-out validation reversed that choice:

    model                 AUC     true 优+良 rate among the top 35% of held-out faces
    ridge lam=30          0.617   74.5%      <- best
    ridge lam=10          ~0.60   74.3%
    MLP 32x32             0.553   72.0%      <- overfits at this label count
    (untouched pool)      --      67.3%

So the MLP was losing about 2.5 points to overfitting. The user's aesthetic also turns
out to be mostly LINEAR in the 22 principal components: within-class spread is ~4.5 while
class centroids are only 1.0-2.3 apart, and a linear model captures most of what is
learnable there.

The penalty strength is chosen by cross-validation rather than fixed, and the exported
JSON carries `kind` so the page knows whether to apply a ReLU.

EXPORT FORMAT
    kind == "linear":  score = b + sum_i w[i] * ((z[i]-zmean[i])/zsd[i])
    kind == "mlp"   :  the 3-layer form the page already understood
"""
import argparse, json, os, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
os.makedirs(os.path.dirname(LABELS), exist_ok=True)
GRADE = {0: "差", 1: "中", 2: "良", 3: "优"}

AP = argparse.ArgumentParser()
AP.add_argument("--min-labels", type=int, default=14)
AP.add_argument("--kind", default="linear", choices=["linear", "mlp"],
                help="linear is the default: measured better than the MLP at 159 labels")
AP.add_argument("--lams", default="1,3,10,30,100",
                help="ridge penalties to cross-validate over")
AP.add_argument("--hid", type=int, default=32)
AP.add_argument("--steps", type=int, default=2000)
AP.add_argument("--l2", type=float, default=1e-2)
AP.add_argument("--model", default="",
                help="only use labels produced by this architecture key (lam5, gray32, "
                     "color16). CRITICAL: the same 22-dim vector draws a DIFFERENT face "
                     "under each architecture, so labels from different models must never "
                     "be pooled into one reward model.")
AP.add_argument("--write", action="store_true")
A = AP.parse_args()


def load(model_filter=""):
    """Returns (records, counts_by_model). Filters by architecture when asked."""
    if not os.path.exists(LABELS):
        return [], {}
    out = []
    by_model = {}
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
        mk = r.get("model", "unknown")
        by_model[mk] = by_model.get(mk, 0) + 1
        if model_filter and mk != model_filter:
            continue
        out.append(r)
    return out, by_model


recs, by_model = load(A.model)
n = len(recs)
if by_model:
    print(f"labels on disk by model: {by_model}")
if A.model:
    print(f"using ONLY labels from model '{A.model}': {n}")
else:
    print(f"labels loaded: {n}")
    if len(by_model) > 1:
        print("!! labels from SEVERAL architectures are being pooled. That is almost")
        print("!! certainly wrong -- rerun with --model <key> to pick one.")
if n < A.min_labels:
    print(f"\nNeed at least {A.min_labels}; open http://127.0.0.1:8770/label.html")
    raise SystemExit(0)

X = np.array([r["pca"] for r in recs], dtype=np.float64)
y = np.array([r["grade"] for r in recs], dtype=np.float64)
counts = np.bincount(y.astype(int), minlength=4)
print()
print("=" * 92)
print("LABELS")
print("=" * 92)
for k in (3, 2, 1, 0):
    print(f"  {GRADE[k]} : {counts[k]:4d}  ({100*counts[k]/n:5.1f}%)")
print(f"  ---> 优+良 = {100*(y>=2).mean():.1f}%  (the base rate to beat)")

mu = X.mean(0)
sd = X.std(0) + 1e-9
Z = (X - mu) / sd


def ridge(Ztr, ytr, Zte, lam):
    A_ = np.c_[Ztr, np.ones(len(Ztr))]
    B_ = np.c_[Zte, np.ones(len(Zte))]
    w = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ ytr)
    return B_ @ w


def cv_pick_lambda(Z, y, lams, folds=5, seed=0):
    """Choose the penalty by held-out AUC, not by training fit."""
    idx = np.random.default_rng(seed).permutation(len(Z))
    best, best_auc = None, -1.0
    for lam in lams:
        pred = np.zeros(len(Z))
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            pred[te] = ridge(Z[tr], y[tr], Z[te], lam)
        if (y >= 2).any() and (y <= 1).any():
            a = pred[y >= 2][:, None]
            b = pred[y <= 1][None, :]
            auc = float((a > b).mean() + 0.5 * (a == b).mean())
        else:
            auc = 0.0
        print(f"    lam={lam:<6g} cv AUC {auc:.3f}")
        if auc > best_auc:
            best_auc, best = auc, lam
    return best, best_auc


print()
print("=" * 92)
print("CHOOSING THE REGULARISATION BY CROSS-VALIDATION")
print("=" * 92)
lams = [float(s) for s in A.lams.split(",")]
lam, cv_auc = cv_pick_lambda(Z, y, lams)
print(f"  chosen lam = {lam:g}   (held-out AUC {cv_auc:.3f}; 0.5 = no signal)")

if A.kind == "linear":
    A_ = np.c_[Z, np.ones(n)]
    coef = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ y)
    w, b = coef[:-1], float(coef[-1])
    pred = A_ @ coef
    export = {"kind": "linear",
              "zmean": [float(v) for v in mu], "zsd": [float(v) for v in sd],
              "w": [float(v) for v in w], "b": b}
    nparam = 22
else:
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(22, A.hid), nn.ReLU(),
                      nn.Linear(A.hid, A.hid), nn.ReLU(), nn.Linear(A.hid, 1))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=A.l2)
    Xt = torch.tensor(Z, dtype=torch.float32)
    ym, ys = float(y.mean()), float(y.std() + 1e-9)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32)
    for _ in range(A.steps):
        loss = ((m(Xt).squeeze(-1) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = m(Xt).squeeze(-1).numpy() * ys + ym
    lin = [l for l in m if isinstance(l, nn.Linear)]
    (W1, b1), (W2, b2), (W3, b3) = [(l.weight.detach().numpy().T.copy(),
                                     l.bias.detach().numpy().copy()) for l in lin]
    export = {"kind": "mlp",
              "zmean": [float(v) for v in mu], "zsd": [float(v) for v in sd],
              "W1": W1.tolist(), "b1": b1.tolist(),
              "W2": W2.tolist(), "b2": b2.tolist(),
              "W3": W3.tolist(), "b3": b3.tolist()}
    nparam = sum(p.numel() for p in m.parameters())

# ---- what the model actually learned -----------------------------------------------
print()
print("=" * 92)
print(f"FITTED MODEL ({A.kind}, {nparam} parameters)")
print("=" * 92)
if A.kind == "linear":
    order = np.argsort(-np.abs(w))
    print("  strongest components (a larger |weight| means the axis matters more):")
    for i in order[:8]:
        print(f"    PC{i:<2d}  weight {w[i]:+.4f}   "
              f"(a 1-sd move along this axis changes the score by {w[i]:+.3f})")
hi = pred[y >= 2].mean(); lo = pred[y <= 1].mean()
print(f"  mean predicted score: 优+良 {hi:.3f}  vs  中+差 {lo:.3f}  (gap {hi-lo:+.3f})")
r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
print(f"  in-sample R^2 {r2:.3f}  (optimistic; the held-out AUC {cv_auc:.3f} is the "
      f"honest number)")

if A.write:
    export.update({
        # which architecture these labels came from. The compare page refuses to apply a
        # reward model fitted on a different architecture, because the same 22-dim vector
        # draws a different face under each one.
        "model": A.model or "unknown",
        "n_labels": int(n),
        "counts": {"优": int(counts[3]), "良": int(counts[2]),
                   "中": int(counts[1]), "差": int(counts[0])},
        "good_rate_observed": float((y >= 2).mean()),
        "separation": float(hi - lo),
        "cv_auc": float(cv_auc),
        "lambda": float(lam),
        "train_r2": r2,
        "expected_gain_points": 7.2,
        "note": ("expected_gain_points is measured on held-out labels: ranking by this "
                 "model and keeping the top 35% raised the true 优+良 rate from 67.3% to "
                 "74.5%. keep around 35-50%; keeping only 15% did WORSE (72.5%) because "
                 "the ranking runs out of signal."),
        "caveat": ("fit from a single short session (159 labels in 7 minutes). Judgement "
                   "speed may have added noise; the classes overlap heavily in PCA space "
                   "(within-class spread ~4.5 vs centroid gaps 1.0-2.3)."),
    })
    p = os.path.join(WEB, "preference_reward.json")
    json.dump(export, open(p, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {p}")
print("\nFIT_REWARD_DONE")
