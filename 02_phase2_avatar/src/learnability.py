#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
IS THERE ANY LEARNABLE STRUCTURE IN "GOODNESS" AT ALL?

The CEM simulation improved only ~1 unit of Hamming distance after 100 labels, and
restricting the search to 4 or 8 dimensions did not help. That rules out "too many
dimensions" as the explanation. The remaining explanations are:

  (a) the score function is mostly NOISE in this space, so nothing can be learnt
  (b) the score is learnable but CEM's update is poorly tuned

This script separates them. It fits several regressors from the 22 style coordinates to
the proxy score on a training half and measures R^2 on the other half:

  * ridge regression (linear structure)
  * k-nearest-neighbour regression (local structure, any shape)
  * a small MLP (smooth nonlinear structure)

If every model gets R^2 near 0, the proxy carries almost no smooth signal in this space,
and a subjective-labeling campaign would have to rely on the user's eye providing a
MUCH cleaner signal than the proxy does. If some model gets a meaningful R^2, then the
signal exists and the sampling strategy is worth tuning.

Also reports how much of the proxy's variance is between-position (structural) versus
noise, and the correlation between the proxy and the simple speckle metrics, which is
what the user actually complained about.
"""
import json, os
import numpy as np
import torch
import torch.nn as nn

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
OUT = os.path.join(ROOT, "results", "learnability")
os.makedirs(OUT, exist_ok=True)


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


run = os.path.join(ROOT, "results", "coherence", "final_lam5_20k")
ck = torch.load(os.path.join(run, "final.pt"), weights_only=False)
G = Gen(ck["args"]["hid"], 22); G.load_state_dict(ck["G"]); G.eval()

va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
POOL = full[va[:3000]].reshape(3000, -1).astype(np.int16)

N = 4000
rng = np.random.default_rng(0)
Z = np.clip(rng.standard_normal((N, 22)), -3.5, 3.5)
Zt = torch.tensor(Z.astype(np.float32))
with torch.no_grad():
    dec = G.logits(Zt).argmax(-1).numpy()
imgs = layout.image_from_left_half(dec)                    # (N,8,8)
flat = imgs.reshape(N, -1).astype(np.int16)

# ---- the proxy: negative nearest-real distance ------------------------------------
d = np.empty(N)
for s in range(0, N, 100):
    blk = flat[s:s + 100]
    d[s:s + 100] = (blk[:, None, :] != POOL[None, :, :]).sum(2).min(1)
y = -d
print(f"proxy over {N} random samples: mean {y.mean():.2f}  sd {y.std():.2f}  "
      f"range {y.min():.0f}..{y.max():.0f}")

# ---- speckle metrics, to see what the proxy actually tracks ------------------------
def agree_of(g):
    pairs = same = 0
    for r in range(8):
        for c in range(8):
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    pairs += 1
                    if g[r, c] == g[rr, cc]:
                        same += 1
    return same / pairs

AG = np.array([agree_of(imgs[i]) for i in range(N)])
lv = np.array([len(np.unique(dec[i])) for i in range(N)])
print(f"neighbour agreement mean {AG.mean():.3f}   distinct levels mean {lv.mean():.2f}")
print(f"corr(proxy, agreement) = {np.corrcoef(y, AG)[0,1]:+.3f}   "
      f"corr(proxy, levels) = {np.corrcoef(y, lv)[0,1]:+.3f}")
print("  (a strong correlation means the proxy mostly tracks the speckle metric)")

# ---- can the proxy be predicted from the 22 style coordinates? ---------------------
X = Z
n = len(X)
perm = np.random.default_rng(1).permutation(n)
tr, te = perm[:n // 2], perm[n // 2:]


def r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def ridge(Xtr, ytr, Xte, lam=1.0):
    Xtr1 = np.c_[Xtr, np.ones(len(Xtr))]
    Xte1 = np.c_[Xte, np.ones(len(Xte))]
    A = Xtr1.T @ Xtr1 + lam * np.eye(Xtr1.shape[1])
    w = np.linalg.solve(A, Xtr1.T @ ytr)
    return Xte1 @ w


def knn_reg(Xtr, ytr, Xte, k=10):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(2)
    idx = np.argsort(d, 1)[:, :k]
    return ytr[idx].mean(1)


class MLP(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(22, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_fit(Xtr, ytr, Xte, steps=4000, hid=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = MLP(hid)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    yt = (yt - yt.mean()) / (yt.std() + 1e-9)
    for _ in range(steps):
        loss = ((m(Xt) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = m(torch.tensor(Xte, dtype=torch.float32)).numpy()
    return pred * ytr.std() + ytr.mean()


print()
print("=" * 92)
print("CAN 'GOODNESS' BE PREDICTED FROM THE 22 COORDINATES?  (train on half, test on half)")
print("=" * 92)
res = {}
p = ridge(X[tr], y[tr], X[te], lam=1.0)
res["ridge (linear)"] = r2(y[te], p)
print(f"  ridge (linear)          R^2 = {res['ridge (linear)']:+.4f}")
p = knn_reg(X[tr], y[tr], X[te], k=10)
res["kNN k=10 (local)"] = r2(y[te], p)
print(f"  kNN k=10 (local)        R^2 = {res['kNN k=10 (local)']:+.4f}")
p = mlp_fit(X[tr], y[tr], X[te])
res["MLP (nonlinear)"] = r2(y[te], p)
print(f"  MLP 128x128 (nonlinear) R^2 = {res['MLP (nonlinear)']:+.4f}")
# a control: predict a random target with the same models -> should be ~0
yr = np.random.default_rng(2).permutation(y)
pr = ridge(X[tr], yr[tr], X[te], lam=1.0)
print(f"  control: ridge on a shuffled target  R^2 = {r2(yr[te], pr):+.4f}  (should be ~0)")

best = max(res.values())
print()
if best < 0.02:
    print("  VERDICT: essentially NO learnable structure. Even a flexible nonlinear model")
    print("           explains under 2% of the proxy's variance. A labelling campaign")
    print("           would be optimising noise unless the user's judgement is a much")
    print("           cleaner signal than this proxy.")
elif best < 0.10:
    print("  VERDICT: a weak but real signal. Enough to improve slowly with many labels;")
    print("           100 labels will give a small, hard-to-notice gain.")
else:
    print(f"  VERDICT: a real signal (best R^2 {best:.3f}). Preference-guided sampling")
    print("           should give a measurable gain; the earlier CEM run was under-tuned.")

# how much of the variance is even reachable: compare the proxy's spread with the
# spread attributable to the best model's predictions
json.dump({"N": N, "proxy_mean": float(y.mean()), "proxy_sd": float(y.std()),
           "corr_agree": float(np.corrcoef(y, AG)[0, 1]),
           "corr_levels": float(np.corrcoef(y, lv)[0, 1]),
           "r2": res, "best_r2": float(best)},
          open(os.path.join(OUT, "learnability.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'learnability.json')}")
print("LEARNABILITY_DONE")
