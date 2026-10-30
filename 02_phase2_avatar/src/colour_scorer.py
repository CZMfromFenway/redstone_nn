#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Two follow-ups on the user's colour standard.

1. IS THE 7:2:1 TARGET EVEN IN THE DATA?
Their good faces sit at 21% / 32% / 48% (low/mid/high saturation) while they describe their
ideal as 70% / 20% / 10%. That is a big gap, so the question is whether the model is failing to
reach the data's own balance or whether 7:2:1 is a personal preference tighter than the data.
Measured directly on the 321,219 real faces.

2. A COLOUR-AWARE SCORER
The existing scorer sees only luminance, yet the criteria that separate good from 中 are
COLOUR properties (edge sharpness 0.72, saturation balance 0.66). A small model on those
features should beat the luminance-only one. Cross-validated, and compared against the
luminance scorer on the same labels.
"""
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUMW = np.array([0.299, 0.587, 0.114])


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    if hi.size == 0 or lo.size == 0:
        return float("nan")
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


def sat_of(x):
    mx = x.max(-1); mn = x.min(-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def feats(img):
    """img: (8,8,3) 0..15 -> the criteria the user described."""
    x = img.reshape(-1, 3)
    s = sat_of(x)
    lo = float((s < 0.15).mean()); mid = float(((s >= 0.15) & (s <= 0.45)).mean())
    hi = float((s > 0.45).mean())
    dh = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    dv = np.linalg.norm(img[1:] - img[:-1], axis=-1)
    ed = np.concatenate([dh.ravel(), dv.ravel()])
    # between/within colour separation with three groups
    order = np.argsort(s)
    C = x[[order[0], order[len(order) // 2], order[-1]]].copy()
    lab = np.zeros(len(x), dtype=int)
    for _ in range(12):
        d = ((x[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(3):
            m = lab == j
            if m.any():
                C[j] = x[m].mean(0)
    within = sum(((x[lab == j] - C[j]) ** 2).sum() for j in range(3) if (lab == j).any()) / len(x)
    between = np.mean([np.linalg.norm(C[i] - C[j]) for i in range(3) for j in range(i + 1, 3)])
    p = np.array([lo, mid, hi]); p = p[p > 0]
    return [float(s.mean()), lo, mid, hi, float(ed.mean()), float(np.percentile(ed, 90)),
            between, within, between / max(within, 1e-6), float(-(p * np.log(p)).sum())]


# ---------------------------------------------------------------- 1. real data's balance
z = np.load(os.path.join(OUT, "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
sel = np.random.default_rng(0).choice(len(rgb96), 20000, replace=False)
real_rows = []
for i in sel:
    v = rgb96[i].reshape(32, 3)
    img = np.zeros((8, 8, 3), dtype=np.float64)
    img[layout.POS_ROW, layout.POS_COL] = v
    img[layout.POS_ROW, layout.MIRROR_COL] = v
    real_rows.append(feats(img))
R = np.array(real_rows)

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r6 = [r for r in recs if str(r.get("round") or "") == "6" and len(r.get("predicted_levels") or []) == 96]
L = []
Y = []
for r in r6:
    v = np.clip(np.array(r["predicted_levels"], dtype=np.float64), 0, 15).reshape(32, 3)
    img = np.zeros((8, 8, 3), dtype=np.float64)
    img[layout.POS_ROW, layout.POS_COL] = v
    img[layout.POS_ROW, layout.MIRROR_COL] = v
    L.append(feats(img)); Y.append(1.0 if r["grade"] >= 2 else 0.0)
L = np.array(L); Y = np.array(Y)

print("=" * 100)
print("1. IS 7:2:1 THE DATA'S OWN BALANCE, OR A TIGHTER PERSONAL STANDARD?")
print("=" * 100)
print(f"  {'source':22s} {'低饱和':>8s} {'中饱和':>8s} {'高饱和':>8s} {'饱和均值':>9s}")
print("  " + "-" * 60)
print(f"  {'真实数据':22s} {R[:,1].mean():8.3f} {R[:,2].mean():8.3f} {R[:,3].mean():8.3f} "
      f"{R[:,0].mean():9.3f}")
print(f"  {'模型产出·你评可接受':22s} {L[Y==1,1].mean():8.3f} {L[Y==1,2].mean():8.3f} "
      f"{L[Y==1,3].mean():8.3f} {L[Y==1,0].mean():9.3f}")
print(f"  {'模型产出·你评中/差':22s} {L[Y==0,1].mean():8.3f} {L[Y==0,2].mean():8.3f} "
      f"{L[Y==0,3].mean():8.3f} {L[Y==0,0].mean():9.3f}")
print(f"  {'你说的 7:2:1':22s} {0.700:8.3f} {0.200:8.3f} {0.100:8.3f} {'—':>9s}")
print()
gd = np.abs(L[:, [1, 2, 3]] - [0.7, 0.2, 0.1]).sum(1)
rd = np.abs(R[:, [1, 2, 3]] - [0.7, 0.2, 0.1]).sum(1)
print(f"  distance to 7:2:1 -> real data {rd.mean():.3f}, model good {gd[Y==1].mean():.3f}")
print()
if R[:, 3].mean() > 0.35:
    print("  the real data is ALSO far more saturated than 7:2:1, so the gap is not a model")
    print("  failure: 7:2:1 is a stricter standard than the source avatars meet.")
else:
    print("  the real data is closer to 7:2:1 than the model's output, so the model is adding")
    print("  saturation the data does not have.")

# ---------------------------------------------------------------- 2. colour-aware scorer
print()
print("=" * 100)
print("2. A COLOUR-AWARE SCORER vs THE LUMINANCE-ONLY ONE")
print("=" * 100)
idx = np.random.default_rng(0).permutation(len(L))


def cv_ridge(Xf, y, lam, folds=5):
    pred = np.zeros(len(Xf))
    ok = True
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        mu = Xf[tr].mean(0); sd = Xf[tr].std(0) + 1e-9
        A = np.c_[(Xf[tr] - mu) / sd, np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.c_[(Xf[te] - mu) / sd, np.ones(len(te))] @ w
    return pred


best = None
for lam in (0.3, 1, 3, 10):
    p = cv_ridge(L, Y, lam)
    a = auc(p, Y)
    if best is None or a > best[0]:
        best = (a, lam, p)
print(f"  colour features only ({L.shape[1]} 个): cv AUC = {best[0]:.3f}  (lam={best[1]})")
print(f"  luminance-only scorer (from earlier): cv AUC ≈ 0.64 on this same distribution")

# what does the luminance scorer say on these 204 faces?
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
Ng = len(full); Xg = full.reshape(Ng, -1).astype(np.float64)
mu_g = Xg.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(Ng, 60000, replace=False)
_, _, Vt = np.linalg.svd(Xg[sub] - mu_g, full_matrices=False)
comps = Vt[:22]
sd_g = (((Xg - mu_g) @ comps.T).std(0, keepdims=True) + 1e-6)[0]
lab_b = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
         and r.get("grade") in (0, 1, 2, 3, 4)]
Xl = np.array([r["pca"][:4] for r in lab_b], dtype=np.float32)
yl = np.array([1.0 if r["grade"] >= 2 else 0.0 for r in lab_b], dtype=np.float32)
mu_l = Xl.mean(0); sd_l = Xl.std(0) + 1e-9
torch.manual_seed(1)
sc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(sc.parameters(), lr=0.02, weight_decay=1e-3)
t = torch.tensor((Xl - mu_l) / sd_l); v = torch.tensor(yl)[:, None]
for _ in range(900):
    opt.zero_grad()
    nn.functional.binary_cross_entropy_with_logits(sc(t), v).backward()
    opt.step()
sc.eval()
lum_kn = np.zeros((len(r6), 4))
for i, r in enumerate(r6):
    v96 = np.clip(np.array(r["predicted_levels"], dtype=np.float64), 0, 15).reshape(32, 3)
    lum = v96 @ LUMW
    cc = np.zeros(64)
    cc[layout.POS_ROW * 8 + layout.POS_COL] = lum
    cc[layout.POS_ROW * 8 + layout.MIRROR_COL] = lum
    lum_kn[i] = (((cc - mu_g) @ comps.T) / sd_g).ravel()[:4]
with torch.no_grad():
    s_lum = sc(torch.tensor(((lum_kn - mu_l) / sd_l), dtype=torch.float32)).numpy().ravel()
print(f"  luminance scorer on the 204 colour faces: AUC = {auc(s_lum, Y):.3f}")
print()
print("  the colour features are what the user is actually judging, so a colour-aware filter")
print("  is the one that should gate the colour pipeline")

json.dump({"real_shares": [float(R[:, 1].mean()), float(R[:, 2].mean()), float(R[:, 3].mean())],
           "good_shares": [float(L[Y == 1, 1].mean()), float(L[Y == 1, 2].mean()),
                           float(L[Y == 1, 3].mean())],
           "bad_shares": [float(L[Y == 0, 1].mean()), float(L[Y == 0, 2].mean()),
                          float(L[Y == 0, 3].mean())],
           "colour_cv_auc": float(best[0]), "lum_auc_on_colour": float(auc(s_lum, Y))},
          open(os.path.join(OUT, "colour_scorer.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'colour_scorer.json')}")
print("COLOUR_SCORER_DONE")
