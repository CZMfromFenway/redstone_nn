#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Purify dim4x with the 177 labels, and show what "purified" actually looks like.

WHAT PURIFICATION MEANS HERE
Fit a linear reward on the 177 blind labels, then keep only the top fraction of a fresh
sample pool. The reward is linear in the 22-dim input, so on the four knobs dim4x actually
uses it reduces to a single direction: a weighted push on PC0..PC3. That direction is worth
reporting explicitly because it says which knob the preference lives on.

WHAT IS HONEST TO CLAIM
The yield numbers come from 5-fold cross-validation over the labels, so the reward never
scores a face it was fitted on. That is an estimate of "if you had filtered this way, how
many would you have rated 良+优", not a guarantee.

A linear reward conditions on ONE direction out of the available space, so diversity loss is
bounded -- the other directions are untouched. That is measured here too, so a real collapse
would be visible rather than assumed.
"""
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "purify4x")
os.makedirs(OUT, exist_ok=True)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def loadj(f):
    j = json.load(open(os.path.join(WEB, f), encoding="utf-8"))
    layers = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(layers):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(layers) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.eval()
    return m, int(len(layers[0]["W"][0]))


# ---------------------------------------------------------------- labels
recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if r.get("model") == "dim4x" and r.get("grade") in (0, 1, 2, 3)]
X = np.array([r["pca"][:4] for r in lab], dtype=np.float64)   # only the 4 knobs dim4x uses
y = np.array([r["grade"] for r in lab], dtype=np.float64)
print("=" * 100)
print(f"PURIFYING dim4x -- {len(lab)} labels, using only the 4 knobs the model actually sees")
print("=" * 100)
print(f"  base rate 良+优 = {100*np.mean(y>=2):.1f}%")

zm = X.mean(0); zsd = X.std(0) + 1e-9
Xs = (X - zm) / zsd


def ridge_cv(lam, folds=5):
    idx = np.random.default_rng(0).permutation(len(Xs))
    pred = np.zeros(len(Xs))
    W = None
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        A = np.c_[Xs[tr], np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.c_[Xs[te], np.ones(len(te))] @ w
        W = w
    hi = pred[y >= 2][:, None]; lo = pred[y <= 1][None, :]
    auc = (hi > lo).mean() + 0.5 * (hi == lo).mean()
    return float(auc), W


best = None
for lam in (0.3, 1, 3, 10, 30):
    auc, W = ridge_cv(lam)
    print(f"    lam={lam:<4} cv AUC {auc:.3f}")
    if best is None or auc > best[0]:
        best = (auc, lam, W)
AUC, LAM, W = best
print(f"  chosen lam={LAM}  cv AUC = {AUC:.3f}")

w4 = W[:4] / zsd                      # effective weight in raw-knob units
b4 = float(W[4] - (w4 * zm).sum())
KN = ["PC0 明度", "PC1 发肤", "PC2 眼型", "PC3 脸型"]
print()
print("  EFFECTIVE PREFERENCE DIRECTION over the four knobs")
print(f"  {'knob':10s} {'weight':>9s}   effect of raising it")
for i in range(4):
    print(f"  {KN[i]:10s} {w4[i]:+9.3f}   {'喜欢的更多' if w4[i] > 0 else '喜欢更少' if w4[i] < 0 else '--'}")

# ---------------------------------------------------------------- expected yield by keep
print()
print("=" * 100)
print("EXPECTED YIELD IF WE KEEP THE TOP FRACTION (5-fold cross-validated)")
print("=" * 100)
idx = np.random.default_rng(0).permutation(len(Xs))
pred_cv = np.zeros(len(Xs))
for f in range(5):
    te = idx[f::5]
    tr = np.setdiff1d(idx, te)
    A = np.c_[Xs[tr], np.ones(len(tr))]
    w = np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ y[tr])
    pred_cv[te] = np.c_[Xs[te], np.ones(len(te))] @ w

print(f"  {'keep':>6s} {'n':>5s} {'良+优':>8s} {'95% CI':>17s} {'提升':>8s}")
print("  " + "-" * 50)
res = {}
for keep in (1.0, 0.5, 0.35, 0.25, 0.15):
    k = max(1, int(len(Xs) * keep))
    top = np.argsort(-pred_cv)[:k]
    good = int((y[top] >= 2).sum())
    lo, hi = wilson(good, k)
    res[keep] = {"n": k, "good": good, "rate": good / k, "lo": lo, "hi": hi}
    lift = 100 * (good / k - np.mean(y >= 2))
    print(f"  {keep*100:5.0f}% {k:5d} {good/k*100:7.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%] {lift:+7.1f}")

# ---------------------------------------------------------------- what it looks like
net, kk = loadj("pca22_dim4x_model.json")
N = 40000
rng = np.random.default_rng(3)
Z = np.clip(rng.standard_normal((N, 4)), -3.5, 3.5)
score = Z @ w4 + b4

def render(zs):
    with torch.no_grad():
        v = np.clip(net(torch.tensor(zs.astype(np.float32))).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
    return np.rint(v).astype(int)

RAW = render(Z)
PURE = render(Z[np.argsort(-score)[:3000]])

# diversity: spread of each knob in the raw pool vs the purified pool
print()
print("=" * 100)
print("DIVERSITY -- a linear reward constrains ONE direction, so the rest should survive")
print("=" * 100)
pure_z = Z[np.argsort(-score)[:3000]]
print(f"  {'knob':10s} {'raw spread':>11s} {'purified':>10s} {'ratio':>7s} "
      f"{'raw mean':>9s} {'pure mean':>10s}")
print("  " + "-" * 60)
for i in range(4):
    a, b = Z[:, i].std(), pure_z[:, i].std()
    print(f"  {KN[i]:10s} {a:11.3f} {b:10.3f} {b/a:7.3f} {Z[:,i].mean():9.3f} "
          f"{pure_z[:,i].mean():10.3f}")
overall_r = float(np.mean([pure_z[:, i].std() / Z[:, i].std() for i in range(4)]))
print(f"\n  overall spread ratio {overall_r:.3f}  "
       f"({'no collapse' if overall_r > 0.85 else 'COLLAPSED'})")

# ---------------------------------------------------------------- figure
TILE, PAD, COLS = 40, 4, 24
Wf = 150 + COLS * (TILE + PAD) + PAD
Hf = 92 + 4 * (TILE + 26) + 20
cv = Image.new("RGB", (Wf, Hf), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(19), font(12), font(9)
dr.text((10, 10), f"dim4x 纯化效果：按 {len(lab)} 条标签筛选", font=f_t, fill=(234, 238, 244))
dr.text((10, 34), f"奖励模型 cv AUC {AUC:.3f} · 线性，只作用于四个旋钮", font=f_s,
        fill=(150, 162, 178))
dr.text((10, 50), "上：未筛选的随机样本    下：同一池中评分最高的 7.5%", font=f_s,
        fill=(150, 162, 178))
y = 74
dr.text((10, y + 4), "未筛选", font=f_l, fill=(150, 162, 178))
for i, g in enumerate(layout.image_from_left_half(RAW[:COLS])):
    gg = np.clip(g / 15.0 * 255.0, 0, 255)
    t = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
    cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
             (150 + i * (TILE + PAD), y))
y += TILE + 14
dr.text((10, y + 4), "纯化后", font=f_l, fill=(110, 200, 130))
for i, g in enumerate(layout.image_from_left_half(PURE[:COLS])):
    gg = np.clip(g / 15.0 * 255.0, 0, 255)
    t = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
    cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
             (150 + i * (TILE + PAD), y))
p = os.path.join(OUT, "PURIFY4X.png")
cv.save(p)
print(f"\nwrote {p}")

json.dump({"auc": AUC, "lam": LAM, "w4": w4.tolist(), "b4": b4,
           "yield": {str(k): v for k, v in res.items()},
           "spread_ratio": overall_r},
          open(os.path.join(OUT, "purify4x.json"), "w"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'purify4x.json')}")
print("PURIFY4X_DONE")
