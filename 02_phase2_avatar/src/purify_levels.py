#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Show several purification strengths side by side, with the diversity cost next to the yield.

THE COMPLAINT THIS ANSWERS
In the first purify figure the purified row looked almost identical face to face, and not
obviously better. That is a real defect, not a rendering artefact, and it exposes a flaw in
the metric used to justify purification:

    the cross-validated yield is a PER-FACE judgement.
    It asks "would this face be rated 良+优", one face at a time.
    It structurally CANNOT notice that forty such faces, seen together, are the same face.

So a 90% yield says nothing about whether the pool is worth looking at. Diversity has to be
tracked separately, and in the end the choice has to be made by eye.

WHAT IS MEASURED PER LEVEL
    keep        how much of the pool survives
    cv yield    expected 良+优 among the survivors (5-fold CV over the labels)
    PC1 sd      the spread of the knob the reward pushes hardest, relative to unfiltered
    overall sd  mean spread ratio over the four knobs
    contrast    mean contrast of the survivors
    washed      share of survivors that are low contrast
The last two catch the other failure mode: filtering can push the model into a corner of
input space where it produces flat or degraded faces even though the reward likes them.
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


# ---------------------------------------------------------------- fit on the 4 knobs
recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if r.get("model") == "dim4x" and r.get("grade") in (0, 1, 2, 3)]
X = np.array([r["pca"][:4] for r in lab], dtype=np.float64)
y = np.array([r["grade"] for r in lab], dtype=np.float64)
zm = X.mean(0); zsd = X.std(0) + 1e-9
Xs = (X - zm) / zsd
LAM = 30.0
A = np.c_[Xs, np.ones(len(Xs))]
w_z = np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ y)
w4 = w_z[:4] / zsd
b4 = float(w_z[4] - (w4 * zm).sum())
print(f"reward fitted on {len(lab)} labels, 4 knobs, weights "
      f"{', '.join(f'{v:+.3f}' for v in w4)}")

# 5-fold CV predictions for the honest yield at each keep fraction
idx = np.random.default_rng(0).permutation(len(Xs))
pred_cv = np.zeros(len(Xs))
for f in range(5):
    te = idx[f::5]
    tr = np.setdiff1d(idx, te)
    Atr = np.c_[Xs[tr], np.ones(len(tr))]
    w = np.linalg.solve(Atr.T @ Atr + LAM * np.eye(Atr.shape[1]), Atr.T @ y[tr])
    pred_cv[te] = np.c_[Xs[te], np.ones(len(te))] @ w

# ---------------------------------------------------------------- pool
net, kk = loadj("pca22_dim4x_model.json")
N = 60000
rng = np.random.default_rng(3)
Z = np.clip(rng.standard_normal((N, 4)), -3.5, 3.5)
score = Z @ w4 + b4


def render(zs):
    with torch.no_grad():
        v = np.clip(net(torch.tensor(zs.astype(np.float32))).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
    return np.rint(v).astype(int)


LEVELS = [("不筛", 1.00), ("保留 50%", 0.50), ("保留 35%", 0.35), ("保留 25%", 0.25),
          ("保留 15%", 0.15), ("保留 7.5%", 0.075), ("保留 3%", 0.03), ("保留 1%", 0.01)]

print()
print("=" * 108)
print("PURIFICATION STRENGTH vs DIVERSITY COST")
print("=" * 108)
print(f"  {'level':>12s} {'n kept':>7s} {'cv 良+优':>9s} {'PC1 sd':>8s} {'PC2 sd':>8s} "
      f"{'overall':>8s} {'contrast':>9s} {'washed':>8s} {'pairs':>9s}")
print("  " + "-" * 78)
rows = []
for name, keep in LEVELS:
    k = max(1, int(len(Xs) * keep))
    top_l = np.argsort(-pred_cv)[:k]
    good = int((y[top_l] >= 2).sum())
    lo_, hi_ = wilson(good, k)
    n_pool = max(1, int(N * keep))
    sel = np.argsort(-score)[:n_pool]
    z_sel = Z[sel]
    sd_rel = [z_sel[:, i].std() / Z[:, i].std() for i in range(4)]
    imgs = render(z_sel[:4000])
    flat = imgs.reshape(len(imgs), -1)
    con = flat.max(1) - flat.min(1)
    # PAIRWISE DIFFERENCE. Per-knob sd does not directly answer "do these look the same";
    # this does -- the average absolute level difference between two random survivors.
    npr = min(4000, len(flat))
    pr = np.random.default_rng(1).integers(0, npr, size=(4000, 2))
    selp = pr[:, 0] != pr[:, 1]
    pair_diff = float(np.abs(flat[pr[selp, 0]] - flat[pr[selp, 1]]).mean())
    rows.append({"name": name, "keep": keep, "good_rate": good / k, "ci": (lo_, hi_),
                 "pc1": sd_rel[1], "pc2": sd_rel[2], "overall": float(np.mean(sd_rel)),
                 "contrast": float(con.mean()), "washed": float(np.mean(con <= 5)),
                 "pair_diff": pair_diff,
                 # a random sample of the SURVIVORS, not the top of the ranking: the top
                 # is the extreme tail and makes every level look the same
                 "imgs": layout.image_from_left_half(
                     render(z_sel[np.random.default_rng(7).choice(
                         len(z_sel), size=min(20, len(z_sel)), replace=False)])),
                 "n_pool": n_pool})
    print(f"  {name:>12s} {n_pool:7d} {good/k*100:8.1f}% {sd_rel[1]:8.3f} {sd_rel[2]:8.3f} "
          f"{np.mean(sd_rel):8.3f} {con.mean():9.2f} {100*np.mean(con<=5):7.1f}% "
          f"{pair_diff:9.2f}")

# ---------------------------------------------------------------- figure
TILE, PAD, COLS = 44, 4, 20
LEFT = 236
Wf = LEFT + COLS * (TILE + PAD) + PAD
Hf = 84 + len(rows) * (TILE + 32) + 30
cv = Image.new("RGB", (Wf, Hf), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(19), font(12), font(9)
dr.text((10, 8), "不同程度的纯化：良率上去了，但脸越来越像同一张", font=f_t, fill=(234, 238, 244))
dr.text((10, 32), f"奖励模型：{len(lab)} 条标签，只用四个旋钮。" 
        "越往下保留越少，PC1（发肤）离散度掉得最快。", font=f_s, fill=(150, 162, 178))
dr.text((10, 48), "良率是逐张判断的指标，它看不出「一整排都是同一张脸」——所以最终要靠眼睛定。",
        font=f_s, fill=(224, 192, 90))
y = 74
for r in rows:
    dr.text((10, y + 2), r["name"], font=f_l, fill=(206, 214, 226))
    dr.text((10, y + 16), f"良率 {r['good_rate']*100:.0f}%", font=f_s, fill=(150, 162, 178))
    pc1 = r["pc1"]
    col = (110, 200, 130) if pc1 > 0.75 else ((224, 192, 90) if pc1 > 0.55 else (224, 122, 95))
    dr.text((96, y + 16), f"PC1 {pc1:.2f}", font=f_s, fill=col)
    dr.text((96, y + 30), f"两两差异 {r['pair_diff']:.1f}", font=f_s, fill=(150, 162, 178))
    for i, g in enumerate(r["imgs"]):
        gg = np.clip(g / 15.0 * 255.0, 0, 255)
        t = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (LEFT + i * (TILE + PAD), y))
    y += TILE + 32
dr.text((10, Hf - 20), "绿 PC1>0.75 安全　黄 0.55~0.75 偏窄　红 <0.55 已经塌陷",
        font=f_s, fill=(140, 152, 168))
p = os.path.join(OUT, "PURIFY_LEVELS.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump([{k: v for k, v in r.items() if k != "imgs"} for r in rows],
          open(os.path.join(OUT, "purify_levels.json"), "w"), indent=2, ensure_ascii=False)
print("PURIFY_LEVELS_DONE")
