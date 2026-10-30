#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Tilt the sampling distribution instead of filtering samples.

THE USER'S FORMULATION, MADE EXACT
They described the goal as "twist the independent normal into a dependent normal with a
higher average score". If the score is quadratic, that twist is available in closed form:

    score           s(z) = c + a^T z + 1/2 z^T B z
    tilted density  p(z) ∝ exp(-1/2 z^T z) exp(beta s(z))
                       = exp(-1/2 z^T (I - beta B) z + beta a^T z)

which is a Gaussian with
    covariance  Sigma = (I - beta B)^{-1}
    mean        mu    = beta Sigma a

So sampling becomes one linear map z' = mu + L u with L L^T = Sigma and u ~ N(0,I). In
redstone that is a 4x4 constant matrix and a 4-vector offset -- 16 constant multiplies, no
true multipliers at all -- and it needs only ONE generated face instead of n rejected ones.

WHAT THIS SCRIPT MEASURES
  * how the predicted score improves as beta grows
  * how the covariance changes (the "dependency" the user asked for)
  * how the spread behaves, so a collapse would be visible
  * what the samples look like

The honest limit: the acceptable RATE of the tilted samples cannot be measured without new
labels, because the labels only exist at the original points. What is reported is the
predicted score and the geometry; the rate needs the next labelling round.
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
OUT = os.path.join(ROOT, "results", "tilt")
os.makedirs(OUT, exist_ok=True)
KNOB = ["PC0 明度", "PC1 发肤", "PC2 眼型", "PC3 脸型"]

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


# ---------------------------------------------------------------- fit the quadratic score
recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
Z = np.array([r["pca"][:4] for r in new], dtype=np.float64)
y = np.array([1.0 if r["grade"] >= 2 else 0.0 for r in new])
d = 4


def feats(P):
    n = len(P)
    F = [P]
    for i in range(d):
        F.append((P[:, i] ** 2)[:, None])
        for j in range(i + 1, d):
            F.append((P[:, i] * P[:, j])[:, None])
    return np.hstack(F)


Zc = Z - Z.mean(0)
F = feats(Zc)
A = np.c_[F, np.ones(len(F))]
LAM = 30.0
w = np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ y)
pred = A @ w
print("=" * 100)
print(f"QUADRATIC SCORE ON {len(new)} STYLE LABELS")
print("=" * 100)
print(f"  in-sample mean score: 可接受 {pred[y==1].mean():+.3f}  vs  中/差 {pred[y==0].mean():+.3f}")
hi = pred[y == 1][:, None]; lo = pred[y == 0][None, :]
print(f"  in-sample AUC {(hi>lo).mean()+0.5*(hi==lo).mean():.3f}  "
      f"(5-fold CV was 0.683, that is the honest number)")

# unpack into a, B  (s = c + a.z + 1/2 z^T B z)
a = w[:d].copy()
B = np.zeros((d, d))
off = d
for i in range(d):
    B[i, i] = 2 * w[off]          # 1/2 B_ii z_i^2 = w z_i^2  ->  B_ii = 2w
    off += 1
    for j in range(i + 1, d):
        B[i, j] = B[j, i] = w[off]
        off += 1
c = w[-1]

print()
print("  the quadratic form, in knob space")
print(f"    linear part  a = " + ", ".join(f"{v:+.3f}" for v in a))
print("    curvature    B:")
for i in range(d):
    print("      " + "  ".join(f"{B[i,j]:+8.3f}" for j in range(d)))

# ---------------------------------------------------------------- tilt
print()
print("=" * 100)
print("TILTING THE PRIOR:  p(z) ∝ N(0,I) * exp(beta * s(z))")
print("=" * 100)
print(f"  {'beta':>7s} {'mean shift':>11s} {'score mean':>11s} {'score p90':>10s} "
      f"{'sd ratio':>9s} {'det(Sigma)':>11s} {'off-diag':>9s}")
print("  " + "-" * 74)
rng = np.random.default_rng(0)
U = np.clip(rng.standard_normal((20000, d)), -3.5, 3.5)
base_score = feats(U - Z.mean(0)) @ w[:14] + c
tilt = {}
for beta in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
    M = np.eye(d) - beta * B
    try:
        Sigma = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print(f"  {beta:7.1f}   singular, skipped")
        continue
    ev = np.linalg.eigvalsh(Sigma)
    if ev.min() <= 0:
        print(f"  {beta:7.1f}   not positive definite, skipped")
        continue
    mu = beta * Sigma @ a
    Lc = np.linalg.cholesky(Sigma)
    S = U @ Lc.T + mu
    sc = feats(S - Z.mean(0)) @ w[:14] + c
    sd_ratio = float(np.mean(S.std(0) / U.std(0)))
    offd = float(np.mean(np.abs(Sigma - np.diag(np.diag(Sigma)))
                         / np.sqrt(np.outer(np.diag(Sigma), np.diag(Sigma)))))
    tilt[beta] = {"mu": mu.tolist(), "Sigma": Sigma.tolist(), "score": float(sc.mean()),
                  "p90": float(np.percentile(sc, 90)), "sd_ratio": sd_ratio,
                  "offdiag": offd, "samples": S}
    print(f"  {beta:7.1f} {np.linalg.norm(mu):11.2f} {sc.mean():11.3f} "
          f"{np.percentile(sc,90):10.3f} {sd_ratio:9.3f} "
          f"{np.linalg.det(Sigma):11.3f} {offd:9.2f}")

print()
print("  'mean shift' is how far the 4D centre moves; 'off-diag' is the average absolute")
print("  correlation that the tilt introduces -- this is the dependency that was asked for.")
print("  'sd ratio' near 1 means the spread is preserved; well below 1 is a collapse.")

# ---------------------------------------------------------------- pictures
net, kk = loadj("pca22_dim4x_model.json")


def render(P):
    with torch.no_grad():
        v = np.clip(net(torch.tensor(P.astype(np.float32))).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
    return layout.image_from_left_half(np.rint(v).astype(int))


BETAS = [b for b in (0.0, 1.0, 2.0, 3.0) if b in tilt]
COLS, TILE, PAD = 22, 40, 4
Wf = 150 + COLS * (TILE + PAD) + PAD
Hf = 86 + len(BETAS) * (TILE + 26) + 22
cv = Image.new("RGB", (Wf, Hf), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(19), font(12), font(9)
dr.text((10, 10), "把先验正态「扭转」而不筛选：一个 4×4 变换就能推高分数", font=f_t,
        fill=(234, 238, 244))
dr.text((10, 34), "z' = μ + L·u，u ~ N(0,I)。红石成本：16 次常数乘法 + 4 次加法，"
        "而且只生成 1 张，不需要拒绝采样。", font=f_s, fill=(150, 162, 178))
y = 66
for b in BETAS:
    t = tilt[b]
    imgs = render(t["samples"][:COLS])
    dr.text((10, y + 2), f"β = {b:g}", font=f_l, fill=(110, 168, 254))
    dr.text((10, y + 16), f"分数 {t['score']:+.2f}", font=f_s, fill=(150, 162, 178))
    for i, g in enumerate(imgs):
        gg = np.clip(g / 15.0 * 255.0, 0, 255)
        tt = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(tt.astype(np.uint8), mode="L").convert("RGB"),
                 (150 + i * (TILE + PAD), y))
    y += TILE + 26
dr.text((10, Hf - 18), "β 越大越偏向高分；看最后一行是否变得单调。", font=f_s,
        fill=(140, 152, 168))
p = os.path.join(OUT, "TILT.png")
cv.save(p)
print(f"\nwrote {p}")

json.dump({str(k): {kk2: vv for kk2, vv in v.items() if kk2 != "samples"}
           for k, v in tilt.items()},
          open(os.path.join(OUT, "tilt.json"), "w"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'tilt.json')}")
print("TILT_DONE")
