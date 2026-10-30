#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Is PCA the right latent for a palette-classification model?

THE USER'S OBJECTION, WHICH IS SHARP
"We use PCA on three colour channels to generate 8 one-hot codes."

That is a real mismatch and it deserves a number rather than an opinion. PCA is a linear
decomposition of a CONTINUOUS 192-dimensional RGB image. The model then throws the continuous
part away and emits eight palette indices. So the latent axes carry "how much of RGB mixture k is
present", while the target carries "which of eight discrete colours sits at this position". Those
are different spaces.

WHAT IS MEASURED HERE
    explained variance     how much of the image variance the 8 components actually keep. If it
                           is low, the latent is lossy and the argmax is being asked to invent
                           detail that the input never carried.
    per-position variance  how much of the total variance lives in "which colour at a position",
                           which is what the model must decide
    a direct comparison    PCA latent versus a one-hot latent of the same dimension, on how well
                           a linear probe can predict the palette index at each position

The last one is the operative test: if a one-hot latent predicts the indices better, PCA really is
the wrong basis.
"""
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
OUT = os.path.join(ROOT, "results", "colour", "pca_check")
os.makedirs(OUT, exist_ok=True)

z = np.load(os.path.join(ROOT, "results", "colour", "cache.npz"))
rgb192 = z["rgb192"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb192)
print(f"colour data: {N} images, {rgb192.shape[1]} dims (8x8x3)")

X = rgb192 - rgb192.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(N, min(60000, N), replace=False)
U, S, Vt = np.linalg.svd(X[sub], full_matrices=False)
ev = (S ** 2) / (S ** 2).sum()
cum = np.cumsum(ev)
print()
print("=" * 92)
print("EXPLAINED VARIANCE OF THE COLOUR PCA")
print("=" * 92)
print(f"  {'components':>10s} {'explained':>10s} {'cumulative':>11s}")
for k in (4, 8, 12, 16, 24, 32, 48, 64):
    if k <= len(ev):
        print(f"  {k:10d} {ev[k-1]*100:9.2f}% {cum[k-1]*100:10.1f}%")
print()
print(f"  The model uses the first 8: they keep {cum[7]*100:.1f}% of the variance.")
print(f"  To reach 90% takes {int(np.searchsorted(cum, 0.90))+1} components; "
      f"95% takes {int(np.searchsorted(cum, 0.95))+1}.")

# how much variance is "which colour at this position"?
pal = json.load(open(os.path.join(ROOT, "webapp", "colour_pal8_model.json"), encoding="utf-8"))
PAL = np.array(pal["palette"], dtype=np.float64)
POS = rgb192.reshape(N, 64, 3)          # rgb192 is the full 8x8 grid, not the half
dd = ((POS[:, :, None, :] - PAL[None, None, :, :]) ** 2).sum(-1)
IDX = dd.argmin(-1)
# between-position variance of the index (i.e. structure) vs within-position variance
print()
print("=" * 92)
print("WHERE THE VARIANCE LIVES")
print("=" * 92)
tot = X.var(0).sum()
for p in range(4):
    pass
# variance explained by position means (the "average face") vs the rest
mu_pos = POS.mean(0)                       # (32,3)
struct = ((mu_pos - mu_pos.mean(0, keepdims=True)) ** 2).sum() * N
print(f"  total variance of the image set        {tot*N/1e3:10.1f}k")
print(f"  explained by the average-face pattern  {struct/1e3:10.1f}k  "
      f"({100*struct/(tot*N):.1f}%)")
print(f"  remaining per-image variation          {(tot*N-struct)/1e3:10.1f}k  "
      f"({100*(tot*N-struct)/(tot*N):.1f}%)")
print()
print("  In words: most of the variance is the SAME average face, and the model's job is the")
print("  remaining per-image part -- which is exactly what the 8 components have to carry.")

# ---- direct test: PCA latent vs one-hot latent, predicting the palette index
print()
print("=" * 92)
print("DIRECT TEST: which latent predicts the palette indices better?")
print("=" * 92)


def probe_accuracy(Phi, tgt, steps=4000, seed=0):
    """Train a tiny softmax regression Phi -> target index and return accuracy."""
    rng = np.random.default_rng(seed)
    n = len(Phi)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    W = np.zeros((Phi.shape[1], int(tgt.max()) + 1))
    b = np.zeros(W.shape[1])
    for _ in range(steps):
        bi = tr[rng.integers(0, len(tr), 512)]
        h = Phi[bi] @ W + b
        h -= h.max(1, keepdims=True)
        p = np.exp(h); p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(bi)), tgt[bi]] -= 1; g /= len(bi)
        W -= 0.5 * (Phi[bi].T @ g)
        b -= 0.5 * g.sum(0)
    h = Phi[te] @ W + b
    return float((h.argmax(1) == tgt[te]).mean())


# target: the palette index at one position (the hardest single decision)
pos = 14
tgt = IDX[:, pos].reshape(-1)   # one position out of 64
# latent A: the 8 colour PCs
PhiP = cpc[:, :8]
# latent B: 8 one-hot of a coarse 2x2x2 RGB quantisation (same dimension, categorical flavour)
def onehot_quant(colors, bits=2):
    q = np.clip((colors / 16.0 * (1 << bits)).astype(int), 0, (1 << bits) - 1)
    code = q[:, 0] * (1 << (2 * bits)) + q[:, 1] * (1 << bits) + q[:, 2]
    out = np.zeros((len(colors), 1 << (3 * bits)))
    out[np.arange(len(colors)), code] = 1
    return out


A_ = onehot_quant(rgb192)
# project the 64-dim one-hot down to 8 dims linearly, so dimensions match
mu = A_.mean(0)
Ua, Sa, Vta = np.linalg.svd(A_ - mu, full_matrices=False)
PhiO = (A_ - mu) @ Vta[:8].T

accP = probe_accuracy(PhiP, tgt)
accO = probe_accuracy(PhiO, tgt)
print(f"  预测第 {pos} 号位的调色板索引：")
print(f"    用 8 个彩色 PCA 分量        {100*accP:.1f}%")
print(f"    用 8 维颜色 one-hot 投影    {100*accO:.1f}%")
print(f"    多数类基线                   {100*np.bincount(tgt).max()/len(tgt):.1f}%")
print()
print("  如果 one-hot 明显更好，就说明 PCA 不是合适的潜变量。")
print("  两者接近则说明：8 维里能装下的信息差不多，问题不在基底选择。")

json.dump({"cum8": float(cum[7]), "cum12": float(cum[11]) if len(cum) > 11 else None,
           "acc_pca": accP, "acc_onehot": accO,
           "majority": float(np.bincount(tgt).max() / len(tgt))},
          open(os.path.join(OUT, "pca_check.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\nPCA_CHECK_DONE")
