#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Does NOT quantizing the output make learning easier?

We test four parameterizations of the SAME tiny problem (predict a pixel from its
3x3 neighbourhood on 8x8 grayscale faces) and measure how much of the real signal
each one recovers. This isolates the "continuous vs discrete output" question from
the "which loss" question.

  (1) continuous output  + MSE          -> learns the conditional MEAN
  (2) continuous output  + MSE, then rounded at display time
  (3) 16-class softmax   + cross-entropy, argmax at display time
  (4) 16-class softmax   + cross-entropy, sampled at display time

Metric that matters: how much VARIANCE across the dataset each one reproduces.
A conditional-mean predictor reproduces structure but destroys variation.

Relative paths only. Writes one PNG preview and one JSON.
"""
import os, json, time
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "L0_probe")
os.makedirs(RES, exist_ok=True)

rgb = np.load(os.path.join(DATA, "faces8_rgb_v2.npy"))
N = len(rgb)
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1) / 255.0        # (N,8,8) in 0..1
print(f"data {gray.shape}  range {gray.min():.3f}..{gray.max():.3f}")

# ---- build a neighbourhood-prediction task ----------------------------------
# Each output position sees its 3x3 neighbourhood in the CURRENT image, but we mask
# the CENTRE (that is the unknown we must predict). Left half only.
LEFT = [(r, c) for r in range(8) for c in range(4)]
nb_idx, tgt = [], []
P = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="edge")
for (r, c) in LEFT:
    nb = P[:, r:r + 3, c:c + 3].reshape(N, 9).copy()
    nb[:, 4] = 0.0                       # hide the centre
    nb_idx.append(nb)
    tgt.append(gray[:, r, c])
X = np.stack(nb_idx, 1)                   # (N,32,9)
Y = np.stack(tgt, 1)                      # (N,32) continuous target
print(f"X {X.shape}  Y {Y.shape}")

sub = np.random.default_rng(0).choice(N, 60000, replace=False)
X, Y = X[sub], Y[sub]

# Add position information so the model can be position-specific
POS = np.zeros((32, 32), dtype=np.float32)
for i in range(32):
    POS[i, i] = 1.0
# Canonical shapes, used consistently from here on:
#   X3 : (N, 32, 41)  features for every (sample, left-half position)
#   Y2 : (N, 32)      continuous target
#   XF : (N*32, 41)   flat view for minibatch training
#   YF : (N*32,)      flat view
X3 = np.concatenate([X, np.tile(POS[None], (len(X), 1, 1))], 2).astype(np.float32)
Y2 = Y.astype(np.float32)
XF = X3.reshape(-1, 41)
YF = Y2.reshape(-1)
NPIX = len(XF)
print(f"X3 {X3.shape}  Y2 {Y2.shape}  flat {XF.shape} / {YF.shape}")

H = 128
rng = np.random.default_rng(1)
W1 = rng.normal(0, np.sqrt(2 / 41), (41, H)).astype(np.float64)
b1 = np.zeros(H)
W2 = rng.normal(0, np.sqrt(2 / H), (H, 1)).astype(np.float64)
b2 = np.zeros(1)


def adam_fit(loss_kind, steps=3000, bs=512, lr=1e-3, ncls=16):
    """loss_kind: 'mse' or 'ce'. Returns predictions on the full flat set."""
    global W1, b1, W2, b2
    # MSE is a REGRESSION: the head must emit exactly one number per position.
    # Passing the default ncls=16 here would silently produce a 16-wide regression head.
    if loss_kind == "mse":
        ncls = 1
    rng2 = np.random.default_rng(2)
    W1 = rng.normal(0, np.sqrt(2 / 41), (41, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0, np.sqrt(2 / H), (H, max(ncls, 1)))
    b2 = np.zeros(max(ncls, 1))
    ps = [W1, b1, W2, b2]
    M = [np.zeros_like(p) for p in ps]
    V = [np.zeros_like(p) for p in ps]
    t = 0
    NPIX = len(XF)
    classes = np.clip((YF.reshape(-1, 1) * (ncls - 1) + 0.5).astype(int), 0, ncls - 1).ravel()
    for it in range(steps):
        idx = rng2.integers(0, NPIX, bs)
        # Xf is (N, 32, 41): flatten the batch AND the pixel dimension, as the training
        # target is per-(sample, position).
        xb = XF[idx]
        yb = YF[idx]
        b = len(xb)                       # use the REAL row count, not the requested bs
        h = np.maximum(xb @ W1 + b1, 0)
        t += 1
        if loss_kind == "mse":
            pred = (h @ W2 + b2).ravel()
            e = pred - yb
            gW2 = (h.T @ e[:, None]) * (2 / b)
            gb2 = e.mean() * 2 * np.ones(1)
            dh = (e[:, None] @ W2.T) * (h > 0)
        else:
            lg = h @ W2 + b2
            lg = lg - lg.max(1, keepdims=True)
            ex = np.exp(lg)
            sm = ex / ex.sum(1, keepdims=True)
            cb = classes[idx].reshape(-1)
            g = sm.copy()
            g[np.arange(b), cb] -= 1.0
            g /= b
            gW2 = h.T @ g
            gb2 = g.sum(0)
            dh = (g @ W2.T) * (h > 0)
        gW1 = xb.T @ dh
        gb1 = dh.sum(0)
        for k, gr in enumerate((gW1, gb1, gW2, gb2)):
            M[k] = 0.9 * M[k] + 0.1 * gr
            V[k] = 0.999 * V[k] + 0.001 * gr ** 2
            ps[k] -= lr * (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
        W1, b1, W2, b2 = ps
    # predict
    out = []
    for s in range(0, NPIX, 200000):
        xb = XF[s:s + 200000]
        h = np.maximum(xb @ W1 + b1, 0)
        if loss_kind == "mse":
            out.append((h @ W2 + b2).ravel())
        else:
            out.append(h @ W2 + b2)
    return np.concatenate(out) if loss_kind == "mse" else np.concatenate(out, 0)


def report(tag, pred_img, ref_img):
    """pred_img, ref_img: (n,32) float in 0..1 (or class indices / 16)."""
    v_pred = pred_img.var(0).mean()
    v_ref = ref_img.var(0).mean()
    # distinct levels per image
    q = np.clip((pred_img * 15 + 0.5).astype(int), 0, 15)
    dist = np.array([len(np.unique(q[i])) for i in range(min(4000, len(q)))])
    print(f"  {tag:34s} variance reproduced {v_pred/v_ref*100:5.1f}%   "
          f"mean distinct levels/image {dist.mean():5.2f}")
    return {"variance_reproduced": float(v_pred / v_ref), "distinct_levels": float(dist.mean())}


Yr = Y2
v_real = Yr.var(0).mean()
qr = np.clip((Yr * 15 + 0.5).astype(int), 0, 15)
d_real = np.array([len(np.unique(qr[i])) for i in range(4000)])
print(f"\n  {'REAL DATA':34s} variance {v_real:.5f}          "
      f"mean distinct levels/image {d_real.mean():5.2f}")
print()

res = {}

t0 = time.time()
p = adam_fit("mse", steps=2500)
res["continuous_MSE"] = report("(1) continuous + MSE", p.reshape(-1, 32), Yr)
print(f"      ({time.time()-t0:.0f}s)")

t0 = time.time()
p = adam_fit("ce", steps=2500, ncls=16)
lg = p.reshape(-1, 32, 16)
res["softmax_argmax"] = report("(3) 16-class + CE, argmax", (lg.argmax(2)) / 15.0, Yr)
# sampled
r3 = np.random.default_rng(3)
sm = np.exp(lg - lg.max(2, keepdims=True)); sm /= sm.sum(2, keepdims=True)
idx = np.array([r3.choice(16, p=sm[i, j]) for i in range(len(sm)) for j in range(32)]).reshape(len(sm), 32)
res["softmax_sample"] = report("(4) 16-class + CE, sampled", idx / 15.0, Yr)
print(f"      ({time.time()-t0:.0f}s)")

print()
print("READING: 'variance reproduced' near 100% means the model keeps the real")
print("structure of the data. Much below 100% means it collapsed toward a mean.")

# preview
def grid(imgs, cols=16, sc=8):
    rows_ = (len(imgs) + cols - 1) // cols
    cv = np.full((rows_ * 8, cols * 8), 40, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r_, c_ = divmod(i, cols)
        cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = np.clip(np.asarray(im) * 255, 0, 255).astype(np.uint8)
    big = np.kron(cv, np.ones((sc, sc), dtype=np.uint8))
    return Image.fromarray(big, mode="L").convert("RGB")


def mirror(half):
    out = np.zeros((8, 8))
    for i, (r, c) in enumerate(LEFT):
        out[r, c] = half[i]
        out[r, 7 - c] = half[i]
    return out


sel = np.random.default_rng(5).choice(len(Yr), 32, replace=False)
grid([mirror(Yr[i]) for i in sel], 16, 8).save(os.path.join(RES, "cmp_REAL.png"))
pm = adam_fit("mse", steps=2500).reshape(-1, 32)
grid([mirror(pm[i]) for i in sel], 16, 8).save(os.path.join(RES, "cmp_continuous_MSE.png"))
pc = adam_fit("ce", steps=2500, ncls=16).reshape(-1, 32, 16)
grid([mirror((pc[i].argmax(1)) / 15.0) for i in sel], 16, 8).save(os.path.join(RES, "cmp_softmax_argmax.png"))
print(f"\npreviews -> {RES}\\cmp_*.png")
json.dump(res, open(os.path.join(RES, "contin_vs_discrete.json"), "w"), indent=2)
print("CMP_DONE")

