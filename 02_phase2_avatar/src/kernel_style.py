#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Discover styles with a LEARNABLE BANK OF TINY CONVOLUTION KERNELS.

The user's idea: instead of hand-designing "style", learn a set of 2x2 and 3x3
convolution kernels and let them specialise. Some should fire on large flat colour
regions, some on "dark hair on top / light skin below", some on the eye region, etc.
Then a face's style = the pattern of its strongest kernel responses.

Why this is the right shape of idea for redstone: a 2x2 or 3x3 convolution with a few
integer weights is EXACTLY the kind of local, feed-forward, bounded-fan-in operation
redstone is good at. If a handful of kernels suffice to describe style, the whole
condition becomes "a few comparator sums", not a lookup table.

Design: a small convolutional autoencoder trained by MSE on the 16-level gray faces.
The 9 kernels are visualised, and the 32-dim bottleneck (which we deliberately keep at
32 so a discrete code can be carved out of it) becomes the style vector.
A k-means over that bottleneck gives the discrete style code, and we report how many
clusters are actually used and what each cluster looks like.

Relative paths only. Existing files are never overwritten.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)

NLEV = 16
LEVELS = np.linspace(0, 255, 16)

AP = argparse.ArgumentParser()
AP.add_argument("--tag", default="kernel_style")
AP.add_argument("--steps", type=int, default=4000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--lr", type=float, default=1e-3)
AP.add_argument("--embed", type=int, default=9, help="number of kernels to learn")
AP.add_argument("--code-k", type=int, default=4, help="style code bits -> 2^k clusters")
AP.add_argument("--seed", type=int, default=0)
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--deadline", default="")
A = AP.parse_args()

torch.manual_seed(A.seed); np.random.seed(A.seed)
torch.set_num_threads(A.threads)
DATA = os.path.join(ROOT, "data_v2_canonical")
OUT = os.path.join(ROOT, "results", "kernel_style", A.tag)
os.makedirs(OUT, exist_ok=True)


def past_deadline():
    if not A.deadline:
        return False
    h, m = A.deadline.split(":")
    now = time.localtime()
    return (now.tm_hour * 60 + now.tm_min) >= (int(h) * 60 + int(m))


full = np.load(os.path.join(DATA, "L0_full_idx.npy"))       # (N,8,8)
sp = np.load(os.path.join(DATA, "L0_split.npz"))
tr, va = sp["train"], sp["val"]
print(f"data {full.shape}  train {len(tr):,}")

# input: one 16-level channel, normalised to [-1,1]
X = torch.from_numpy(full.astype(np.float32)).unsqueeze(1) / 15.0 * 2 - 1   # (N,1,8,8)


class KernelAE(nn.Module):
    """2x2 and 3x3 kernels -> responses -> a compact style vector -> reconstruction.

    Keeping the convolution kernels 2x2/3x3 is deliberate: that is the size redstone
    can implement with a bounded fan-in.
    """
    def __init__(self, embed):
        super().__init__()
        self.k3 = nn.Conv2d(1, embed, 3, padding=1, bias=False)   # 3x3 kernels
        self.k2 = nn.Conv2d(1, embed, 2, padding=0, bias=False)   # 2x2 kernels
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(2 * embed * 64, 32), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x):
        r3 = torch.relu(self.k3(x))                              # (B,embed,8,8)
        r2 = torch.relu(self.k2(x))                              # (B,embed,7,7)
        r2 = F.pad(r2, (0, 1, 0, 1))                             # align to 8x8
        resp = torch.cat([r3, r2], 1)
        z = self.enc(resp)
        rec = self.dec(z).view(-1, 8, 8)
        return rec, z, r3, r2


M = KernelAE(A.embed)
opt = torch.optim.Adam(M.parameters(), lr=A.lr)
nP = sum(p.numel() for p in M.parameters())
print(f"kernels: {A.embed} of 3x3 + {A.embed} of 2x2   params {nP:,}")

rng = np.random.default_rng(0)
t0 = time.time()
for step in range(1, A.steps + 1):
    if step % 500 == 0 and past_deadline():
        print(f"  deadline at step {step}", flush=True); break
    bi = tr[rng.integers(0, len(tr), A.batch)]
    xb = X[bi]
    rec, z, _, _ = M(xb)
    tgt = xb.squeeze(1)
    loss = F.mse_loss(rec, tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1:
        print(f"  step {step:5d}  recon MSE {loss.item():.5f}  ({time.time()-t0:.0f}s)", flush=True)

# ---- analyse the learned kernels -------------------------------------------------
with torch.no_grad():
    k3 = M.k3.weight.detach().cpu().numpy().reshape(A.embed, 9)
    k2 = M.k2.weight.detach().cpu().numpy().reshape(A.embed, 4)
print()
print("=== learned 3x3 kernels (row-major, rounded to 2 dp) ===")
for i in range(A.embed):
    v = k3[i]
    print(f"  k3[{i}]: [{v[0]:6.2f}{v[1]:6.2f}{v[2]:6.2f}] [{v[3]:6.2f}{v[4]:6.2f}{v[5]:6.2f}] "
          f"[{v[6]:6.2f}{v[7]:6.2f}{v[8]:6.2f}]   |w|={np.abs(v).sum():5.2f}")
print("=== learned 2x2 kernels ===")
for i in range(A.embed):
    v = k2[i]
    print(f"  k2[{i}]: [{v[0]:6.2f}{v[1]:6.2f}] [{v[2]:6.2f}{v[3]:6.2f}]   |w|={np.abs(v).sum():5.2f}")

# ---- responses and a discrete style code ----------------------------------------
with torch.no_grad():
    Z = []
    for s in range(0, len(full), 20000):
        xb = X[s:s + 20000]
        _, z, _, _ = M(xb)
        Z.append(z)
    Z = torch.cat(Z).numpy()
print(f"\nstyle embedding: {Z.shape}")

K = 2 ** A.code_k
rngc = np.random.default_rng(2)
C = Z[rngc.choice(len(Z), K, replace=False)].copy()
for _ in range(30):
    d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(2)
    lab = d.argmin(1)
    for j in range(K):
        if (lab == j).any():
            C[j] = Z[lab == j].mean(0)
prior = np.bincount(lab, minlength=K) / len(lab)
eff = float(np.exp(-(prior * np.log(np.clip(prior, 1e-12, 1))).sum()))
print(f"style code: {A.code_k} bits -> {K} clusters, effective {eff:.2f}, "
      f"sizes {np.bincount(lab, minlength=K).tolist()}")

# what does each cluster look like? average face per cluster
means = np.zeros((K, 8, 8), dtype=np.uint8)
for j in range(K):
    if (lab == j).any():
        means[j] = np.clip(full[lab == j].mean(0), 0, 15).astype(np.uint8)


def grid(imgs, cols=16, sc=8):
    rows_ = (len(imgs) + cols - 1) // cols
    cv = np.full((rows_ * 8, cols * 8), 40, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r_, c_ = divmod(i, cols)
        cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = np.clip(np.asarray(im), 0, 255).astype(np.uint8)
    big = np.kron(cv, np.ones((sc, sc), dtype=np.uint8))
    return Image.fromarray(big, mode="L").convert("RGB")


grid([LEVELS[means[j]] for j in range(K)], min(K, 16)).save(os.path.join(OUT, "cluster_mean_faces.png"))
# a few real examples per cluster, for interpretation
samp = []
for j in range(K):
    idx = np.flatnonzero(lab == j)
    if len(idx):
        pick = np.random.default_rng(j).choice(idx, min(8, len(idx)), replace=False)
        samp.extend([full[i] for i in pick])
grid([LEVELS[s] for s in samp], 8).save(os.path.join(OUT, "cluster_examples.png"))

# reconstruction check: does the embedding keep enough to rebuild the face?
with torch.no_grad():
    rec, _, _, _ = M(X[va[:2000]])
    mse = float(F.mse_loss(rec, X[va[:2000]].squeeze(1)))
    recq = np.clip(np.round((rec.numpy() + 1) / 2 * 15), 0, 15).astype(np.uint8)
grid([LEVELS[recq[i]] for i in range(32)], 16).save(os.path.join(OUT, "recon_from_style.png"))
grid([LEVELS[full[va[i]]] for i in range(32)], 16).save(os.path.join(OUT, "recon_REAL.png"))
print(f"held-out reconstruction MSE (grayscale units/255): {mse:.5f}")

np.save(os.path.join(OUT, "embedding.npy"), Z.astype(np.float32))
np.save(os.path.join(OUT, "labels.npy"), lab.astype(np.int32))
json.dump({"embed": A.embed, "kernels_3x3": k3.tolist(), "kernels_2x2": k2.tolist(),
           "code_k": A.code_k, "K": K, "prior": prior.tolist(), "effective_clusters": eff,
           "recon_mse": mse, "params": nP, "steps": A.steps,
           "seconds": round(time.time() - t0, 1)},
          open(os.path.join(OUT, "kernel_info.json"), "w"), indent=2)
print(f"wrote {OUT}")
print("KERNEL_STYLE_DONE")
