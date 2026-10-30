#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Decisive diagnostic: can THIS generator architecture learn the data at all?

The adversarial runs failed identically in three configurations (plain, zero-init,
frozen-weak-D): D reached ~0.997 accuracy and G's loss did not move. Before spending
more time on GAN stabilisers, we must know whether the failure is

    (a) the ADVERSARIAL DYNAMICS            -> a teacher-forced model should work fine
    (b) the ARCHITECTURE / OPTIMISATION     -> teacher forcing will fail too

So we train the SAME generator MLP on a plain conditional cross-entropy task:
given a latent code z (fixed per image) and the 32 KNOWN left-half levels, predict
the 32 MASKED right-half levels. No discriminator involved.

If this reaches a low loss and produces face-like output, the architecture is fine
and the problem is purely adversarial. We also report the trivial baseline
(predict the dataset marginal) so the number means something.

Relative paths only. Existing files are never overwritten.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

AP = argparse.ArgumentParser()
AP.add_argument("--steps", type=int, default=8000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--lr", type=float, default=1e-3)
AP.add_argument("--hid", type=int, default=256)
AP.add_argument("--zdim", type=int, default=64)
AP.add_argument("--seed", type=int, default=0)
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--tag", default="diag_masked_ce")
A = AP.parse_args()

torch.manual_seed(A.seed); np.random.seed(A.seed)
torch.set_num_threads(A.threads)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "gan_v2", A.tag)
os.makedirs(OUT, exist_ok=True)
NLEV = 16

full = np.load(os.path.join(DATA, "L0_full_idx.npy"))          # (N,8,8)
sp = np.load(os.path.join(DATA, "L0_split.npz"))
tr, va = sp["train"], sp["val"]
print(f"data {full.shape}  train {len(tr):,}  val {len(va):,}")

# left half = cols 0..3 of each row (same convention as the GAN)
LCOL = np.array([0, 1, 2, 3])
RCOL = np.array([7, 6, 5, 4])


def make_batch(idx):
    """Returns (x_known, x_pred, mask) as flattened (B,32) long tensors.
    known = left half (4 cols x 8 rows), pred = right half."""
    x = torch.from_numpy(full[idx].astype(np.int64))                # (B,8,8)
    known = torch.cat([x[:, :, LCOL], x[:, :, RCOL]], dim=1)        # (B,8,8) all 8 cols
    # we only observe the LEFT 4 cols; predict the RIGHT 4 cols
    L = x[:, :, LCOL].reshape(len(idx), 32)                          # (B,32)
    R = x[:, :, RCOL].reshape(len(idx), 32)                          # (B,32)
    return L, R


print("=== training the SAME generator MLP with teacher forcing (no adversary) ===")
G = nn.Sequential(
    nn.Linear(32, A.hid), nn.ReLU(),
    nn.Linear(A.hid, A.hid), nn.ReLU(),
    nn.Linear(A.hid, 32 * NLEV),
)
nG = sum(p.numel() for p in G.parameters())
opt = torch.optim.Adam(G.parameters(), lr=A.lr)
print(f"params {nG:,}   input 32 (observed left half)   output 32x16 = 512 logits")

# the trivial baseline: predict the dataset's marginal distribution at each position
marg = np.zeros((32, NLEV), dtype=np.float64)
Rall = full[tr][:, :, RCOL].reshape(len(tr), 32)
for j in range(32):
    marg[j] = np.bincount(Rall[:, j], minlength=NLEV) / len(Rall)
marg = np.clip(marg, 1e-9, 1)
CE_marg = float(-(marg * np.log(marg)).sum(1).mean())
print(f"baseline CE (predict the marginal, ignoring the input): {CE_marg:.4f} nats")

t0 = time.time()
for step in range(1, A.steps + 1):
    bi = tr[np.random.default_rng(step).integers(0, len(tr), A.batch)]
    L, R = make_batch(bi)
    lg = G(L.float()).view(-1, 32, NLEV)
    loss = F.cross_entropy(lg.reshape(-1, NLEV), R.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0 or step == 1:
        with torch.no_grad():
            acc = float((lg.argmax(-1) == R).float().mean())
        print(f"  step {step:6d}  CE {loss.item():.4f}  acc {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# held-out evaluation
with torch.no_grad():
    bi = va[:4096]
    L, R = make_batch(bi)
    lg = G(L.float()).view(-1, 32, NLEV)
    ce = float(F.cross_entropy(lg.reshape(-1, NLEV), R.reshape(-1)))
    acc = float((lg.argmax(-1) == R).float().mean())
    # what does the model produce when its OWN predicted half is fed back? (free-running)
    cur = L.clone()
    for _ in range(4):
        lg2 = G(cur.float()).view(-1, 32, NLEV)
        cur = lg2.argmax(-1)
    free_acc = float((cur == R).float().mean())
print()
print(f"held-out: CE {ce:.4f}  acc {acc:.4f}   (marginal baseline CE {CE_marg:.4f})")
print(f"free-running accuracy (feed own predictions back 4x): {free_acc:.4f}")
print(f"  -> CE well below the marginal baseline means the architecture CAN learn structure")

# render
LEVELS = np.linspace(0, 255, 16)
def grid(imgs, cols=16, sc=8):
    rows_ = (len(imgs) + cols - 1) // cols
    cv = np.full((rows_ * 8, cols * 8), 40, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r_, c_ = divmod(i, cols)
        cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = np.clip(np.asarray(im), 0, 255).astype(np.uint8)
    big = np.kron(cv, np.ones((sc, sc), dtype=np.uint8))
    return Image.fromarray(big, mode="L").convert("RGB")

n = 64
bi = va[:n]
L, R = make_batch(bi)
with torch.no_grad():
    Pr = G(L.float()).view(-1, 32, NLEV).argmax(-1)
real = np.zeros((n, 8, 8), dtype=np.uint8)
pred = np.zeros((n, 8, 8), dtype=np.uint8)
for k in range(n):
    real[k][:, LCOL] = L[k].numpy().reshape(8, 4)
    real[k][:, RCOL] = R[k].numpy().reshape(8, 4)
    pred[k][:, LCOL] = L[k].numpy().reshape(8, 4)
    pred[k][:, RCOL] = Pr[k].numpy().reshape(8, 4)
grid([LEVELS[real[k]] for k in range(n)]).save(os.path.join(OUT, "masked_REAL.png"))
grid([LEVELS[pred[k]] for k in range(n)]).save(os.path.join(OUT, "masked_PRED.png"))
print(f"previews -> {OUT}\\masked_*.png")
json.dump({"params": nG, "heldout_ce": ce, "heldout_acc": acc,
           "marginal_baseline_ce": CE_marg, "free_running_acc": free_acc,
           "steps": A.steps, "seconds": round(time.time() - t0, 1)},
          open(os.path.join(OUT, "diag.json"), "w"), indent=2)
print("DIAG_DONE")
