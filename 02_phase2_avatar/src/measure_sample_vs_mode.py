#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Measure the difference between the two decoding modes, on the real trained model.

  sample  : draw a gray level from the 16-way distribution at each position
  mode    : take the single most likely level (argmax) at each position

The user observes that MODE looks better here. That is a real and interesting finding,
because for the earlier mixture model the opposite held (argmax collapsed the image).
This script puts numbers on it:

  * how peaked are the distributions? (max probability, entropy)
  * distinct gray levels per image under each mode
  * diversity ACROSS images under each mode (this is what sampling is for)
  * how often the two disagree
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16

RUN = os.path.join(ROOT, "results", "sweep", "sweep_pca22")
ck = torch.load(os.path.join(RUN, "final.pt"), weights_only=False)
a = ck["args"]


class Gen(torch.nn.Module):
    def __init__(self, hid, zdim, cdim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cdim + zdim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 32 * NLEV))
    def logits(self, cond, z):
        return self.net(torch.cat([cond, z], 1)).view(-1, 32, NLEV)


G = Gen(a["hid"], a["zdim"], 22); G.load_state_dict(ck["G"]); G.eval()

# rebuild the PCA condition for a few real faces, exactly as the sweep script does
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
LEVELS = np.linspace(0, 255, 16)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
lc, lr, rc = layout.POS_COL, layout.POS_ROW, layout.MIRROR_COL
m = (gyi[:, lr, lc] == gyi[:, lr, rc]).mean(1) >= 0.95
Gg = gray[m]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)

sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
va = sp["val"]

N = 256
bv = va[:N]
cond = torch.tensor(((X[bv] - mu[0]) @ comps.T) / sd[0], dtype=torch.float32)
Rtrue = torch.from_numpy(full[bv].astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)

torch.manual_seed(0)
with torch.no_grad():
    lg = G.logits(cond, torch.randn(N, a["zdim"]))
    pr = F.softmax(lg, -1)
    mx = pr.max(-1).values                       # (N,32) confidence per position
    ent = -(pr * torch.log(pr.clamp_min(1e-12))).sum(-1)   # (N,32) entropy in nats
    smp = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1).numpy()
    mode = lg.argmax(-1).numpy()

print("=" * 92)
print("HOW PEAKED IS THE MODEL'S DISTRIBUTION?")
print("=" * 92)
print(f"  max probability per position : mean {mx.mean():.3f}  median {mx.median():.3f}  "
      f"min {mx.min():.3f}")
print(f"  entropy per position (nats)  : mean {ent.mean():.3f}   (uniform = ln16 = {np.log(16):.3f})")
print(f"  positions where max prob > 0.5 : {float((mx>0.5).float().mean())*100:.1f}%")
print(f"  positions where max prob > 0.9 : {float((mx>0.9).float().mean())*100:.1f}%")
print("  -> the closer max-prob is to 1, the less sampling matters and the more argmax and")
print("     sampling agree. A flat distribution (entropy near ln16) means the model has no idea.")

print()
print("=" * 92)
print("SAMPLE vs MODE")
print("=" * 92)
print(f"  pixel accuracy vs the REAL right half:")
print(f"    sample {float((torch.from_numpy(smp)==Rtrue).float().mean()):.4f}"
      f"   mode {float((torch.from_numpy(mode)==Rtrue).float().mean()):.4f}")
print(f"  the two decodings disagree on {float((smp!=mode).mean())*100:.1f}% of positions")

lv_s = np.array([len(np.unique(smp[i])) for i in range(N)])
lv_m = np.array([len(np.unique(mode[i])) for i in range(N)])
print(f"  distinct levels per image:  sample {lv_s.mean():.2f}   mode {lv_m.mean():.2f}")

# DIVERSITY: this is the entire reason to sample at all.
# With a fixed condition, sampling should give DIFFERENT faces each draw;
# argmax gives one face, forever.
print()
print("  DIVERSITY ACROSS REPEATED DRAWS, with the condition held fixed:")
for tag, draws in (("sample", 8), ("mode", 8)):
    outs = []
    for t in range(draws):
        torch.manual_seed(1000 + t)
        with torch.no_grad():
            l = G.logits(cond[:16], torch.randn(16, a["zdim"]))
        outs.append((F.gumbel_softmax(l, 1.0, hard=True).argmax(-1).numpy() if tag == "sample"
                     else l.argmax(-1).numpy()))
    # mean pairwise Hamming distance between the draws of the same face
    d = []
    for i in range(16):
        for x in range(draws):
            for y in range(x + 1, draws):
                d.append(int((outs[x][i] != outs[y][i]).sum()))
    uniq_faces = len({outs[t][0].tobytes() for t in range(draws)})
    print(f"    {tag:7s}: mean pairwise Hamming {np.mean(d):5.2f} / 32 positions; "
          f"distinct faces for face #0 across {draws} draws: {uniq_faces}")

print()
print("=" * 92)
print("READING")
print("=" * 92)
print("  mode (argmax) = deterministic: one face per condition, and it is the single most")
print("                  likely level everywhere -> sharp, no random dither.")
print("  sample        = draws from the distribution -> variation, but also random dither")
print("                  in positions where the model is unsure.")
print("  Whichever 'looks better' depends on whether the model's uncertainty is real")
print("  (multi-modal, so sampling explores genuinely different faces) or just noise")
print("  (flat distribution, so sampling adds speckle). The numbers above say which.")
print("MEASURE_DONE")
