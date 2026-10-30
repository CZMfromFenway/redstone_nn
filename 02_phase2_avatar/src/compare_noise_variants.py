#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Head-to-head: the WITH-noise generator vs the user's NO-noise generator, same inputs,
same measuring stick. Also re-measures the speckle metrics, which is the property the
user actually complained about (麻麻赖赖 / too much colour speckle).

Runs compared
  sweep_pca22      : 22-dim condition + 64-dim z, 1500 steps
  pca22_long       : same, 10000 steps (if finished)
  pca22_nonoise    : 22-dim condition ONLY (no z), 6000 steps
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16


class Gen(nn.Module):
    def __init__(self, hid, zdim, cdim, use_noise=True):
        super().__init__()
        self.zdim, self.cdim, self.use_noise = zdim, cdim, use_noise
        in_dim = cdim + (zdim if use_noise else 0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 32 * NLEV))
    def logits(self, cond, z):
        if self.cdim == 0: x = z
        elif self.use_noise: x = torch.cat([cond, z], 1)
        else: x = cond
        return self.net(x).view(-1, 32, NLEV)


# ---- rebuild the PCA condition once -------------------------------------------------
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
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]

N = 400
bv = va[:N]
cond = torch.tensor(((X[bv] - mu[0]) @ comps.T) / sd[0], dtype=torch.float32)
real_img = full[bv]
Rtrue = torch.from_numpy(full[bv].astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)


def speckle(imgs):
    n = imgs.shape[0]
    agree, iso, blobs = [], [], []
    for k in range(n):
        g = imgs[k]
        pairs = same = 0
        for r in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r, c] == g[rr, cc]: same += 1
        agree.append(same / pairs)
        i = 0
        for r in range(8):
            for c in range(8):
                nb = [g[rr, cc] for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                      if 0 <= rr < 8 and 0 <= cc < 8]
                if nb and all(v != g[r, c] for v in nb): i += 1
        iso.append(i)
        seen = np.zeros((8, 8), bool); cnt = 0
        for r in range(8):
            for c in range(8):
                if seen[r, c]: continue
                lv = g[r, c]; st = [(r, c)]; seen[r, c] = True
                while st:
                    rr, cc = st.pop()
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        y, x = rr+dr, cc+dc
                        if 0 <= y < 8 and 0 <= x < 8 and not seen[y, x] and g[y, x] == lv:
                            seen[y, x] = True; st.append((y, x))
                cnt += 1
        blobs.append(cnt)
    return float(np.mean(agree)), float(np.mean(iso)), float(np.mean(blobs))


ref = speckle(real_img)
print("=" * 110)
print("REAL faces (16-level) as the smoothness reference")
print(f"  neighbour-agreement {ref[0]*100:5.1f}%   isolated px {ref[1]:5.2f}   blobs {ref[2]:5.1f}")
print("=" * 110)
print(f"{'run':16s} {'steps':>6s} {'CE':>7s} {'acc':>6s} {'Dacc':>6s} {'levels':>6s} "
      f"{'agree%':>7s} {'isolated':>8s} {'blobs':>6s} {'maxprob':>8s} {'entropy':>8s}")
print("-" * 110)

for tag in ("sweep_pca22", "pca22_long", "pca22_nonoise"):
    d = os.path.join(ROOT, "results", "sweep", tag)
    p = os.path.join(d, "final.pt")
    if not os.path.exists(p):
        print(f"{tag:16s}  (no checkpoint yet)")
        continue
    ck = torch.load(p, weights_only=False)
    a = ck["args"]
    un = not a.get("no_noise", False)
    G = Gen(a["hid"], a["zdim"], ck["cond_dim"], use_noise=un)
    G.load_state_dict(ck["G"]); G.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        lg = G.logits(cond, torch.randn(N, a["zdim"]))
        pr = F.softmax(lg, -1)
        mx = float(pr.max(-1).values.mean())
        ent = float((-(pr * pr.clamp_min(1e-12).log()).sum(-1)).mean())
        smp = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1).numpy()
        mod = lg.argmax(-1).numpy()
    acc_s = float((torch.from_numpy(smp) == Rtrue).float().mean())
    acc_m = float((torch.from_numpy(mod) == Rtrue).float().mean())
    ace, aci, acb = speckle(layout.image_from_left_half(mod))
    lv = np.mean([len(np.unique(mod[i])) for i in range(N)])
    print(f"{tag:16s} {a['steps']:6d} {0:7.4f} {max(acc_s,acc_m):6.4f} "
          f"{0:6.3f} {lv:6.2f} {ace*100:7.1f} {aci:8.2f} {acb:6.1f} {mx:8.3f} {ent:8.3f}")
    print(f"{'':16s}   acc: sample {acc_s:.4f}  mode {acc_m:.4f}")

print()
print("=" * 110)
print("Key columns")
print("  agree%   = fraction of pixel-neighbour pairs sharing a level (higher = smoother)")
print("  isolated = pixels whose four neighbours all differ (lower = smoother)")
print("  blobs    = connected same-level regions per image (lower = bigger colour patches)")
print("  maxprob  = mean of the max softmax probability; higher = a more confident model")
print("  entropy  = mean per-position entropy in nats; uniform would be ln16 = 2.773")
print("HEADS_UP_DONE")
