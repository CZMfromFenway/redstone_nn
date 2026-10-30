#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Quantify "speckle" (the user's 麻麻赖赖) and compare real faces against the model's
output under both decodings.

The user's claim: after quantisation to 16 levels, REAL skin is smoother -- same-colour
regions form larger connected blobs -- while the generated images look like noise was
added. If true, this shows up as:

  * more connected components of the SAME level (more, smaller blobs) in generated images
  * lower fraction of pixels whose 4 neighbours share their level ("same-colour islands")
  * higher edge density (adjacent differences)
  * a higher count of ISOLATED pixels (all four neighbours differ)

This is exactly the failure the old project called "per-pixel independence".
"""
import os
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

N = 400
bv = va[:N]
cond = torch.tensor(((X[bv] - mu[0]) @ comps.T) / sd[0], dtype=torch.float32)
torch.manual_seed(0)
with torch.no_grad():
    lg = G.logits(cond, torch.randn(N, a["zdim"]))
    smp = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1).numpy()
    mod = lg.argmax(-1).numpy()

smp_img = layout.image_from_left_half(smp)
mod_img = layout.image_from_left_half(mod)
real_img = full[bv]


def speckle(imgs, tag):
    """imgs: (n,8,8) int levels."""
    n = imgs.shape[0]
    # 4-neighbour agreement (how often a pixel equals each of its neighbours)
    agree = []
    edges = []
    isolated = []
    for k in range(n):
        g = imgs[k]
        pairs = 0
        same = 0
        for r in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r, c] == g[rr, cc]:
                            same += 1
        agree.append(same / pairs)
        e = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
        edges.append(e)
        # isolated: pixel whose four neighbours all differ
        iso = 0
        for r in range(8):
            for c in range(8):
                nb = [g[rr, cc] for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                      if 0 <= rr < 8 and 0 <= cc < 8]
                if nb and all(v != g[r, c] for v in nb):
                    iso += 1
        isolated.append(iso)
    # connected components per level (4-connectivity), counting only real blobs
    comps_cnt = []
    for k in range(n):
        g = imgs[k]
        seen = np.zeros((8, 8), bool)
        cnt = 0
        for r in range(8):
            for c in range(8):
                if seen[r, c]:
                    continue
                lv = g[r, c]
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        y, x = rr+dr, cc+dc
                        if 0 <= y < 8 and 0 <= x < 8 and not seen[y, x] and g[y, x] == lv:
                            seen[y, x] = True
                            stack.append((y, x))
                cnt += 1
        comps_cnt.append(cnt)
    print(f"  {tag:22s} neighbour-agreement {np.mean(agree)*100:5.1f}%   "
          f"edge {np.mean(edges):5.2f}   isolated px {np.mean(isolated):5.2f}   "
          f"connected blobs {np.mean(comps_cnt):5.1f}")
    return {"agree": float(np.mean(agree)), "edge": float(np.mean(edges)),
            "isolated": float(np.mean(isolated)), "blobs": float(np.mean(comps_cnt))}


print("=" * 96)
print("SPECKLE METRICS  (higher neighbour-agreement / fewer blobs = smoother)")
print("=" * 96)
r_real = speckle(real_img, "REAL (16-level)")
r_mod = speckle(mod_img, "GENERATED mode")
r_smp = speckle(smp_img, "GENERATED sample")

print()
print("  the user's observation, as numbers:")
print(f"    real vs sample : neighbour-agreement {r_real['agree']/r_smp['agree']:.2f}x, "
      f"blobs {r_smp['blobs']/r_real['blobs']:.2f}x more, "
      f"isolated pixels {r_smp['isolated']/max(r_real['isolated'],1e-9):.1f}x more")
print(f"    mode is smoother than sample by: agreement "
      f"{(r_mod['agree']-r_smp['agree'])*100:+.1f} points, "
      f"isolated {r_mod['isolated']-r_smp['isolated']:+.2f} px")

print()
print("=" * 96)
print("IS THE CONDITION ITSELF ENOUGH?  (the user's 'drop the 64-d noise' question)")
print("=" * 96)
# how much does the image change when only z changes, condition held fixed?
torch.manual_seed(5)
outs = []
for t in range(6):
    with torch.no_grad():
        l = G.logits(cond[:64], torch.randn(64, a["zdim"]))
    outs.append(l.argmax(-1).numpy())
d = []
for x in range(6):
    for y in range(x + 1, 6):
        d.append(float((outs[x] != outs[y]).mean()))
print(f"  changing ONLY z (same 22 PCA values): {np.mean(d)*100:.1f}% of positions change")
print(f"  (0% would mean z is ignored entirely; 100% would mean z dominates the condition)")
# and how much does changing the condition change the image, z held fixed?
torch.manual_seed(6)
zfix = torch.randn(64, a["zdim"])
with torch.no_grad():
    l = G.logits(cond[:64], zfix).argmax(-1).numpy()
    l2 = G.logits(cond[64:128], zfix).argmax(-1).numpy()
print(f"  changing the CONDITION (different faces, same z): {float((l!=l2).mean())*100:.1f}% differ")
# variance decomposition on the continuous logits
torch.manual_seed(7)
Z = torch.randn(64, a["zdim"])
with torch.no_grad():
    base = G.logits(cond[:64], Z)
    zvar = []
    for t in range(8):
        zvar.append(G.logits(cond[:64], torch.randn(64, a["zdim"])))
    cvar = G.logits(cond[64:128], Z)
vz = float(torch.stack(zvar).var(0).mean())
vc = float((base - cvar).var().mean())
print(f"  variance from z (holding condition): {vz:.4f}")
print(f"  variance from condition (holding z): {vc:.4f}")
print(f"  -> condition explains {vc/(vc+vz)*100:.1f}% of the logit variance; "
      f"z explains {vz/(vc+vz)*100:.1f}%")
print("MEASURE2_DONE")
