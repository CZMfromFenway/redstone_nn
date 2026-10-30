#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Cross-check the sweep's headline numbers independently, because one of them contradicts
an earlier statement ("the unconditional output is better than pure noise").

For results/sweep/sweep_uncond we
  1. reload the saved generator,
  2. regenerate samples with a KNOWN seed,
  3. recompute CE / accuracy / levels / near-real distance from scratch,
  4. compare against what summary.json claims,
  5. and compare the distribution of generated levels against the uniform distribution.
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

TAG = "sweep_uncond"
D_ = os.path.join(ROOT, "results", "sweep", TAG)
ck = torch.load(os.path.join(D_, "final.pt"), weights_only=False)
a = ck["args"]
print(f"args: mode={a['mode']} steps={a['steps']} hid={a['hid']} cdim={ck['cond_dim']}")
print(f"summary.json claims: {json.load(open(os.path.join(D_,'summary.json'),encoding='utf-8'))['history'][-1]}")

class Gen(torch.nn.Module):
    def __init__(self, hid, zdim, cdim):
        super().__init__()
        self.zdim, self.cdim = zdim, cdim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cdim + zdim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 32 * NLEV))
    def logits(self, cond, z):
        x = z if self.cdim == 0 else torch.cat([cond, z], 1)
        return self.net(x).view(-1, 32, NLEV)

G = Gen(a["hid"], a["zdim"], ck["cond_dim"]); G.load_state_dict(ck["G"]); G.eval()

full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
va = sp["val"]
full_t = torch.from_numpy(full.astype(np.int64))
Rtrue_all = full_t[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)

torch.manual_seed(0)
with torch.no_grad():
    n = 1024
    bv = va[:n]
    z = torch.zeros(n, a["zdim"])                     # deterministic evaluation
    lg = G.logits(torch.zeros(n, ck["cond_dim"]), z)
    ce = float(F.cross_entropy(lg.reshape(-1, NLEV), Rtrue_all[bv].reshape(-1)))
    acc = float((lg.argmax(-1) == Rtrue_all[bv]).float().mean())
    gen = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1).numpy()

print()
print("=== independently recomputed (unconditional generator, no condition) ===")
print(f"  held-out CE {ce:.4f}   accuracy {acc:.4f}")
print(f"  uniform baseline CE {np.log(16):.4f}, accuracy 0.0625")
print(f"  -> the generator is {acc/0.0625:.2f}x better than uniform, i.e. barely")

lv = np.array([len(np.unique(gen[i])) for i in range(n)])
print(f"  distinct levels per sample: mean {lv.mean():.2f} (max 16)")
# how close is the per-position marginal to uniform?
hist = np.bincount(gen.reshape(-1), minlength=NLEV) / gen.size
H = float(-(hist * np.log(np.clip(hist, 1e-12, 1))).sum())
print(f"  generated level histogram (fraction): {np.round(hist,3).tolist()}")
print(f"  entropy of that marginal: {H:.4f} nats  (uniform = {np.log(16):.4f})")
print(f"  -> {H/np.log(16)*100:.1f}% of maximum entropy = essentially uniform noise")

# is it really symmetric, and does it look like noise?
img = layout.image_from_left_half(gen)
print(f"  mirror symmetry: {(img[:, :, :4] == img[:, :, 4:][:, :, ::-1]).mean():.4f}")

# near-real distance
POOL = full[va[np.random.default_rng(5).choice(len(va), min(2000, len(va)), replace=False)]]
fg = img.reshape(n, -1).astype(np.int16); fp = POOL.reshape(len(POOL), -1).astype(np.int16)
dd = []
for s in range(0, n, 64):
    dd.extend((fg[s:s+64, None, :] != fp[None, :, :]).sum(2).min(1).tolist())
print(f"  nearest-real distance: mean {np.mean(dd):.2f}, min {np.min(dd)}")
print()
print("VERDICT: the unconditional generator produces essentially uniform noise.")
print("  My earlier remark that it was 'better than pure noise' was based on the")
print("  distinct-levels count (13.6), which a uniform random image also scores high on.")
print("  That metric cannot distinguish structure from noise; pixel accuracy can.")
print("VERIFY_DONE")

