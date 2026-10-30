#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Export BOTH generators for the web page, with a SHARED gallery.

Why a shared gallery: to compare "with noise" against "without noise" honestly, both
pages must start from the same real faces with the same PCA vectors. Here the gallery
rows are drawn once and reused, and the PCA basis is rebuilt once and shared, so the two
models see identical inputs.

Outputs (in 02_phase2_avatar/webapp/):
    pca22_model.json          the conditional GAN (condition + 64-d noise)
    pca22_nonoise_model.json  the same task with the noise removed (condition only)

Both also carry a `pooled_gallery`: all 48 rows and their PCA values, so the page can
switch model WITHOUT changing the face you are looking at.
"""
import hashlib, json, os
import numpy as np
import torch
import torch.nn as nn

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
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 32 * NLEV))

    def logits(self, cond, z):
        if self.cdim == 0:
            x = z
        elif self.use_noise:
            x = torch.cat([cond, z], 1)
        else:
            x = cond
        return self.net(x).view(-1, 32, NLEV)


# ---------------------------------------------------------------- shared PCA basis
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
ev = (S ** 2) / (S ** 2).sum()
print(f"shared PCA basis: 22 comps, {100*ev[:22].sum():.2f}% variance")

# ---------------------------------------------------------------- SHARED gallery rows
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
GAL_N = 24
shared_rows = va[np.random.default_rng(11).choice(len(va), GAL_N, replace=False)]

shared_gallery, pooled = [], []
for r in shared_rows:
    pca = ((X[r] - mu[0]) @ comps.T) / sd[0]
    entry = {"rgb": full[r].reshape(-1).tolist(),
             "pca": [round(float(v), 6) for v in pca]}
    shared_gallery.append(entry)
    pooled.append(entry)
print(f"shared gallery: {len(shared_gallery)} rows")

# ---------------------------------------------------------------- export each model
EXPORTS = [
    ("pca22_model.json", "sweep_pca22", True),
    ("pca22_nonoise_model.json", "pca22_nonoise", False),
]

outdir = os.path.join(ROOT, "webapp")
os.makedirs(outdir, exist_ok=True)

for fname, tag, use_noise in EXPORTS:
    run = os.path.join(ROOT, "results", "sweep", tag)
    ck = torch.load(os.path.join(run, "final.pt"), weights_only=False)
    a = ck["args"]
    un = not a.get("no_noise", False)          # trust the checkpoint, not our assumption
    G = Gen(a["hid"], a["zdim"], ck["cond_dim"], use_noise=un)
    G.load_state_dict(ck["G"]); G.eval()
    W1 = G.net[0].weight.detach().numpy(); b1 = G.net[0].bias.detach().numpy()
    W2 = G.net[2].weight.detach().numpy(); b2 = G.net[2].bias.detach().numpy()
    W3 = G.net[4].weight.detach().numpy(); b3 = G.net[4].bias.detach().numpy()
    print(f"\n{tag}: steps={a['steps']} use_noise={un} cdim={ck['cond_dim']} "
          f"in_dim={W1.shape[1]} hidden={a['hid']}")
    print(f"  W1 {W1.shape}  W2 {W2.shape}  W3 {W3.shape}")

    out = {
        "meta": {
            "source_run": tag,
            "use_noise": bool(un),
            "in_dim": int(W1.shape[1]),
            "steps_trained": int(a["steps"]),
            "batch": int(a["batch"]),
            "lr_g": float(a["lr_g"]), "lr_d": float(a["lr_d"]),
            "ce_weight": float(a["ce_weight"]),
            "r1_every": int(a["r1_every"]), "r1_gamma": float(a["r1_gamma"]),
            "zdim": int(a["zdim"]), "hid": int(a["hid"]), "pca_k": 22,
            "pca_variance_explained": float(ev[:22].sum()),
            "seed": int(a["seed"]),
            "layout": "canonical: p = col*8 + row over the left half; mirror within each row",
            "note": ("condition + latent noise" if un else
                     "condition ONLY (no latent z): the generator commits to one answer per condition"),
        },
        "levels": [float(v) for v in LEVELS],
        "pca": {"mean": [float(v) for v in mu[0]],
                "components": [[float(v) for v in row] for row in comps],
                "sd": [float(v) for v in sd[0]]},
        "gen": {"W1": W1.tolist(), "b1": b1.tolist(),
                "W2": W2.tolist(), "b2": b2.tolist(),
                "W3": W3.tolist(), "b3": b3.tolist()},
        "gallery": shared_gallery,
        "pooled_gallery": pooled,
    }
    p = os.path.join(outdir, fname)
    json.dump(out, open(p, "w", encoding="utf-8"))
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"  wrote {fname}  ({os.path.getsize(p)/1e6:.2f} MB)  sha256 {h[:16]}")

print("\nEXPORT_BOTH_DONE")
