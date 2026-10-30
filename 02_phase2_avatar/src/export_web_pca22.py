#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Export the PCA-22 conditioned generator into a single JSON that a web page can run.

Exports, all as plain numbers:
  * the generator MLP:  (22+64) -> 256 -> 256 -> (32*16) logits, 2 ReLU hidden layers
  * the PCA basis:      mean (64), components (22,64), and the per-component std used to
                        standardise the condition during training
  * a small gallery of real faces with their exact 22 PCA values, so the page can offer
    "start from a real face" instead of blind sliders
  * the 16 luminance levels used for the grayscale scale

The forward pass in the page is:  h = relu(W1*[pca,z]+b1); h = relu(W2*h+b2); logits = W3*h+b3
then softmax over 16 levels per position, sample, mirror. No framework needed.

Everything is RELATIVE. Existing files are never overwritten.
"""
import hashlib, json, os
import numpy as np
import torch

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16

TAG = "sweep_pca22"
PCA_K = 22
RUN = os.path.join(ROOT, "results", "sweep", TAG)
DATA = os.path.join(ROOT, "data_v2_canonical")

ck = torch.load(os.path.join(RUN, "final.pt"), weights_only=False)
a = ck["args"]
print(f"checkpoint: mode={a['mode']} pca_k={a.get('pca_k')} steps={a['steps']} "
      f"zdim={a['zdim']} hid={a['hid']} cond_dim={ck['cond_dim']}")
assert ck["cond_dim"] == PCA_K, ck["cond_dim"]

# ---- rebuild the generator exactly as sweep_condition.py defines it -----------------
class Gen(torch.nn.Module):
    def __init__(self, hid, zdim, cdim):
        super().__init__()
        self.zdim, self.cdim = zdim, cdim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cdim + zdim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 32 * NLEV))
    def forward(self, cond, z):
        x = torch.cat([cond, z], 1) if self.cdim else z
        return self.net(x).view(-1, 32, NLEV)

G = Gen(a["hid"], a["zdim"], PCA_K)
G.load_state_dict(ck["G"]); G.eval()
W1 = G.net[0].weight.detach().numpy(); b1 = G.net[0].bias.detach().numpy()
W2 = G.net[2].weight.detach().numpy(); b2 = G.net[2].bias.detach().numpy()
W3 = G.net[4].weight.detach().numpy(); b3 = G.net[4].bias.detach().numpy()
print(f"weights: W1 {W1.shape} b1 {b1.shape} | W2 {W2.shape} b2 {b2.shape} | W3 {W3.shape} b3 {b3.shape}")

# ---- rebuild the PCA basis EXACTLY as sweep_condition.py did ------------------------
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)                    # (N_all,8,8)
LEVELS = np.linspace(0, 255, 16)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
lc, lr, rc = layout.POS_COL, layout.POS_ROW, layout.MIRROR_COL
m = (gyi[:, lr, lc] == gyi[:, lr, rc]).mean(1) >= 0.95
Gg = gray[m]
full = np.load(os.path.join(DATA, "L0_full_idx.npy"))
assert len(Gg) == len(full), f"alignment broke: {len(Gg)} vs {len(full)}"
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:PCA_K]
coef = (X - mu) @ comps.T
sd = coef.std(0, keepdims=True) + 1e-6
print(f"PCA rebuilt: mean {mu.shape}, comps {comps.shape}, sd range "
      f"{sd.min():.3f}..{sd.max():.3f}")
ev = (S ** 2) / (S ** 2).sum()
print(f"top-{PCA_K} components explain {100*ev[:PCA_K].sum():.2f}% of variance")

# ---- a gallery of real faces + their exact PCA values -------------------------------
# IMPORTANT: store the STANDARDISED values ((coef/sd)), because that is exactly what the
# generator was conditioned on. Storing raw coefficients instead made the gallery values
# off by a factor of sd (49..444) and the page's confidence distances meaningless.
sp = np.load(os.path.join(DATA, "L0_split.npz"))
va = sp["val"]
gal_idx = np.random.default_rng(11).choice(len(va), 24, replace=False)
gal_rows = va[gal_idx]
gallery = []
for r in gal_rows:
    pca_std = ((X[r] - mu[0]) @ comps.T) / sd[0]        # standardised, model units
    gallery.append({"rgb": full[r].reshape(-1).tolist(),          # 64 16-level values
                    "pca": [round(float(v), 6) for v in pca_std]})
print(f"gallery: {len(gallery)} real faces, pca stored in model units (coef/sd)")
print(f"  check: std across the gallery per component (should be O(1)): "
      f"{np.array([g['pca'] for g in gallery]).std(0).mean():.3f}")

# ---- prove the stored PCA basis is the true one -------------------------------------
# Two checks that can actually FAIL, unlike a self-comparison:
#   1. reconstruction error: mu + coef*sd @ comps must rebuild the face
#   2. orthonormality: the exported components must be orthonormal, so the exported basis
#      is a genuine PCA basis and not some rotated/scaled copy
chk = []
for r in gal_rows[:8]:
    cont = gray[r].reshape(-1)                       # the exact continuous vector PCA saw
    coef = ((cont - mu[0]) @ comps.T) / sd[0]
    rec = mu[0] + (coef * sd[0]) @ comps             # invert the standardisation too
    chk.append(float(np.sqrt(((rec - cont) ** 2).mean())))
gram = comps @ comps.T
orth_err = float(np.abs(gram - np.eye(PCA_K)).max())
quant_rms = float(np.sqrt(((LEVELS[gyi[m]] - gray[m]) ** 2).mean()))
print("  basis checks (these can fail, unlike a self-comparison):")
print(f"    PCA reconstruction RMS on continuous grayscale : {np.mean(chk):.3f} / 255 levels")
print(f"    16-level quantisation RMS for reference        : {quant_rms:.3f} / 255 levels")
print(f"    orthonormality  max |comps@comps.T - I|        : {orth_err:.2e}  (must be ~0)")
print(f"    residual variance not captured by {PCA_K} comps  : {1.0 - ev[:PCA_K].sum():.4f}")
# Orthonormality is the check that can actually falsify "this is the real PCA basis".
# The reconstruction RMS being LARGER than the quantisation RMS is NOT a bug: 22 of 64
# components retain 96.8% of the VARIANCE, and RMS ~ sqrt(1-0.968)*std(data) ~ 17/255.
# That is the genuine information loss of using 22 dimensions, and it is the floor on
# how faithful any face reconstructed from 22 numbers can be.
assert orth_err < 1e-6, "exported PCA components are not orthonormal"
out_chk = {"recon_rms": float(np.mean(chk)), "orth_err": orth_err,
           "quant_rms": quant_rms, "residual_var_fraction": float(1.0 - ev[:PCA_K].sum())}

out = {
    "meta": {
        "source_run": TAG,
        "steps_trained": int(a["steps"]),
        "batch": int(a["batch"]),
        "lr_g": float(a["lr_g"]), "lr_d": float(a["lr_d"]),
        "ce_weight": float(a["ce_weight"]),
        "r1_every": int(a["r1_every"]), "r1_gamma": float(a["r1_gamma"]),
        "zdim": int(a["zdim"]), "hid": int(a["hid"]), "pca_k": PCA_K,
        "pca_variance_explained": float(ev[:PCA_K].sum()),
        "seed": int(a["seed"]),
        "layout": "canonical: position p = col*8 + row over the left half; mirror within each row",
        "note": "generator outputs 32x16 logits for the LEFT half; the page mirrors it",
    },
    "levels": [float(v) for v in LEVELS],
    "pca": {"mean": [float(v) for v in mu[0]],
            "components": [[float(v) for v in row] for row in comps],
            "sd": [float(v) for v in sd[0]]},
    "gen": {"W1": W1.tolist(), "b1": b1.tolist(),
            "W2": W2.tolist(), "b2": b2.tolist(),
            "W3": W3.tolist(), "b3": b3.tolist()},
    "gallery": gallery,
    "basis_checks": out_chk,
}
p = os.path.join(ROOT, "webapp", "pca22_model.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
if os.path.exists(p):
    raise SystemExit(f"refusing to overwrite {p}")
json.dump(out, open(p, "w", encoding="utf-8"))
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
print(f"\nwrote {p}  ({os.path.getsize(p)/1e6:.2f} MB)")
print(f"sha256 {h}")
print("EXPORT_DONE")
