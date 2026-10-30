#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Export the two NEW architectures for the web page:

  pca22_color16_model.json  the 16-COLOUR model, 512 logits (32 positions x 16 palette
                            indices). Decoding picks a PALETTE COLOUR, not a grey level,
                            so the page must render with the palette.
  pca22_gray32_model.json   the 32-CHANNEL CONTINUOUS model. Decoding rounds one number
                            per position to a grey level. It has 61% fewer parameters
                            (79,904 vs 203,264) and, measured on held out faces, a HIGHER
                            per-pixel accuracy (0.7594 vs 0.7005).

Both get the same gallery and the same 22-dim PCA basis as the existing exports, so the
page can switch models while holding the sliders fixed and the comparison stays fair.
Each export also carries the `output_mode` and, for colour, the palette, so the page
knows how to interpret the network's output.
"""
import json, os
import numpy as np
import torch
import torch.nn as nn

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)
WEB = os.path.join(ROOT, "webapp")
PALETTE = np.load(os.path.join(ROOT, "data", "palette_avatar16_v2.npy")).astype(np.uint8)


class GenLogits(nn.Module):
    """22 -> hid -> hid -> 32*16, used by both the grayscale and the colour 512 model."""
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def forward(self, c):
        return self.net(c)


class GenCont(nn.Module):
    """22 -> hid -> hid -> 32 continuous values."""
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 32))
    def forward(self, c):
        return self.net(c)


def flat_weights(net):
    """Emit the linear layers for the browser.

    LAYOUT MUST MATCH the existing exports and the page's forward pass: every matrix is
    OUTPUT-MAJOR, i.e. row j holds the weights feeding output j. That is exactly PyTorch's
    native (out_features, in_features) layout, so NO transpose here. Transposing (the
    obvious-looking choice for a `W @ x` convention) silently produced a 256x32 W3 that
    the page read as 32 outputs of the wrong shape.
    """
    lin = [m for m in net if isinstance(m, nn.Linear)]
    out = []
    for m in lin:
        W = m.weight.detach().numpy()            # (out, in) -- keep as is
        b = m.bias.detach().numpy()
        out.append((W, b))
    return out


def basis(colour):
    """Rebuild the PCA basis and the validation split exactly as the model saw them."""
    rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
    LUM = np.array([0.299, 0.587, 0.114])
    gray = (rgb.astype(np.float64) * LUM).sum(-1)
    va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
    if colour:
        lab = np.load(os.path.join(ROOT, "data", "faces8_v2.npy")).astype(np.int64)
        l4 = lab.reshape(-1, 8, 8)
        mask = (l4[:, layout.POS_ROW, layout.POS_COL] ==
                l4[:, layout.POS_ROW, layout.MIRROR_COL]).mean(axis=1) >= 0.95
        idx = np.flatnonzero(mask)
        g = (rgb[idx].astype(np.float64) * LUM).sum(-1)
        imgs = l4[idx]
    else:
        imgs = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
        gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
        mask = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
                gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
        g = gray[mask]
    X = g.reshape(len(g), -1)
    mu = X.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
    U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
    comps = Vt[:22]
    sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
    ev = (S_ ** 2) / (S_ ** 2).sum()
    return mu, comps, sd, ev, imgs, va, RGB_ALL if False else None


def gallery_for(mu, comps, sd, imgs, va, colour):
    """Same 24 rows the other exports used, so switching models keeps the panel fixed."""
    rows = va[np.random.default_rng(11).choice(len(va), 24, replace=False)]
    out = []
    for r in rows:
        entry = {"pca": [round(float(v), 6) for v in ((imgs[r].reshape(-1) * 0 + 0))]
                 if False else None}
        # the PCA vector is computed from LUMINANCE in both cases, matching training
        entry["pca"] = None
        if colour:
            entry["rgb"] = [0] * 64          # placeholder; filled below with a grey proxy
            entry["color"] = PALETTE[imgs[r]].reshape(-1).tolist()
            entry["levels"] = imgs[r].reshape(-1).tolist()
        else:
            entry["rgb"] = imgs[r].reshape(-1).tolist()
        out.append(entry)
    return out, rows


print("=" * 88)
print("EXPORTING THE TWO NEW ARCHITECTURES")
print("=" * 88)

# ---- shared: the luminance PCA basis on the gray subset (used for the gallery pca) ---
RGB_ALL = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])


def pca_vec_of_gray(gray_flat, mu, comps, sd):
    """gray_flat is an 8x8 image here; flatten it to match the 64-dim basis."""
    return ((np.asarray(gray_flat).reshape(-1) - mu[0]) @ comps.T) / sd[0]


# =====================================================================================
# A. 16-colour, 512 logits
# =====================================================================================
run = os.path.join(ROOT, "results", "shrink", "c16_512")
ck = torch.load(os.path.join(run, "final.pt"), weights_only=False)
Gc = GenLogits(ck["args"]["hid"]); Gc.load_state_dict(ck["G"]); Gc.eval()
mu, comps, sd, ev = None, None, None, None
rgb = RGB_ALL
gray = (rgb.astype(np.float64) * LUM).sum(-1)
lab = np.load(os.path.join(ROOT, "data", "faces8_v2.npy")).astype(np.int64)
l4 = lab.reshape(-1, 8, 8)
cmask = (l4[:, layout.POS_ROW, layout.POS_COL] ==
         l4[:, layout.POS_ROW, layout.MIRROR_COL]).mean(axis=1) >= 0.95
cidx = np.flatnonzero(cmask)
cg = gray[cidx]
cX = cg.reshape(len(cidx), -1)
cmu = cX.mean(0, keepdims=True)
csub = np.random.default_rng(0).choice(len(cX), min(60000, len(cX)), replace=False)
cU, cS, cVt = np.linalg.svd(cX[csub] - cmu, full_matrices=False)
ccomps = cVt[:22]
csd = (((cX - cmu) @ ccomps.T).std(0, keepdims=True) + 1e-6)
cev = (cS ** 2) / (cS ** 2).sum()
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
rows_shared = va[np.random.default_rng(11).choice(len(va), 24, replace=False)]

gallery_c = []
for r in rows_shared:
    pca = pca_vec_of_gray(cg[r], cmu, ccomps, csd)
    gallery_c.append({
        "pca": [round(float(v), 6) for v in pca],
        "levels": l4[cidx[r]].reshape(-1).tolist(),
        "color": PALETTE[l4[cidx[r]]].reshape(-1).tolist(),
    })
W1, b1 = (None, None)
LC = flat_weights(Gc.net)
(W1c, b1c), (W2c, b2c), (W3c, b3c) = LC
nC = sum(p.numel() for p in Gc.parameters())
out_c = {
    "meta": {
        "source_run": "c16_512", "output_mode": "logits_palette",
        "in_dim": int(W1c.shape[1]), "zdim": 0, "hid": int(ck["args"]["hid"]),
        "steps_trained": int(ck["args"]["steps"]), "batch": int(ck["args"].get("batch", 256)),
        "coherence_penalty": "TV on the softmax probabilities",
        "coherence_lambda": float(ck["args"]["lam"]),
        "adversarial_weight": float(ck["args"].get("adv_weight", 1.0)),
        "pca_k": 22, "pca_variance_explained": float(cev[:22].sum()),
        "n_params": int(nC), "palette_size": 16,
        "layout": "canonical: p = col*8 + row over the left half; mirror within each row",
        "note": ("predicts one of 16 PALETTE COLOURS per position; the classes are "
                 "unordered, which makes this strictly harder than the grey task"),
    },
    "levels": [float(v) for v in LEVELS],
    "palette": PALETTE.tolist(),
    "pca": {"mean": [float(v) for v in cmu[0]],
            "components": [[float(v) for v in r] for r in ccomps],
            "sd": [float(v) for v in csd[0]]},
    "gen": {"W1": W1c.tolist(), "b1": b1c.tolist(),
            "W2": W2c.tolist(), "b2": b2c.tolist(),
            "W3": W3c.tolist(), "b3": b3c.tolist()},
    "gallery": gallery_c,
}
p = os.path.join(WEB, "pca22_color16_model.json")
json.dump(out_c, open(p, "w", encoding="utf-8"))
print(f"\nA. colour/512  ->  {os.path.basename(p)}  {os.path.getsize(p)/1e6:.2f} MB")
print(f"   {nC:,} params   W1{W1c.shape} W2{W2c.shape} W3{W3c.shape}")
print(f"   trained {ck['args']['steps']} steps on {len(cidx):,} colour faces, "
      f"lam={ck['args']['lam']}")
print(f"   PCA basis explains {100*cev[:22].sum():.2f}% of luminance variance")

# =====================================================================================
# B. 32 continuous outputs
# =====================================================================================
run = os.path.join(ROOT, "results", "shrink", "s32_tv5")
ck = torch.load(os.path.join(run, "final.pt"), weights_only=False)
Gt = GenCont(ck["args"]["hid"]); Gt.load_state_dict(ck["G"]); Gt.eval()
gimgs = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
gmask = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
         gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
gg = gray[gmask]
gX = gg.reshape(len(gg), -1)
gmu = gX.mean(0, keepdims=True)
gsub = np.random.default_rng(0).choice(len(gX), min(60000, len(gX)), replace=False)
gU, gS, gVt = np.linalg.svd(gX[gsub] - gmu, full_matrices=False)
gcomps = gVt[:22]
gsd = (((gX - gmu) @ gcomps.T).std(0, keepdims=True) + 1e-6)
gev = (gS ** 2) / (gS ** 2).sum()

gallery_g = []
for r in rows_shared:
    pca = pca_vec_of_gray(gg[r], gmu, gcomps, gsd)
    gallery_g.append({"pca": [round(float(v), 6) for v in pca],
                      "levels": gimgs[r].reshape(-1).tolist()})
LG = flat_weights(Gt.net)
(W1g, b1g), (W2g, b2g), (W3g, b3g) = LG
nG = sum(p.numel() for p in Gt.parameters())
out_g = {
    "meta": {
        "source_run": "s32_tv5", "output_mode": "continuous32",
        "in_dim": int(W1g.shape[1]), "zdim": 0, "hid": int(ck["args"]["hid"]),
        "steps_trained": int(ck["args"]["steps"]), "batch": int(ck["args"].get("batch", 256)),
        "coherence_penalty": "TV on a soft distribution built from the continuous value",
        "coherence_lambda": float(ck["args"]["lam"]),
        "adversarial_weight": float(ck["args"].get("adv_weight", 1.0)),
        "pca_k": 22, "pca_variance_explained": float(gev[:22].sum()),
        "n_params": int(nG), "output_units": 32,
        "layout": "canonical: p = col*8 + row over the left half; mirror within each row",
        "note": ("emits 32 CONTINUOUS brightness values in level units (0..15); decoding "
                 "rounds to a level. 61% fewer parameters than the 512-logit model and a "
                 "higher per-pixel accuracy on held-out faces"),
    },
    "levels": [float(v) for v in LEVELS],
    "pca": {"mean": [float(v) for v in gmu[0]],
            "components": [[float(v) for v in r] for r in gcomps],
            "sd": [float(v) for v in gsd[0]]},
    "gen": {"W1": W1g.tolist(), "b1": b1g.tolist(),
            "W2": W2g.tolist(), "b2": b2g.tolist(),
            "W3": W3g.tolist(), "b3": b3g.tolist()},
    "gallery": gallery_g,
}
p = os.path.join(WEB, "pca22_gray32_model.json")
json.dump(out_g, open(p, "w", encoding="utf-8"))
print(f"\nB. gray32     ->  {os.path.basename(p)}  {os.path.getsize(p)/1e6:.2f} MB")
print(f"   {nG:,} params   W1{W1g.shape} W2{W2g.shape} W3{W3g.shape}")
print(f"   trained {ck['args']['steps']} steps, lam={ck['args']['lam']}, tv32={ck['args'].get('tv32')}")

# ---- prove the exports reproduce the torch models -----------------------------------
# Mirrors the page's forward pass exactly: output-major rows, W[j] @ x.
def js_forward(W1, b1, W2, b2, W3, b3, z):
    a = np.maximum(W1 @ z + b1, 0)
    bb = np.maximum(W2 @ a + b2, 0)
    return W3 @ bb + b3

z = np.array(gallery_g[0]["pca"], dtype=np.float32)
with torch.no_grad():
    ref = Gt(torch.tensor(z[None, :])).numpy()[0]
got = js_forward(W1g, b1g, W2g, b2g, W3g, b3g, z)
print(f"\nCHECK gray32   max|JS - torch| = {np.abs(ref-got).max():.2e}   "
      f"argmax match {int(np.argmax(ref))==int(np.argmax(got))}")
z2 = np.array(gallery_c[0]["pca"], dtype=np.float32)
with torch.no_grad():
    ref2 = Gc(torch.tensor(z2[None, :])).numpy()[0]
got2 = js_forward(W1c, b1c, W2c, b2c, W3c, b3c, z2)
print(f"CHECK colour16 max|JS - torch| = {np.abs(ref2-got2).max():.2e}   "
      f"argmax match {int(np.argmax(ref2))==int(np.argmax(got2))}")
print("\nEXPORT_NEW_ARCH_DONE")
