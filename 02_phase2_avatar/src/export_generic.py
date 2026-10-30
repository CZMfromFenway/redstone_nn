#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Export ANY of the 32-output students (or the 512 teacher) to the web page, in one format.

The point is that phases 2-5 of the 512->32 plan produce several candidate checkpoints
(hard-target, distilled, distilled + various TV weights, and later shrunk hidden sizes).
Rather than writing a new exporter per checkpoint, this one reads whichever .pt it is
pointed at, works out the architecture from the weights, and writes a page-ready JSON.

It also RECOMPUTES the model's quality on one fixed 1000-face hold-out sample, so every
exported model carries trustworthy numbers instead of numbers copied from an older run
with a different architecture.

Layout reminder (it bit me earlier): the page expects OUTPUT-MAJOR weights, i.e. row j
holds the weights feeding output j, which is PyTorch's native (out, in) layout. No
transpose.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)
WEB = os.path.join(ROOT, "webapp")

AP = argparse.ArgumentParser()
AP.add_argument("--pt", required=True, help="checkpoint path")
AP.add_argument("--name", required=True, help="output json name in webapp/")
AP.add_argument("--label", default="", help="human label written into the metadata")
AP.add_argument("--teacher", default="results/coherence/final_lam5_20k/final.pt",
                help="reference checkpoint for the shared PCA basis")
A = AP.parse_args()


class Gen512(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def forward(self, c):
        return self.net(c)


class Gen32(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 32))
    def forward(self, c):
        return self.net(c)


def build(widths, out_dim):
    """Generic MLP from a list of hidden widths."""
    seq = []
    d = 22
    for h in widths:
        seq += [nn.Linear(d, h), nn.ReLU()]
        d = h
    seq += [nn.Linear(d, out_dim)]
    return nn.Sequential(*seq)


# ---- PCA basis and split: identical to every other export --------------------------
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U_, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
ev = (S_ ** 2) / (S_ ** 2).sum()
COND = torch.tensor(((X - mu) @ comps.T) / sd[0], dtype=torch.float32)
RTRUE = torch.from_numpy(full.astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]

# ---- load the checkpoint and infer its shape --------------------------------------
ck = torch.load(os.path.join(ROOT, A.pt), weights_only=False)
state = ck.get("G") or ck.get("S")
if state is None:
    raise SystemExit("checkpoint has neither 'G' nor 'S'")
# Normalise the key prefix. Some training scripts wrap their layers in `self.net` (keys
# "net.0.weight"), others store a bare Sequential (keys "0.weight"). Strip the prefix and
# index by the part BEFORE the first dot, so both forms parse.
norm = {}
for k, v in state.items():
    norm[k.replace("net.", "", 1) if k.startswith("net.") else k] = v
state = norm
lin_idx = sorted({int(k.split(".")[0]) for k in state if k.endswith(".weight")})
widths = []
for j in lin_idx:
    widths.append(int(state[f"{j}.weight"].shape[0]))
out_dim = widths.pop()
print(f"detected architecture: 22 -> {' -> '.join(map(str, widths))} -> {out_dim}")
mode = "logits" if out_dim == 32 * NLEV else "continuous32"
print(f"output mode: {mode}  ({out_dim} outputs for 32 positions)")

net = build(widths, out_dim)
# the trained modules wrap their Sequential in `self.net`, so the checkpoint keys carry a
# "net." prefix while the bare Sequential built here does not
net.load_state_dict({k.replace("net.", "", 1) if k.startswith("net.") else k: v
                     for k, v in state.items()})
net.eval()

# ---- quality, recomputed on one fixed sample --------------------------------------
NS = 1000
bv = va[:NS]
with torch.no_grad():
    raw = net(COND[bv])
if mode == "logits":
    dec = raw.view(-1, 32, NLEV).argmax(-1)
else:
    dec = raw.clamp(0, 15).round().long()
Rt = RTRUE[bv]
acc = float((dec == Rt).float().mean())
mae = float((dec.float() - Rt.float()).abs().mean())
imgs = layout.image_from_left_half(dec.numpy())


def agree_of(a):
    out = []
    for g in a:
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
        out.append(same / pairs)
    return float(np.mean(out))


def detail_of(a):
    d = []
    for g in a:
        d.append((np.abs(np.diff(g.astype(float), axis=1)).mean()
                  + np.abs(np.diff(g.astype(float), axis=0)).mean()) / 2)
    return float(np.mean(d))


agree = agree_of(imgs)
det = detail_of(imgs)
real_agree = agree_of(full[bv].reshape(-1, 8, 8))
real_det = detail_of(full[bv].reshape(-1, 8, 8))
print(f"\nquality on {NS} held-out faces:")
print(f"  accuracy {acc:.4f}   MAE {mae:.3f}   agreement {agree*100:.1f}%   "
      f"detail {det:.2f}")
print(f"  real reference:      agreement {real_agree*100:.1f}%   detail {real_det:.2f}")

n_params = sum(p.numel() for p in net.parameters())
macs = 22 * (widths[0] if widths else out_dim)
d = widths[0] if widths else 22
for i, h in enumerate(widths):
    nxt = widths[i + 1] if i + 1 < len(widths) else out_dim
    macs += h * nxt
print(f"  parameters {n_params:,}   multiplies {macs:,}")

# ---- the shared 24-face gallery ---------------------------------------------------
rows24 = va[np.random.default_rng(11).choice(len(va), 24, replace=False)]
gallery = [{"pca": [round(float(v), 6) for v in ((X[r] - mu[0]) @ comps.T) / sd[0]],
            "levels": full[r].reshape(-1).tolist()} for r in rows24]

out = {
    "meta": {
        "source_run": os.path.basename(os.path.dirname(A.pt)),
        "label": A.label,
        "output_mode": mode,
        "in_dim": 22, "zdim": 0, "hid": widths[0] if widths else 0,
        "widths": widths, "out_dim": out_dim,
        "n_layers": len(widths) + 1,
        "n_params": int(n_params), "multiplies": int(macs),
        "steps_trained": int(ck.get("args", {}).get("steps", 0)),
        "training_mode": ck.get("args", {}).get("mode", "?"),
        "coherence_lambda": float(ck.get("args", {}).get("lam", 0.0)),
        "pca_k": 22, "pca_variance_explained": float(ev[:22].sum()),
        "layout": "canonical: p = col*8 + row over the left half; mirror within each row",
        "measured": {"accuracy": acc, "mae": mae, "agreement": agree, "detail": det,
                     "real_agreement": real_agree, "real_detail": real_det, "n": NS},
        "note": ("numbers in `measured` were recomputed for THIS checkpoint on the same "
                 "1000-face hold-out sample, so they are comparable across exports"),
    },
    "levels": [float(v) for v in LEVELS],
    "pca": {"mean": [float(v) for v in mu[0]],
            "components": [[float(v) for v in r] for r in comps],
            "sd": [float(v) for v in sd[0]]},
    # ARBITRARY DEPTH. The first 512/32 exports hardcoded three matrices named W1/W2/W3,
    # which silently broke any net with a different number of layers: for a two-linear-layer
    # net the output layer was emitted twice and applied twice, giving an 88-unit error.
    # The page now reads a list and applies a ReLU between consecutive layers.
    "gen": {"layers": [
        {"W": m.weight.detach().numpy().tolist(),
         "b": m.bias.detach().numpy().tolist()}
        for m in net if isinstance(m, nn.Linear)]},
    "gallery": gallery,
}
p = os.path.join(WEB, A.name)
json.dump(out, open(p, "w", encoding="utf-8"))
print(f"\nwrote {p}  ({os.path.getsize(p)/1e6:.2f} MB)")

# ---- verify the export reproduces torch -------------------------------------------
z = np.array(gallery[0]["pca"], dtype=np.float32)
with torch.no_grad():
    ref = net(torch.tensor(z[None, :])).numpy()[0]
# replay the page's arithmetic from the exported list: ReLU between layers, linear last
h = z
layers = out["gen"]["layers"]
for i, L in enumerate(layers):
    h = np.array(L["W"]) @ h + np.array(L["b"])
    if i < len(layers) - 1:
        h = np.maximum(h, 0)
print(f"CHECK  max|JS - torch| = {np.abs(ref-h).max():.2e}   "
      f"layers={len(layers)}  argmax_match="
      f"{int(np.argmax(ref))==int(np.argmax(h))}")
print("EXPORT_GENERIC_DONE")
