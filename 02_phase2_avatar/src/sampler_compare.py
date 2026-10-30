#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Two questions the first contrast measurement left open.

1. WHY "too bright / too dark" as well as "grey"?
   The mean brightness matched real faces, so the tails must be the problem. Sampling the
   condition as INDEPENDENT standard normals produces inputs that no real face would
   produce: PC1 (brightness) is drawn up to +-3.5 standard deviations, and real faces
   essentially never go there. So the sampler manufactures extreme brightness that the
   training set does not contain.

2. IS SAMPLING FROM THE REAL CONDITION DISTRIBUTION BETTER?
   The obvious alternative to N(0,1)^k is to draw the k coefficients from REAL faces
   (bootstrap the empirical distribution). That keeps the input on the manifold the model
   was trained on. This measures both samplers on the same model and compares the resulting
   contrast / brightness distributions against real faces.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "dimsweep")
NLEV = 16
LEVELS = np.linspace(0, 255, 16)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_json_model(path):
    j = json.load(open(path, encoding="utf-8"))
    layers = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(layers):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(layers) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.eval()
    return m, int(len(layers[0]["W"][0]))


def stats(dec):
    per = dec.reshape(-1, 32).astype(float)
    con = per.max(1) - per.min(1)
    bri = per.mean(1)
    return {"contrast": float(con.mean()), "bright": float(bri.mean()),
            "dark": float(np.mean(bri < 3)), "bright_tail": float(np.mean(bri > 12)),
            "flat": float(np.mean(con <= 2)), "washed": float(np.mean(con <= 5)),
            "n": len(con)}


# ---------------------------------------------------------------- real reference + conditions
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
real = full[:, layout.POS_ROW, layout.MIRROR_COL]
X = full.reshape(len(full), -1).astype(np.float64)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
_, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
ZZ = (((X - mu) @ comps.T) / sd[0]).astype(np.float32)     # (N,22) the real conditions

rs = stats(real)
print("=" * 104)
print("REAL FACES AS THE REFERENCE")
print("=" * 104)
print(f"  contrast {rs['contrast']:5.2f}  brightness {rs['bright']:5.2f}  "
      f"very dark (<3) {100*rs['dark']:5.2f}%  very bright (>12) {100*rs['bright_tail']:5.2f}%  "
      f"washed {100*rs['washed']:5.2f}%")

print()
print(f"  how extreme is PC1 in real faces?  "
      f"|PC1|>2: {100*np.mean(np.abs(ZZ[:,0])>2):.3f}%   "
      f"|PC1|>3: {100*np.mean(np.abs(ZZ[:,0])>3):.3f}%   max {np.abs(ZZ[:,0]).max():.2f}")
print(f"  an independent N(0,1) draw would give |PC1|>2 {100*0.0455:.2f}% "
      f"and |PC1|>3 {100*0.0027:.2f}%")

# ---------------------------------------------------------------- compare the two samplers
rng = np.random.default_rng(7)
N = 6000
MODELS = [("dim22", "pca22_dim22_model.json", 22),
          ("dim8", "pca22_dim8_model.json", 8),
          ("dim4", "pca22_dim4_model.json", 4)]
print()
print("=" * 104)
print("TWO WAYS TO DRAW THE CONDITION")
print("=" * 104)
print("  A: independent N(0,1) per slot, clipped to +-3.5   (what the page does now)")
print("  B: bootstrap the k coefficients from REAL faces   (stays on the training manifold)")
print()
print(f"  {'model':7s} {'sampler':22s} {'contrast':>9s} {'bright':>7s} {'dark%':>7s} "
      f"{'bright%':>8s} {'washed%':>8s}")
print("  " + "-" * 76)
results = {}
for tag, fname, k in MODELS:
    net, kk = load_json_model(os.path.join(WEB, fname))
    for name, Z in (
        ("A: N(0,1) clipped", np.clip(rng.standard_normal((N, k)), -3.5, 3.5).astype(np.float32)),
        ("B: bootstrap real", ZZ[rng.integers(0, len(ZZ), N)][:, :k].astype(np.float32)),
    ):
        with torch.no_grad():
            dec = np.clip(np.rint(net(torch.tensor(Z)).numpy()), 0, 15)
        s = stats(dec)
        results[f"{tag}|{name}"] = s
        print(f"  {tag:7s} {name:22s} {s['contrast']:9.2f} {s['bright']:7.2f} "
              f"{100*s['dark']:6.2f}% {100*s['bright_tail']:7.2f}% {100*s['washed']:7.2f}%")

print()
print("  real faces for comparison      "
      f"{rs['contrast']:9.2f} {rs['bright']:7.2f} {100*rs['dark']:6.2f}% "
      f"{100*rs['bright_tail']:7.2f}% {100*rs['washed']:7.2f}%")

json.dump({"real": rs, "results": results},
          open(os.path.join(OUT, "sampler_compare.json"), "w"), indent=2)
print()
print(f"wrote {os.path.join(OUT, 'sampler_compare.json')}")
print("SAMPLER_COMPARE_DONE")
