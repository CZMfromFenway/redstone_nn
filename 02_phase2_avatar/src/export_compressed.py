#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Export a dim_sweep checkpoint to the webapp JSON format, for the blind comparison.

Same schema as the existing exports so label.html needs no special case:
    {"gen": {"layers": [{"W":..., "b":...}, ...]}, "meta": {...}}
    meta.output_mode = "continuous32" and meta.post = "stretch" -- the two fields that, when
    missing, produced all-black faces earlier in this project.

Nothing is distilled or copied: each checkpoint here was trained from scratch on the data with
the identical recipe, only the width and layer count differ.
"""
import argparse
import json
import os
import sys
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "dimsweep")
WEB = os.path.join(ROOT, "webapp")

A = argparse.ArgumentParser()
A.add_argument("--tag", required=True)
A.add_argument("--name", required=True)
A.add_argument("--note", default="")
A = A.parse_args()

p = os.path.join(RES, A.tag, "final.pt")
if not os.path.exists(p):
    raise SystemExit(f"missing {p}")
ck = torch.load(p, weights_only=False)
st = ck.get("G") or ck.get("S")
st = {(k.replace("net.", "", 1) if k.startswith("net.") else k): v for k, v in st.items()}
idx = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
layers = []
mults = 0
for j in idx:
    W = st[f"{j}.weight"]
    b = st[f"{j}.bias"]
    layers.append({"W": W.tolist(), "b": b.tolist()})
    mults += int(W.shape[0]) * int(W.shape[1])

args = ck.get("args", {})
meta = {
    "in_dim": int(st[f"{idx[0]}.weight"].shape[1]),
    "out_dim": int(st[f"{idx[-1]}.weight"].shape[0]),
    "k": 4,
    "multiplies": int(mults),
    "steps": args.get("steps"),
    "hid": args.get("hid"),
    "layers": args.get("layers", 2),
    "lam": args.get("lam"),
    "adv": args.get("adv"),
    "acc": args.get("acc"),
    "agree": args.get("agree"),
    "mode": "gray32",
    # both of these are load-bearing: without output_mode the decoder falls back to logits_gray
    # and reads indices up to 511 from a 32-wide array, which renders as an all-black face.
    "output_mode": "continuous32",
    "post": "stretch",
    "note": A.note,
}
out = {"gen": {"layers": layers}, "meta": meta}
fp = os.path.join(WEB, A.name)
json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False)

# immediate sanity check: the exported model must reproduce the checkpoint and must not be flat
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
net = nn.Sequential(*seq).eval()
import numpy as np
Z = np.clip(np.random.default_rng(0).standard_normal((600, 4)), -3.5, 3.5).astype(np.float32)
with torch.no_grad():
    v = np.clip(net(torch.tensor(Z)).numpy(), 0, 15)
lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
frozen = int((v.std(0) < 0.35).sum())
# A single constant position is NOT degeneration. After the min-max stretch, the pixel that is
# always the darkest point of the face is forced to exactly 0, so its sd is 0 even though the
# model is healthy -- measured at 78-87% of faces in three models that tripped the old check.
# Genuine collapse looked like 21 of 32 positions constant at once.
DEGENERATE = frozen > 4
print(f"wrote {fp}  ({os.path.getsize(fp)/1e3:.0f} KB)")
print(f"  {meta['in_dim']} -> " + " -> ".join(str(len(L['W'])) for L in layers))
print(f"  multiplies {mults:,}   constant positions {frozen}/32"
      + ("   <-- DEGENERATE" if DEGENERATE else
         ("   (extreme position, normal)" if frozen else "")))
if DEGENERATE:
    raise SystemExit("EXPORT REJECTED: degenerate output (many constant positions)")
print("EXPORT_OK")
