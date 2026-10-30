#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Exactly what happens from input to output in the model being labelled (final_lam5_20k),
and what each stage would cost in redstone.

Two halves:

PART A -- the architecture, stated precisely: every layer, its shape, how many
multiply-accumulates it performs, what the activation does, how decoding works, and where
each number a redstone builder would have to represent comes from.

PART B -- the honest cost. Redstone has no multiplier primitive; a multiply is a build.
So the question is not "203,264 parameters is a lot" but "how many multiply-accumulates
must actually be executed per image". This script counts them and compares against what
is buildable. It then trains CHEAPER substitutes (a direct linear map, and a linear map
with ternary weights where a multiply degenerates into add / subtract / skip) and
measures the quality cost, because that is the real path to something buildable.
"""
import json, os, sys
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
OUT = os.path.join(ROOT, "results", "redstone")
os.makedirs(OUT, exist_ok=True)

# =====================================================================================
# PART A -- the architecture of the model being labelled
# =====================================================================================
RUN = os.path.join(ROOT, "results", "coherence", "final_lam5_20k")
ck = torch.load(os.path.join(RUN, "final.pt"), weights_only=False)
a = ck["args"]
print("=" * 100)
print("PART A -- THE ARCHITECTURE OF final_lam5_20k (the model being labelled)")
print("=" * 100)
print(f"  trained {a['steps']} steps, TV coherence penalty lam={a['lam']}, "
      f"decode={a['decode']}")

lin = [(k, v.shape) for k, v in ck["G"].items()]
print()
print("  weights as stored in the checkpoint:")
for k, s in lin:
    print(f"    {k:12s} {str(tuple(s)):>16s}")

n_params = sum(int(np.prod(s)) for _, s in lin)
macs = 22 * 256 + 256 * 256 + 256 * 512
print()
print(f"  total parameters          : {n_params:,}")
print(f"  multiply-accumulates/image: {macs:,}")
print(f"  additions (bias)          : {256 + 256 + 512:,}")
print(f"  ReLU comparisons          : {256 + 256:,}")
print(f"  final argmax comparisons  : {32*15:,}   (32 positions x 15 comparisons each)")

print()
print("  DATA FLOW, one image:")
print("    1. INPUT  22 numbers. Each is (PCA coefficient / that component's std), i.e. a")
print("       number in roughly -3.5..+3.5 measured in STANDARD DEVIATIONS. There is no")
print("       latent noise: the 64-dim z of the older model is absent.")
print("    2. LAYER1  for each of 256 hidden units: h1[j] = relu( b1[j] + sum_{i<22} W1[j][i]*x[i] )")
print("               -> 22 x 256 = 5,632 multiplies")
print("    3. LAYER2  h2[j] = relu( b2[j] + sum_{i<256} W2[j][i]*h1[i] )")
print("               -> 256 x 256 = 65,536 multiplies")
print("    4. LAYER3  logits[j] = b3[j] + sum_{i<256} W3[j][i]*h2[i],  j < 512")
print("               -> 256 x 512 = 131,072 multiplies")
print("    5. DECODE  the 512 logits are 32 independent groups of 16. For each group take")
print("               the argmax index in 0..15 = a grey level. 32 x 15 comparisons.")
print("    6. LAYOUT  the 32 values are positions p = col*8 + row over the LEFT half;")
print("               the right half is the mirror, face[row][7-col] = face[row][col].")
print("    7. RENDER  grey level L -> pixel value 255*L/15, i.e. 0,17,34,...,255.")

# =====================================================================================
# PART B -- redstone cost
# =====================================================================================
print()
print("=" * 100)
print("PART B -- WHAT THAT WOULD TAKE IN REDSTONE")
print("=" * 100)
print("""  The binding constraint is NOT memory (weights can be baked into the build as
  constant wiring) but EXECUTED MULTIPLY-ACCUMULATES. Redstone has no multiply
  primitive: a multiply is a circuit, and each one consumes space, ticks, and lag.

  Reference points from the redstone community, for scale:
    * a compact 4-bit multiplier        ~ tens of logic gates
    * an 8-bit multiplier               ~ hundreds of gates
    * a working redstone CPU            ~ 10^4 .. 10^6 components
  These are order-of-magnitude figures; the exact numbers vary hugely by design.""")

per_mult_gates = {"4-bit": 60, "8-bit": 300}
print()
print(f"  current model: {macs:,} multiplies")
for name, g in per_mult_gates.items():
    print(f"    if each is a {name} multiplier (~{g} gates): "
          f"{macs*g/1e6:,.1f} million gates")
print()
print("  -> that is 2-3 orders of magnitude beyond the largest practical redstone builds.")
print("     The current model is NOT redstone-realizable, and no amount of clever wiring")
print("     changes that: the multiply count is structural.")

# ---- what WOULD be buildable, and what it costs in quality -------------------------
print()
print("=" * 100)
print("PART B2 -- CHEAPER ARCHITECTURES, AND WHAT THEY COST IN QUALITY")
print("=" * 100)
print("  Three progressively cheaper forms are trained and compared. 'ternary' weights")
print("  (-1, 0, +1) matter because a ternary multiply degenerates into add / subtract /")
print("  skip, which is the difference between needing multipliers and needing adders.")

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
U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
COND = torch.tensor(((X - mu) @ comps.T) / sd[0], dtype=torch.float32)
RTRUE = torch.from_numpy(full.astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]


def pen_tv_cont(v):
    """TV on the continuous output field, mirroring the penalty used in training."""
    p = v.view(-1, 4, 8)
    return (p[:, 1:] - p[:, :-1]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()


def train_linear(hid_layers, ternary, steps=6000, lam=5.0, seed=0):
    torch.manual_seed(seed)
    seq = []
    d_in = 22
    for h in hid_layers:
        seq += [nn.Linear(d_in, h), nn.ReLU()]
        d_in = h
    seq += [nn.Linear(d_in, 32)]
    net = nn.Sequential(*seq)
    if ternary:
        # ternarise the weights in the forward pass (straight-through on the sign scale)
        class Tern(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                W = self.m.weight
                s = W.abs().mean().clamp_min(1e-6) * 1.5
                Wq = torch.clamp(torch.round(W / s), -1, 1) * s
                return F.linear(x, Wq, self.m.bias)
        for i in range(0, len(net), 2):
            net[i] = Tern(net[i])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        bi = tr[rng.integers(0, len(tr), 256)]
        out = net(COND[bi])
        tgt = RTRUE[bi].float()
        loss = F.mse_loss(out.clamp(0, 15), tgt) + lam * pen_tv_cont(out.clamp(0, 15))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        o = net(COND[va[:1000]]).clamp(0, 15)
        dec = o.round().long()
        Rt = RTRUE[va[:1000]]
        acc = float((dec == Rt).float().mean())
        mae = float((dec.float() - Rt.float()).abs().mean())
    imgs = layout.image_from_left_half(dec.numpy())
    agree = []
    for g in imgs:
        pairs = same = 0
        for r in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r, c] == g[rr, cc]:
                            same += 1
        agree.append(same / pairs)
    np_ = sum(p.numel() for p in net.parameters())
    mac = 0
    d = 22
    for h in hid_layers:
        mac += d * h
        d = h
    mac += d * 32
    return {"net": str(hid_layers), "ternary": ternary, "params": int(np_),
            "macs": int(mac), "acc": acc, "mae": mae, "agree": float(np.mean(agree))}


print()
print(f"  {'architecture':22s} {'ternary':>8s} {'params':>9s} {'multiplies':>11s} "
      f"{'acc':>7s} {'MAE':>6s} {'agree':>7s}")
print("  " + "-" * 80)
rows = []
for hid in ([], [16], [32], [64]):
    for tern in (False, True):
        r = train_linear(hid, tern)
        rows.append(r)
        arch = "22->32" if not hid else "22->" + "->".join(map(str, hid)) + "->32"
        print(f"  {arch:22s} {str(tern):>8s} {r['params']:9,d} {r['macs']:11,d} "
              f"{r['acc']:7.4f} {r['mae']:6.3f} {r['agree']*100:6.1f}%")

print()
print(f"  reference: the full model (final_lam5_20k) has 203,264 params, "
      f"202,240 multiplies, acc 0.7005, agree 60.7%")
print(f"  reference: real faces, agree 60.5%")

json.dump({"full_model": {"params": n_params, "macs": macs,
                          "acc": 0.7005, "agree": 0.607},
           "cheap_models": [{k: v for k, v in r.items() if k != "net"} | {"hid": r["net"]}
                            for r in rows]},
          open(os.path.join(OUT, "redstone_cost.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'redstone_cost.json')}")
print("REDSTONE_ANALYSIS_DONE")
