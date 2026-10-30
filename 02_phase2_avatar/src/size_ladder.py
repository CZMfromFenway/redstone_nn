#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
How small can the generator get before quality collapses? A fair ladder.

The first attempt at this question used plain MSE on level values and a ternary
quantisation written as `torch.round`, which has ZERO gradient almost everywhere -- so
the ternary models could not train at all and scored below chance. Both were my bugs, not
properties of the task. This version fixes them:

  * the loss is CROSS-ENTROPY over the 16 levels per position, matching how the real model
    was trained (a softmax objective is far easier to optimise than MSE on an ordinal
    value, and the real model's 0.70 accuracy came from CE)
  * ternary weights use a proper STRAIGHT-THROUGH estimator, so they can train
  * every size gets the same step budget and the same TV coherence penalty

The ladder runs from the full architecture down to a single linear map, so the cost of
each reduction is readable directly.
"""
import json, os, sys, time
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
torch.set_num_threads(10)

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


class TernLinear(nn.Module):
    """Linear layer whose weights are constrained to {-1, 0, +1} times a per-layer scale.

    A ternary multiply degenerates into add / subtract / skip, which is the whole point:
    it replaces multipliers with adders. The straight-through estimator matters -- without
    it `round` kills the gradient and the layer cannot learn at all.
    """
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=bias)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        W = self.lin.weight
        s = self.scale.abs().clamp_min(1e-4)
        Wq = torch.clamp(torch.round(W / s), -1, 1) * s
        Wq = W + (Wq - W).detach()          # straight-through
        return F.linear(x, Wq, self.lin.bias)


class CheapGen(nn.Module):
    """22 -> (hidden...) -> 512 logits, i.e. 32 positions x 16 levels."""
    def __init__(self, hids, ternary=False):
        super().__init__()
        def mk(a, b):
            return TernLinear(a, b) if ternary else nn.Linear(a, b)
        seq = []
        d = 22
        for h in hids:
            seq += [mk(d, h), nn.ReLU()]
            d = h
        seq += [mk(d, 32 * NLEV)]
        self.net = nn.Sequential(*seq)
        self.macs = 22 * (hids[0] if hids else 32 * NLEV)
        d = hids[0] if hids else 22
        for i, h in enumerate(hids):
            nxt = hids[i + 1] if i + 1 < len(hids) else 32 * NLEV
            self.macs += h * nxt
    def forward(self, c):
        return self.net(c).view(-1, 32, NLEV)


def tv(lg):
    p = F.softmax(lg, -1).reshape(-1, 4, 8, NLEV)
    return (p[:, 1:] - p[:, :-1]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()


def speckle(imgs):
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
    return float(np.mean(agree))


def run_one(hids, ternary, steps=8000, lam=5.0, seed=0):
    torch.manual_seed(seed)
    net = CheapGen(hids, ternary)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        bi = tr[rng.integers(0, len(tr), 256)]
        lg = net(COND[bi])
        loss = F.cross_entropy(lg.reshape(-1, NLEV), RTRUE[bi].reshape(-1)) + lam * tv(lg)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        lg = net(COND[va[:1000]])
        dec = lg.argmax(-1).numpy()
        Rt = RTRUE[va[:1000]].numpy()
        acc = float((dec == Rt).mean())
        mae = float(np.abs(dec - Rt).mean())
    agree = speckle(layout.image_from_left_half(dec))
    params = sum(p.numel() for p in net.parameters())
    return {"hids": hids, "ternary": bool(ternary), "params": int(params),
            "macs": int(net.macs), "acc": acc, "mae": mae, "agree": agree,
            "seconds": round(time.time() - t0, 1)}


LADDER = [
    ([256, 256], False, "full (256,256)"),
    ([128, 128], False, "half  (128,128)"),
    ([64, 64], False, "small (64,64)"),
    ([32], False, "tiny  (32)"),
    ([16], False, "micro (16)"),
    ([], False, "linear only"),
    ([32], True, "tiny ternary"),
    ([64, 64], True, "small ternary"),
]

print("=" * 104)
print("HONEST SIZE LADDER (cross-entropy loss + proper straight-through for ternary)")
print("=" * 104)
print(f"  {'architecture':22s} {'ternary':>8s} {'params':>9s} {'multiplies':>11s} "
      f"{'acc':>7s} {'MAE':>6s} {'agree':>7s} {'sec':>6s}")
print("  " + "-" * 90)
rows = []
for hids, tern, label in LADDER:
    r = run_one(hids, tern)
    r["label"] = label
    rows.append(r)
    print(f"  {label:22s} {str(tern):>8s} {r['params']:9,d} {r['macs']:11,d} "
          f"{r['acc']:7.4f} {r['mae']:6.3f} {r['agree']*100:6.1f}% {r['seconds']:6.0f}",
          flush=True)

print()
print(f"  reference: real faces, neighbour agreement 60.5%")
print(f"  reference: final_lam5_20k = 203,264 params, 202,240 multiplies, "
      f"acc 0.7005, agree 60.7%")
best_small = max([r for r in rows if r["macs"] < 20000], key=lambda r: r["acc"])
print()
print(f"  best model under 20,000 multiplies: {best_small['label']} "
      f"({best_small['macs']:,} multiplies, acc {best_small['acc']:.4f})")
json.dump({"ladder": rows}, open(os.path.join(OUT, "size_ladder.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'size_ladder.json')}")
print("LADDER_DONE")
