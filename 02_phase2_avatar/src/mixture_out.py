#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Can a MIXTURE output beat the single continuous value? This attacks the root cause.

DIAGNOSIS RECAP
    MSE's optimum is the conditional MEAN. Measured at the start of the project: 28.14% of
    positions have at least two conditional modes holding >=20% mass each. At those
    positions the mean sits BETWEEN them (target 6 or 7 -> answer 6.5), which renders as a
    gradient instead of a step. The 512-logit teacher represents the bimodality explicitly
    and its argmax commits to one mode; a single scalar cannot.

    Every penalty I tried failed, including straight-through rounding, and the reason is
    now clear: a straight-through estimator has the SAME gradient as the unrounded loss,
    so it cannot change what the model learns.

THE CANDIDATE
    Emit TWO values per position (64 outputs total). Train with a winner-take-all loss so
    the two components can specialise on the two modes:
        loss = mean over positions of min( (m1 - t)^2, (m2 - t)^2 )
    Decode by taking the component that sits closest to a level, then rounding:
        pick k minimising |m_k - round(m_k)|, output round(m_k)
    That committal choice is the point: it reproduces the teacher's behaviour of picking a
    mode instead of averaging.

Cost: 22->128->128->64 is 2,816 + 16,384 + 8,192 = 27,392 multiplies, still 13.5% of the
512 teacher, and the output is 64 values rather than 32.
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
OUT = os.path.join(ROOT, "results", "mixture")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(10)

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.POS_MIRROR_COL]).mean(1) >= 0.95 if False else \
     (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U_, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
COND = torch.tensor(((X - mu) @ comps.T) / sd[0], dtype=torch.float32)
RTRUE = torch.from_numpy(full.astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]


def build(hids, out_dim):
    seq = []
    d = 22
    for h in hids:
        seq += [nn.Linear(d, h), nn.ReLU()]
        d = h
    seq += [nn.Linear(d, out_dim)]
    return nn.Sequential(*seq)


def decode_mixture(raw):
    """raw: (n, 64) -> levels (n, 32). Pick the more committed component per position."""
    v = raw.view(-1, 32, 2).clamp(0, 15)
    # distance of each component from its nearest integer level
    commit = (v - v.round()).abs()
    # choose the component that is closest to a level (index 1 if the second is better)
    pick = (commit[:, :, 1] < commit[:, :, 0]).long()
    sel = v.gather(2, pick.unsqueeze(2)).squeeze(2)
    return sel.round().long(), sel


def measure(net, out_dim, ns=1000):
    net.eval()
    bv = va[:ns]
    with torch.no_grad():
        raw = net(COND[bv]).clamp(0, 15)
    if out_dim == 64:
        dec, sel = decode_mixture(raw)
        v = raw.view(-1, 32, 2)
        # how often is at least one component near a level?
        best_commit = (v - v.round()).abs().min(2).values
        on_level = float((best_commit < 0.1).float().mean())
        mean_frac = float(best_commit.mean())
    else:
        sel = raw
        dec = raw.round().long()
        frac = (raw - raw.round()).abs()
        on_level = float((frac < 0.1).float().mean())
        mean_frac = float(frac.mean())
    Rt = RTRUE[bv]
    acc = float((dec == Rt).float().mean())
    imgs = layout.image_from_left_half(dec.numpy())
    ag, det = [], []
    for g in imgs:
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
        ag.append(same / pairs)
        det.append((np.abs(np.diff(g.astype(float), axis=1)).mean()
                    + np.abs(np.diff(g.astype(float), axis=0)).mean()) / 2)
    net.train()
    return {"acc": acc, "agree": float(np.mean(ag)), "detail": float(np.mean(det)),
            "on_level": on_level, "mean_frac": mean_frac}


HIDS = [128, 128]
STEPS = 9000
CONFIGS = [
    ("single value (32 out)", 32),
    ("mixture of 2 (64 out)", 64),
    ("mixture of 3 (96 out)", 96),
]
print("=" * 100)
print("MIXTURE OUTPUT vs SINGLE VALUE  (22->128->128->out)")
print("=" * 100)
print(f"  {'config':28s} {'out':>4s} {'acc':>7s} {'on-level%':>10s} "
      f"{'mean|frac|':>11s} {'agree':>8s} {'detail':>7s}")
print("  " + "-" * 78)
rows = []
for name, out_dim in CONFIGS:
    torch.manual_seed(0)
    net = build(HIDS, out_dim)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    t0 = time.time()
    K = out_dim // 32
    for step in range(1, STEPS + 1):
        bi = rng.integers(0, len(tr), 256)
        tgt = RTRUE[tr[bi]].float()
        raw = net(COND[tr[bi]]).clamp(0, 15)
        if K == 1:
            loss = F.mse_loss(raw, tgt)
        else:
            v = raw.view(-1, 32, K)
            # winner-take-all: each position is served by whichever component fits best
            per = ((v - tgt.unsqueeze(2)) ** 2)
            loss = per.min(2).values.mean()
            # a light repulsion keeps the components from collapsing onto each other,
            # which would turn the mixture back into a single value
            if K > 1:
                diff = (v[:, :, :, None] - v[:, :, None, :]).abs()
                mask = 1 - torch.eye(K, device=raw.device)
                rep = (diff * mask).sum((2, 3)) / (K * (K - 1))
                loss = loss + 0.05 * F.relu(2.0 - rep).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ev = measure(net, out_dim)
    ev.update({"config": name, "out_dim": out_dim, "k": K,
               "seconds": round(time.time() - t0, 1)})
    macs = 22 * HIDS[0] + HIDS[0] * HIDS[1] + HIDS[1] * out_dim
    ev["multiplies"] = macs
    rows.append(ev)
    print(f"  {name:28s} {out_dim:4d} {ev['acc']:7.4f} {ev['on_level']*100:9.1f}% "
          f"{ev['mean_frac']:11.3f} {ev['agree']*100:7.1f}% {ev['detail']:7.2f}",
          flush=True)
    d = os.path.join(OUT, f"k{K}")
    os.makedirs(d, exist_ok=True)
    torch.save({"G": net.state_dict(),
                "args": {"hids": HIDS, "out_dim": out_dim, "k": K, "steps": STEPS,
                         "mode": "mixture"}}, os.path.join(d, "final.pt"))

print()
print("=" * 100)
print("VERDICT")
print("=" * 100)
base = rows[0]
for r in rows[1:]:
    print(f"  {r['config']}: acc {r['acc']:.4f} ({r['acc']-base['acc']:+.4f}), "
          f"on-level {r['on_level']*100:.1f}% ({(r['on_level']-base['on_level'])*100:+.1f}), "
          f"multiplies {r['multiplies']:,}")
print(f"\n  teacher: acc 0.7151, on-level 100%, 202,240 multiplies")
print(f"  baseline single value: acc {base['acc']:.4f}, on-level {base['on_level']*100:.1f}%")
best = max(rows[1:], key=lambda r: r["on_level"]) if len(rows) > 1 else None
if best and best["on_level"] > base["on_level"] + 0.10:
    print(f"\n  -> the mixture DOES commit more often: {best['on_level']*100:.0f}% vs "
          f"{base['on_level']*100:.0f}%. Worth pursuing.")
else:
    print("\n  -> the mixture did not materially increase commitment either.")
json.dump({"rows": rows, "teacher": {"acc": 0.7151, "on_level": 1.0,
                                     "multiplies": 202240}},
          open(os.path.join(OUT, "mixture.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'mixture.json')}")
print("MIXTURE_DONE")
