#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
The fix the diagnosis points to: train the loss on the ROUNDED output.

DIAGNOSIS (measures just above):
    the 32-output students put only 25-29% of their values within 0.1 of a level, with a
    mean fractional part of 0.22. The 512 teacher emits one of 16 levels and therefore
    sits exactly on a level 100% of the time.
    MSE on a continuous target rewards predicting the MEAN, so where the model is unsure
    between level 6 and 7 it answers 6.5. That is optimal for MSE and wrong for the
    picture: 6.5 renders as a gradient between two levels, which is the visible softness.
    The block-count and TV penalties could not fix this because they do not change what
    the loss considers a good answer.

THE FIX
    train against the rounded output, with a straight-through estimator:
        y_st   = y + (round(y) - y).detach()
        loss   = MSE(y_st, target)
    Forward, the loss sees the image that will actually be drawn. Backward, gradients
    still reach y. This makes "commit to a level" the thing being optimised, so the
    fractional part should collapse toward 0.

Compared against the plain continuous baseline at the same width and step budget, plus a
control where the target is the teacher's soft expectation (which keeps the averaging
behaviour and should NOT fix the gradient issue).
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
OUT = os.path.join(ROOT, "results", "commit")
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
U_, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
COND = torch.tensor(((X - mu) @ comps.T) / sd[0], dtype=torch.float32)
RTRUE = torch.from_numpy(full.astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]


def build(widths):
    seq = []
    d = 22
    for h in widths:
        seq += [nn.Linear(d, h), nn.ReLU()]
        d = h
    seq += [nn.Linear(d, 32)]
    return nn.Sequential(*seq)


def measure(net, ns=1000):
    net.eval()
    bv = va[:ns]
    with torch.no_grad():
        out = net(COND[bv]).clamp(0, 15)
    dec = out.round().long()
    frac = (out - out.round()).abs()
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
            "on_level": float((frac < 0.1).float().mean()),
            "mean_frac": float(frac.mean())}


WIDTHS = [128, 128]
STEPS = 9000
CONFIGS = [
    ("plain continuous (baseline)", "plain"),
    ("rounded target (ST)", "rounded"),
    ("rounded + soft-start", "rounded_gauss"),
]
print("=" * 100)
print("FORCING THE OUTPUT TO COMMIT TO A LEVEL")
print("=" * 100)
print(f"  {'config':30s} {'acc':>7s} {'on-level%':>10s} {'mean|frac|':>11s} "
      f"{'agree':>8s} {'detail':>7s}")
print("  " + "-" * 78)
rows = []
for name, kind in CONFIGS:
    torch.manual_seed(0)
    net = build(WIDTHS)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        bi = rng.integers(0, len(tr), 256)
        tgt = RTRUE[tr[bi]].float()
        pred = net(COND[tr[bi]]).clamp(0, 15)
        if kind == "plain":
            loss = F.mse_loss(pred, tgt)
        elif kind == "rounded":
            y_st = pred + (pred.round() - pred).detach()
            loss = F.mse_loss(y_st, tgt)
        else:
            y_st = pred + (pred.round() - pred).detach()
            # ease in: early steps use the plain objective so the model first learns the
            # rough shape, then switch to the committed objective
            w = min(1.0, step / (0.4 * STEPS))
            loss = (1 - w) * F.mse_loss(pred, tgt) + w * F.mse_loss(y_st, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    ev = measure(net)
    ev.update({"config": name, "kind": kind, "seconds": round(time.time() - t0, 1)})
    rows.append(ev)
    print(f"  {name:30s} {ev['acc']:7.4f} {ev['on_level']*100:9.1f}% "
          f"{ev['mean_frac']:11.3f} {ev['agree']*100:7.1f}% {ev['detail']:7.2f}",
          flush=True)
    d = os.path.join(OUT, kind)
    os.makedirs(d, exist_ok=True)
    torch.save({"G": net.state_dict(),
                "args": {"widths": WIDTHS, "steps": STEPS, "kind": kind,
                         "mode": "commit"}}, os.path.join(d, "final.pt"))

base = rows[0]
print()
print("=" * 100)
print("VERDICT")
print("=" * 100)
print(f"  baseline: {base['on_level']*100:.1f}% of outputs on a level, accuracy "
      f"{base['acc']:.4f}")
for r in rows[1:]:
    print(f"  {r['config']}: {r['on_level']*100:.1f}% on a level "
          f"({(r['on_level']-base['on_level'])*100:+.1f} points), accuracy {r['acc']:.4f} "
          f"({r['acc']-base['acc']:+.4f})")
best = max(rows[1:], key=lambda r: r["on_level"]) if len(rows) > 1 else None
if best and best["on_level"] > base["on_level"] + 0.3:
    print(f"\n  -> committing to levels WORKS: {best['on_level']*100:.0f}% of outputs now "
          f"sit on a level, which should remove the gradients.")
else:
    print("\n  -> the straight-through rounded loss did NOT make the output commit.")
json.dump({"rows": rows}, open(os.path.join(OUT, "commit.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'commit.json')}")
print("COMMIT_DONE")
