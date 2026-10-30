#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Attack the one remaining quality gap: FRAGMENTATION.

Measured on 1000 faces: real faces break into 13.8 connected same-level regions, the 512
teacher into 13.9, but the 32-output students into 17-19. The students cut the face into
more pieces. TV does not fix this, because TV penalises the DIFFERENCE between neighbours
(it makes the field smooth) but never encourages two neighbours to land on the SAME level.
A smooth field with values 6.3 and 6.8 still rounds to 6 and 7 and still creates a border.

Two penalties are tested here that target levels rather than smoothness:

  integer attraction  beta * mean( distance of each output to the nearest integer level )
      pushes outputs to commit to a level instead of hovering between two, which should
      merge regions wherever the true face is uniform

  total variation    lam * mean( |neighbouring difference| )
      the control, which the earlier sweep showed to be weak

Both are applied on the 22->128->128->32 architecture, the best cost/quality point found so
far (23,296 multiplies, accuracy 0.7392, 18.9 blobs at baseline).
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
OUT = os.path.join(ROOT, "results", "fragment")
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


def tv_cont(v):
    p = v.view(-1, 4, 8)
    return ((p[:, 1:] - p[:, :-1]).abs().mean()
            + (p[:, :, 1:] - p[:, :, :-1]).abs().mean())


def integer_dist(v):
    """Distance of each output to the nearest integer level, capped at 0.5."""
    return (v - v.round()).abs().clamp(max=0.5).mean()


def measure(net, ns=1000):
    net.eval()
    bv = va[:ns]
    with torch.no_grad():
        out = net(COND[bv]).clamp(0, 15)
    dec = out.round().long()
    Rt = RTRUE[bv]
    acc = float((dec == Rt).float().mean())
    imgs = layout.image_from_left_half(dec.numpy())
    ag, blobs, det = [], [], []
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
        seen = np.zeros((8, 8), bool)
        cnt = 0
        for r_ in range(8):
            for c in range(8):
                if seen[r_, c]:
                    continue
                lv = g[r_, c]
                st = [(r_, c)]
                seen[r_, c] = True
                while st:
                    rr, cc = st.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        y, x = rr + dr, cc + dc
                        if 0 <= y < 8 and 0 <= x < 8 and not seen[y, x] and g[y, x] == lv:
                            seen[y, x] = True
                            st.append((y, x))
                cnt += 1
        blobs.append(cnt)
        det.append((np.abs(np.diff(g.astype(float), axis=1)).mean()
                    + np.abs(np.diff(g.astype(float), axis=0)).mean()) / 2)
    net.train()
    return {"acc": acc, "agree": float(np.mean(ag)), "blobs": float(np.mean(blobs)),
            "detail": float(np.mean(det))}


_ri = full[va[:1000]].reshape(-1, 8, 8)
_ag, _bl, _det = [], [], []
for g in _ri:
    pairs = same = 0
    for r_ in range(8):
        for c in range(8):
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r_ + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    pairs += 1
                    if g[r_, cc] == g[rr, cc]:
                        same += 1
    _ag.append(same / pairs)
    seen = np.zeros((8, 8), bool)
    cnt = 0
    for r_ in range(8):
        for c in range(8):
            if seen[r_, c]:
                continue
            lv = g[r_, c]
            st = [(r_, c)]
            seen[r_, c] = True
            while st:
                rr, cc = st.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    y, x = rr + dr, cc + dc
                    if 0 <= y < 8 and 0 <= x < 8 and not seen[y, x] and g[y, x] == lv:
                        seen[y, x] = True
                        st.append((y, x))
            cnt += 1
    _bl.append(cnt)
    _det.append((np.abs(np.diff(g.astype(float), axis=1)).mean()
                 + np.abs(np.diff(g.astype(float), axis=0)).mean()) / 2)
REAL = {"agree": float(np.mean(_ag)), "blobs": float(np.mean(_bl)),
        "detail": float(np.mean(_det))}
print(f"REAL: agreement {REAL['agree']*100:.1f}%  blobs {REAL['blobs']:.1f}  "
      f"detail {REAL['detail']:.2f}")
print(f"teacher 512: agreement 77.6%  blobs 13.9  detail 2.03  accuracy 0.7151")

WIDTHS = [128, 128]
STEPS = 10000
CONFIGS = [
    ("baseline", 0.0, 0.0),
    ("int 0.1", 0.0, 0.1),
    ("int 0.3", 0.0, 0.3),
    ("int 1.0", 0.0, 1.0),
    ("tv 0.3", 0.3, 0.0),
    ("tv 0.3 + int 0.3", 0.3, 0.3),
]
print()
print("=" * 104)
print("TARGETING FRAGMENTATION on 22->128->128->32")
print("=" * 104)
print(f"  {'config':20s} {'acc':>7s} {'agree':>8s} {'blobs':>7s} {'detail':>7s} "
      f"{'vs real blobs':>14s}")
print("  " + "-" * 70)
rows = []
for name, lam, beta in CONFIGS:
    torch.manual_seed(0)
    net = build(WIDTHS)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        bi = rng.integers(0, len(tr), 256)
        pred = net(COND[tr[bi]]).clamp(0, 15)
        loss = F.mse_loss(pred, RTRUE[tr[bi]].float())
        if lam > 0:
            loss = loss + lam * tv_cont(pred)
        if beta > 0:
            loss = loss + beta * integer_dist(pred)
        opt.zero_grad(); loss.backward(); opt.step()
    ev = measure(net)
    ev.update({"config": name, "lam": lam, "beta": beta,
               "seconds": round(time.time() - t0, 1)})
    rows.append(ev)
    print(f"  {name:20s} {ev['acc']:7.4f} {ev['agree']*100:7.1f}% {ev['blobs']:7.1f} "
          f"{ev['detail']:7.2f} {ev['blobs']-REAL['blobs']:+13.1f}", flush=True)
    d = os.path.join(OUT, name.replace(" ", "_").replace(".", "p"))
    os.makedirs(d, exist_ok=True)
    torch.save({"G": net.state_dict(),
                "args": {"widths": WIDTHS, "steps": STEPS, "lam": lam, "beta": beta,
                         "mode": "fragment"}}, os.path.join(d, "final.pt"))

print()
print("=" * 104)
print("VERDICT")
print("=" * 104)
best = min(rows, key=lambda r: abs(r["blobs"] - REAL["blobs"]) + 3 * max(0, 0.72 - r["acc"]))
print(f"  real blobs {REAL['blobs']:.1f}; baseline student {rows[0]['blobs']:.1f}")
b = min(rows, key=lambda r: r["blobs"])
print(f"  fewest blobs: {b['config']} -> {b['blobs']:.1f} (accuracy {b['acc']:.4f})")
print(f"  best balance (blobs near real, accuracy >= 0.72): {best['config']} "
      f"-> blobs {best['blobs']:.1f}, acc {best['acc']:.4f}")
json.dump({"real": REAL, "rows": rows}, open(os.path.join(OUT, "fragment.json"), "w"),
          indent=2)
print(f"\nwrote {os.path.join(OUT, 'fragment.json')}")
print("FRAGMENT_DONE")
