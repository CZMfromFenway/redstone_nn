#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
The right penalty for fragmentation: TV on the ROUNDED output, via straight-through.

WHY THE OBVIOUS PENALTIES FAIL
Real faces break into 13.8 connected same-level regions; the 32-output student into 20-22.
Neither of the two obvious penalties fixes this:

  TV on the continuous field  penalises the SIZE of neighbour differences. But two
      neighbours holding 6.49 and 6.51 have a difference of 0.02, which TV barely sees,
      and they ROUND TO DIFFERENT LEVELS. The penalty is blind to exactly the event that
      creates a border.
  integer attraction          penalises distance to the nearest level. But MSE already
      pulls outputs onto integer targets, so the model is nearly integral anyway and this
      term is redundant. Measured: blobs 22.4 -> 22.5, no effect.

THE FIX
Penalise the difference between neighbours AFTER rounding:
    y_st = y + (round(y) - y).detach()        # forward rounds, backward is identity
    penalty = mean | y_st[i] - y_st[j] |      # over the 4-neighbourhood
Now the penalty is large exactly when two neighbours land on different levels. The
straight-through detach keeps the gradient flowing to y, so the term is trainable.

This is what the 512-logit teacher gets for free: TV on its probability distributions makes
neighbouring DISTRIBUTIONS similar, so their argmaxes agree. This reproduces that effect
for a continuous output.
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
OUT = os.path.join(ROOT, "results", "fragment2")
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


def tv_rounded(v):
    """TV on the rounded field with a straight-through estimator."""
    vr = v + (v.round() - v).detach()
    p = vr.view(-1, 4, 8)
    return ((p[:, 1:] - p[:, :-1]).abs().mean()
            + (p[:, :, 1:] - p[:, :, :-1]).abs().mean())


def tv_cont(v):
    p = v.view(-1, 4, 8)
    return ((p[:, 1:] - p[:, :-1]).abs().mean()
            + (p[:, :, 1:] - p[:, :, :-1]).abs().mean())


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


# real reference on the same sample
def blobs_of(g):
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
    return cnt


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
    _bl.append(blobs_of(g))
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
    ("baseline", "none", 0.0),
    ("tv rounded 0.02", "round", 0.02),
    ("tv rounded 0.05", "round", 0.05),
    ("tv rounded 0.1", "round", 0.1),
    ("tv rounded 0.2", "round", 0.2),
    ("tv cont 0.1 (control)", "cont", 0.1),
]
print()
print("=" * 104)
print("TV ON THE ROUNDED FIELD (straight-through) vs TV on the continuous field")
print("=" * 104)
print(f"  {'config':24s} {'acc':>7s} {'agree':>8s} {'blobs':>7s} {'detail':>7s} "
      f"{'vs real':>9s}")
print("  " + "-" * 72)
rows = []
for name, kind, w in CONFIGS:
    torch.manual_seed(0)
    net = build(WIDTHS)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        bi = rng.integers(0, len(tr), 256)
        pred = net(COND[tr[bi]]).clamp(0, 15)
        loss = F.mse_loss(pred, RTRUE[tr[bi]].float())
        if kind == "round" and w > 0:
            loss = loss + w * tv_rounded(pred)
        elif kind == "cont" and w > 0:
            loss = loss + w * tv_cont(pred)
        opt.zero_grad(); loss.backward(); opt.step()
    ev = measure(net)
    ev.update({"config": name, "kind": kind, "w": w,
               "seconds": round(time.time() - t0, 1)})
    rows.append(ev)
    print(f"  {name:24s} {ev['acc']:7.4f} {ev['agree']*100:7.1f}% {ev['blobs']:7.1f} "
          f"{ev['detail']:7.2f} {ev['blobs']-REAL['blobs']:+8.1f}", flush=True)
    d = os.path.join(OUT, name.replace(" ", "_").replace(".", "p"))
    os.makedirs(d, exist_ok=True)
    torch.save({"G": net.state_dict(),
                "args": {"widths": WIDTHS, "steps": STEPS, "kind": kind, "w": w,
                         "mode": "fragment2"}}, os.path.join(d, "final.pt"))

print()
print("=" * 104)
print("VERDICT")
print("=" * 104)
base = rows[0]
best = min([r for r in rows if r["acc"] >= 0.70], key=lambda r: r["blobs"],
           default=None)
print(f"  baseline: blobs {base['blobs']:.1f}, acc {base['acc']:.4f}")
if best:
    print(f"  best with acc>=0.70: {best['config']} -> blobs {best['blobs']:.1f} "
          f"({best['blobs']-base['blobs']:+.1f} vs baseline), acc {best['acc']:.4f}")
print(f"  real is {REAL['blobs']:.1f}")
json.dump({"real": REAL, "rows": rows},
          open(os.path.join(OUT, "fragment2.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'fragment2.json')}")
print("FRAGMENT2_DONE")
