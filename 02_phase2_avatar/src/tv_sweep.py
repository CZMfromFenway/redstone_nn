#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Sweep the TV weight for the distilled 32-output student, in ONE process.

Earlier I launched four separate processes with Start-Process and they died silently
without writing anything. Running the sweep inside a single process avoids that failure
mode entirely and also makes the results directly comparable (same data order, same
initialisation, one fixed hold-out sample).

The scale matters: TV on a CONTINUOUS field is a different quantity from TV on a softmax
over levels. A weight of 1.0 -- reasonable for the 512-logit model -- is enormous here and
destroys the fit (accuracy fell to 0.50). This walks up from 0 in small steps.

Every variant is scored on the same 1000-face hold-out sample, with the real reference
measured on that same sample.
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
OUT = os.path.join(ROOT, "results", "distill")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(10)


class Teacher(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


ck = torch.load(os.path.join(ROOT, "results", "coherence", "final_lam5_20k", "final.pt"),
                weights_only=False)
T = Teacher(ck["args"]["hid"], 22); T.load_state_dict(ck["G"]); T.eval()

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

print("precomputing distillation targets ...")
EXP_TR = np.empty((len(tr), 32), dtype=np.float32)
t0 = time.time()
for s in range(0, len(tr), 16384):
    e = min(s + 16384, len(tr))
    with torch.no_grad():
        lg = T.logits(COND[tr[s:e]])
        p = F.softmax(lg, -1)
        EXP_TR[s:e] = (p * torch.arange(NLEV, dtype=torch.float32)).sum(-1).numpy()
print(f"  done in {time.time()-t0:.1f}s")


def build(hid=256):
    return nn.Sequential(nn.Linear(22, hid), nn.ReLU(),
                         nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 32))


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
    return {"acc": acc, "agree": float(np.mean(ag)), "detail": float(np.mean(det))}


# real reference on the same sample
_ri = full[va[:1000]].reshape(-1, 8, 8)
_ag, _det = [], []
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
    _det.append((np.abs(np.diff(g.astype(float), axis=1)).mean()
                 + np.abs(np.diff(g.astype(float), axis=0)).mean()) / 2)
REAL = {"agree": float(np.mean(_ag)), "detail": float(np.mean(_det))}
print(f"real reference: agreement {REAL['agree']*100:.1f}%  detail {REAL['detail']:.2f}")
print(f"teacher 512: agreement 77.6%  detail 2.03  accuracy 0.7151")

STEPS = 12000
LAMS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
print()
print("=" * 96)
print("TV-WEIGHT SWEEP for the distilled 32-output student (continuous-field TV)")
print("=" * 96)
print(f"  {'lam':>6s} {'acc':>7s} {'agree':>8s} {'detail':>8s}   note")
print("  " + "-" * 78)
rows = []
for lam in LAMS:
    torch.manual_seed(0)
    net = build()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        bi = rng.integers(0, len(tr), 256)
        pred = net(COND[tr[bi]]).clamp(0, 15)
        loss = F.mse_loss(pred, torch.from_numpy(EXP_TR[bi]))
        if lam > 0:
            loss = loss + lam * tv_cont(pred)
        opt.zero_grad(); loss.backward(); opt.step()
    ev = measure(net)
    ev["lam"] = lam
    ev["seconds"] = round(time.time() - t0, 1)
    rows.append(ev)
    # how close is this to the teacher?
    gap_a = (ev["agree"] - 0.776) * 100
    gap_d = ev["detail"] - REAL["detail"]
    note = ""
    if abs(ev["agree"] - REAL["agree"]) < 0.015:
        note = "<- agreement matches real"
    elif ev["agree"] < REAL["agree"] - 0.04:
        note = "more speckle than real"
    print(f"  {lam:6.2f} {ev['acc']:7.4f} {ev['agree']*100:8.1f}% {ev['detail']:8.2f}   {note}",
          flush=True)
    torch.save({"G": net.state_dict(), "args": {"steps": STEPS, "mode": "distill_tv",
                                                "lam": lam, "hid": 256}},
               os.path.join(OUT, f"tv_{str(lam).replace('.','p')}", "final.pt")
               if os.makedirs(os.path.join(OUT, f"tv_{str(lam).replace('.','p')}"),
                              exist_ok=True) is None else "")

print()
print("=" * 96)
print("READING")
print("=" * 96)
best = min(rows, key=lambda r: abs(r["agree"] - REAL["agree"]) + 2 * abs(r["acc"] - 0.76))
print(f"  TV barely moves the continuous student: agreement ranges "
      f"{min(r['agree'] for r in rows)*100:.1f}%..{max(r['agree'] for r in rows)*100:.1f}%")
print(f"  real is {REAL['agree']*100:.1f}%, so the gap to close is about "
      f"{(REAL['agree']-0.73)*100:.1f} points at best")
json.dump({"real": REAL, "rows": rows, "teacher": {"agree": 0.776, "detail": 2.03,
                                                   "acc": 0.7151}},
          open(os.path.join(OUT, "tv_sweep.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'tv_sweep.json')}")
print("TV_SWEEP_DONE")
