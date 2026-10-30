#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
PHASE 5 of 512 -> 32: after the output is 32 values, the middle layer becomes the
bottleneck and must be shrunk.

Once the final layer is 256 -> 32 instead of 256 -> 512, the multiply budget shifts:

    22->256->256->512   5,632 + 65,536 + 131,072 = 202,240
    22->256->256->32    5,632 + 65,536 +   8,192 =  79,360     (61% saved)
    22->128->128->32    2,816 + 16,384 +   4,096 =  23,296     (88% saved)
    22->64 ->64 ->32    1,408 +  4,096 +   2,048 =   7,552     (96% saved)

So shrinking the hidden layers is now the dominant lever. This trains a ladder of widths
with the SAME output format and the SAME objective as the winning output experiment, and
reports the quality cost of each step. The numbers are computed on one fixed 1000-face
hold-out sample, with the real reference measured on the same sample, so they are
comparable to every other result in this project.

Two objectives are offered because earlier measurements disagreed about which target is
better:
  hard     : the true grey level (this beat distillation on accuracy: 0.7719 vs 0.7348)
  distill  : the 512 teacher's expected level
"""
import argparse, json, os, sys, time
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
OUT = os.path.join(ROOT, "results", "shrink5")
os.makedirs(OUT, exist_ok=True)

AP = argparse.ArgumentParser()
AP.add_argument("--objective", default="hard", choices=["hard", "distill"])
AP.add_argument("--steps", type=int, default=10000)
AP.add_argument("--widths", default="256,256;128,128;64,64;64;32")
AP.add_argument("--tv", type=float, default=0.0, help="TV on the continuous field")
AP.add_argument("--threads", type=int, default=10)
A = AP.parse_args()
torch.set_num_threads(A.threads)


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

# distillation targets, precomputed once
EXP_TR = None
if A.objective == "distill":
    print("precomputing the teacher's expected levels ...")
    EXP_TR = np.empty((len(tr), 32), dtype=np.float32)
    for s in range(0, len(tr), 8192):
        e = min(s + 8192, len(tr))
        with torch.no_grad():
            lg = T.logits(COND[tr[s:e]])
            p = F.softmax(lg, -1)
            EXP_TR[s:e] = (p * torch.arange(NLEV, dtype=torch.float32)).sum(-1).numpy()
    print("  done")


def build(widths):
    seq = []
    d = 22
    for h in widths:
        seq += [nn.Linear(d, h), nn.ReLU()]
        d = h
    seq += [nn.Linear(d, 32)]
    return nn.Sequential(*seq)


def mults(widths):
    m = 22 * (widths[0] if widths else 32)
    d = widths[0] if widths else 22
    for i, h in enumerate(widths):
        nxt = widths[i + 1] if i + 1 < len(widths) else 32
        m += h * nxt
    return m


def tv_cont(v):
    p = v.view(-1, 4, 8)
    return ((p[:, 1:] - p[:, :-1]).abs().mean()
            + (p[:, :, 1:] - p[:, :, :-1]).abs().mean())


def evaluate(net, ns=1000):
    net.eval()
    bv = va[:ns]
    with torch.no_grad():
        out = net(COND[bv]).clamp(0, 15)
    dec = out.round().long()
    Rt = RTRUE[bv]
    acc = float((dec == Rt).float().mean())
    mae = float((dec.float() - Rt.float()).abs().mean())
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
    pats = set()
    for g in imgs:
        cell = np.round(g.reshape(2, 4, 2, 4).mean(axis=(1, 3)) / 15 * 5).astype(int)
        pats.add(tuple(cell.reshape(-1).tolist()))
    net.train()
    return {"acc": acc, "mae": mae, "agree": float(np.mean(ag)),
            "detail": float(np.mean(det)), "patterns": len(pats)}


# real reference on the same sample
_rb = va[:1000]
_ri = full[_rb].reshape(-1, 8, 8)
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
print(f"real reference: agreement {np.mean(_ag)*100:.1f}%  detail {np.mean(_det):.2f}")
print(f"teacher (512): 202,240 multiplies, accuracy 0.7151")

ladder = []
for spec in A.widths.split(";"):
    ladder.append([int(x) for x in spec.split(",") if x.strip()])

print()
print("=" * 104)
print(f"PHASE 5 LADDER  objective={A.objective}  tv={A.tv}  steps={A.steps}")
print("=" * 104)
print(f"  {'widths':22s} {'multiplies':>11s} {'vs 512':>8s} {'acc':>7s} {'agree':>8s} "
      f"{'detail':>7s} {'patterns':>9s}")
print("  " + "-" * 84)
results = []
for widths in ladder:
    t0 = time.time()
    net = build(widths)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    for step in range(1, A.steps + 1):
        bi = rng.integers(0, len(tr), 256)
        pred = net(COND[tr[bi]]).clamp(0, 15)
        if A.objective == "hard":
            loss = F.mse_loss(pred, RTRUE[tr[bi]].float())
        else:
            loss = F.mse_loss(pred, torch.from_numpy(EXP_TR[bi]))
        if A.tv > 0:
            loss = loss + A.tv * tv_cont(pred)
        opt.zero_grad(); loss.backward(); opt.step()
    ev = evaluate(net)
    m = mults(widths)
    ev.update({"widths": widths, "multiplies": m,
               "vs512": 1 - m / 202240, "seconds": round(time.time() - t0, 1)})
    results.append(ev)
    sp_ = "->".join(map(str, widths))
    print(f"  22->{sp_:18s} {m:11,d} {100*(1-m/202240):7.1f}% {ev['acc']:7.4f} "
          f"{ev['agree']*100:7.1f}% {ev['detail']:7.2f} {ev['patterns']:9d}", flush=True)
    torch.save({"G": net.state_dict(), "args": {"widths": widths, "objective": A.objective,
                                                "steps": A.steps, "tv": A.tv}},
               os.path.join(OUT, "ladder_" + "_".join(map(str, widths)) + ".pt"))

json.dump({"objective": A.objective, "tv": A.tv, "real_agree": float(np.mean(_ag)),
           "real_detail": float(np.mean(_det)), "results": results},
          open(os.path.join(OUT, "ladder.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'ladder.json')}")
print("PHASE5_DONE")
