#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
The decisive comparison, all on the SAME 1000 held-out faces.

An earlier measurement compared variants using neighbour-agreement values whose REAL
reference was computed on a different sample: the real value came out 78.6% on 16 faces
and 60.5% on 400. That makes cross-sample comparisons meaningless. Every number here is
computed on one fixed sample of 1000 faces, real reference included.

Variants:
  real             the held-out faces themselves
  teacher_argmax   final_lam5_20k, 512 logits, argmax decode
  teacher_expect   the same teacher, decoded by rounding its expected level
  hard32           trained on the hard level (a 32-output run without distillation)
  distill_plain    distilled from the teacher's expected level, no TV  <- the candidate
  s32_tv30         the earlier 32-output run with TV lam=30 (for context)

Reported: accuracy, neighbour agreement, texture detail, distinct coarse patterns, and
the redstone multiply count so the trade is explicit.
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
OUT = os.path.join(ROOT, "results", "distill")
os.makedirs(OUT, exist_ok=True)


class Teacher(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


class Student(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 32))
    def forward(self, c):
        return self.net(c).clamp(0, 15)


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
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]

NS = 1000
bv = va[:NS]
Rt = RTRUE[bv].numpy()


def agree_of(imgs):
    out = []
    for g in np.asarray(imgs).reshape(-1, 8, 8):
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
        out.append(same / pairs)
    return float(np.mean(out))


def detail(imgs):
    d = []
    for g in np.asarray(imgs).reshape(-1, 8, 8).astype(float):
        d.append((np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()) / 2)
    return float(np.mean(d))


def distinct(imgs):
    pats = set()
    for g in np.asarray(imgs).reshape(-1, 8, 8).astype(float):
        cell = np.round(g.reshape(2, 4, 2, 4).mean(axis=(1, 3)) / 15 * 5).astype(int)
        pats.add(tuple(cell.reshape(-1).tolist()))
    return len(pats)


rows = []


def add(name, levels, cost):
    imgs = layout.image_from_left_half(np.asarray(levels).reshape(-1, 32))
    acc = float((np.asarray(levels).reshape(-1, 32) == Rt).mean())
    rows.append({"name": name, "acc": acc, "agree": agree_of(imgs),
                 "detail": detail(imgs), "distinct": distinct(imgs), "multiplies": cost})
    print(f"  {name:30s} {acc:7.4f} {agree_of(imgs)*100:8.1f}% {detail(imgs):8.2f} "
          f"{distinct(imgs):9d} {cost:12,d}", flush=True)


print("=" * 108)
print(f"DECISIVE COMPARISON on the same {NS} held-out faces")
print("=" * 108)
print(f"  {'variant':30s} {'acc':>7s} {'agree':>9s} {'detail':>8s} {'patterns':>9s} "
      f"{'multiplies':>12s}")
print("  " + "-" * 104)

real = full[bv].reshape(-1, 32)
rows.append({"name": "real (reference)", "acc": float("nan"), "agree": agree_of(real),
             "detail": detail(real), "distinct": distinct(real), "multiplies": 0})
print(f"  {'real (reference)':30s} {'--':>7s} {agree_of(real)*100:8.1f}% "
      f"{detail(real):8.2f} {distinct(real):9d} {0:12,d}")

with torch.no_grad():
    lg = T.logits(COND[bv])
    t_arg = lg.argmax(-1).numpy()
    p = F.softmax(lg, -1)
    t_exp = (p * torch.arange(NLEV, dtype=torch.float32)).sum(-1).numpy()
add("teacher 512 (argmax)", t_arg, 202240)
add("teacher 512 (expect.round)", t_exp.round(), 202240)

for tag, name, costf in (
        ("distill_plain", "STUDENT distill (no TV)", lambda h: 22 * h + h * h + h * 32),
        ("distill_tv1", "student distill + TV lam=1", lambda h: 22 * h + h * h + h * 32),
):
    pth = os.path.join(ROOT, "results", "distill", tag, "final.pt")
    if not os.path.exists(pth):
        continue
    ckS = torch.load(pth, weights_only=False)
    hid = ckS["args"].get("hid", 256) if isinstance(ckS["args"], dict) else 256
    S = Student(hid, 22); S.load_state_dict(ckS["S"]); S.eval()
    with torch.no_grad():
        sv = S(COND[bv]).numpy()
    add(name, sv.round(), costf(hid))

# the earlier non-distilled 32-output runs, for context
for tag, name in (("s32_tv5", "earlier 32out hard+TV5"),
                  ("s32_tv30", "earlier 32out hard+TV30")):
    pth = os.path.join(ROOT, "results", "shrink", tag, "final.pt")
    if not os.path.exists(pth):
        continue
    ckS = torch.load(pth, weights_only=False)
    Gs = Student(ckS["args"]["hid"], 22); Gs.load_state_dict(ckS["G"]); Gs.eval()
    with torch.no_grad():
        sv = Gs(COND[bv]).numpy()
    add(name, sv.round(), 22 * 256 + 256 * 256 + 256 * 32)

print()
print("=" * 108)
print("VERDICT")
print("=" * 108)
stu = [r for r in rows if "distill (no TV)" in r["name"]]
tea = [r for r in rows if "teacher 512 (argmax)" in r["name"]]
if stu and tea:
    s, t = stu[0], tea[0]
    print(f"  student vs teacher:")
    print(f"    accuracy  {s['acc']:.4f} vs {t['acc']:.4f}   ({s['acc']-t['acc']:+.4f})")
    print(f"    agreement {s['agree']*100:.1f}% vs {t['agree']*100:.1f}%   "
          f"({(s['agree']-t['agree'])*100:+.1f} points)")
    print(f"    detail    {s['detail']:.2f} vs {t['detail']:.2f}")
    print(f"    patterns  {s['distinct']} vs {t['distinct']}")
    print(f"    multiplies {s['multiplies']:,} vs {t['multiplies']:,}   "
          f"({100*(1-s['multiplies']/t['multiplies']):.1f}% fewer)")
    ok = s["acc"] >= t["acc"] - 0.01 and s["agree"] >= t["agree"] - 0.02
    print()
    print(f"  {'PASS' if ok else 'FAIL'}: the 32-output student matches the 512 teacher "
          f"while using {100*(1-s['multiplies']/t['multiplies']):.0f}% fewer multiplies")
json.dump({"n": NS, "rows": rows}, open(os.path.join(OUT, "decisive.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'decisive.json')}")
print("DECISIVE_DONE")
