#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Train the PCA-k condition at the CURRENT BEST 32-output recipe, for a ladder of k.

WHY THIS EXPERIMENT
The goal for the project is "generate a plausible RANDOM avatar", not "reproduce a
specified face". Under that goal, accuracy against a particular target face is the wrong
metric -- the right one is the share of random draws a human rates as acceptable. Earlier
measurements hinted that fewer PCA components give SMOOTHER output (agreement 74.5% at
k=4 versus 65.1% at k=22) which is closer to real faces (77.6%), so a small k might
actually yield MORE usable random avatars despite lower accuracy.

THE RECIPE, TAKEN VERBATIM FROM THE BEST 32-OUTPUT RUN
    results/shrink/s32_tv5, which reached accuracy 0.7719 and agreement 72.8%:
        condition = first k PCA coefficients (standardised), NO latent noise
        architecture   k -> 256 -> 256 -> 32 continuous values
        loss           MSE against the true grey level
                       + 5.0 * total variation on a soft distribution built from the
                         continuous output
                       + adversarial (weight 1.0)
        12000 steps, batch 256, Adam 3e-4, hid 256
The ONLY change here is k. Everything else is identical, so any difference in yield is
attributable to the number of condition dimensions.

Outputs one checkpoint per k under results/dimsweep/dim{k}/.
"""
import argparse
import json
import os
import sys
import time
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
OUT = os.path.join(ROOT, "results", "dimsweep")
os.makedirs(OUT, exist_ok=True)

AP = argparse.ArgumentParser()
AP.add_argument("--dims", default="1,2,4,8", help="comma separated k values to train")
AP.add_argument("--steps", type=int, default=12000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--hid", type=int, default=256)
AP.add_argument("--layers", type=int, default=2,
                    help="1 = k->hid->32,  2 = k->hid->hid->32")
AP.add_argument("--lam", type=float, default=5.0)
AP.add_argument("--adv", type=float, default=1.0)
AP.add_argument("--eval-every", type=int, default=3000)
AP.add_argument("--threads", type=int, default=10)
AP.add_argument("--contrast-weight", type=float, default=0.0,
                help="penalise shrinkage of per-sample spread; 0 disables")
AP.add_argument("--tag", default=None,
                help="output directory name; defaults to dim{k}")
AP.add_argument("--stretch-targets", action="store_true",
                help="stretch every target face to [0,15] so a face always means a "
                     "full-range face; removes the low-contrast option entirely")
A = AP.parse_args()
torch.set_num_threads(A.threads)

# ---------------------------------------------------------------- data
# THE CONDITION AND THE TARGET MUST COME FROM THE SAME ARRAY.
# An earlier version built the PCA basis from the symmetry-filtered RGB luminance but took
# the targets from L0_full_idx.npy. Those are two different row orderings of the same faces
# (they disagree at 75% of rows, and only partially line up at a +2 offset), so every
# condition was paired with a DIFFERENT face's target. Training then converged to the
# conditional mean: accuracy 0.05, i.e. chance, with suspiciously high neighbour agreement
# because the output was a smooth average face.
# L0_full_idx is already the canonical symmetric 8x8 image, so both sides come from it.
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
X = full.reshape(len(full), -1).astype(np.float64)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U_, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
ZZ = (((X - mu) @ comps.T) / sd[0]).astype(np.float32)     # (N,22) standardised
RTRUE = torch.from_numpy(full)[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
# the discriminator scores a FULL 8x8 image, so keep the real faces in that shape as well
RIMG = torch.from_numpy(full)
# TGT is what the model is asked to produce. With --stretch-targets each face is
# rescaled to use the whole 0..15 range, so low contrast is not an available answer.
TGT = RTRUE.float()
if A.stretch_targets:
    _lo = TGT.min(1, keepdim=True)[0]
    _hi = TGT.max(1, keepdim=True)[0]
    TGT = (TGT - _lo) / (_hi - _lo).clamp_min(1e-6) * 15.0
    RTRUE = TGT.round().long()
    RIMG = torch.from_numpy(layout.image_from_left_half(np.asarray(RTRUE)))
    print(f"  stretch-targets ON: every target now spans 0..15 "
          f"(mean spread {float((TGT.max(1)[0]-TGT.min(1)[0]).mean()):.2f})")
else:
    TGT = TGT  # unchanged
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
print(f"data: train {len(tr):,}  val {len(va):,}   PCA basis 22 components")

# ---------------------------------------------------------------- alignment guard
# A linear readout from the first k PCA coordinates to the target must beat chance by a wide
# margin. If it does not, the condition and the target are not describing the same face, and
# no amount of training will help -- so stop here instead of wasting a run.
_g = np.random.default_rng(1).choice(len(tr), min(4000, len(tr)), replace=False)
for _k in sorted({int(s) for s in A.dims.split(",") if s.strip()}):
    _A = np.c_[ZZ[tr[_g]][:, :_k].astype(np.float64), np.ones(len(_g))]
    _t = RTRUE[tr[_g]].numpy().astype(np.float64)
    _W = np.linalg.lstsq(_A, _t, rcond=None)[0]
    _pred = np.clip(np.rint(_A @ _W), 0, 15)
    _acc = float((_pred == _t).mean())
    print(f"  alignment guard: linear readout from {_k:2d} PCA -> target accuracy {_acc:.4f}"
          f"  (chance is 0.0625)")
    if _acc < 0.12:
        raise SystemExit(
            f"ABORT: with {_k} PCA inputs a linear readout reaches only {_acc:.3f}, "
            "so the condition does not describe the target. Check the data alignment "
            "before training anything.")



class Gen(nn.Module):
    def __init__(self, k, hid=256, layers=2):
        super().__init__()
        # layers=1 lets the width sweep include a single-hidden-layer generator, so the
        # comparison covers shape as well as width rather than assuming two layers are needed.
        if layers == 1:
            self.net = nn.Sequential(nn.Linear(k, hid), nn.ReLU(), nn.Linear(hid, 32))
        else:
            self.net = nn.Sequential(nn.Linear(k, hid), nn.ReLU(),
                                     nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 32))
    def forward(self, c):
        return self.net(c)


class Dis(nn.Module):
    def __init__(self, k, c=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(NLEV, emb)
        self.conv = nn.Sequential(
            nn.Conv2d(emb, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 64 + k, 64),
                                  nn.LeakyReLU(0.2, True), nn.Linear(64, 1))
    def forward(self, idx_img, cond):
        oh = F.one_hot(idx_img, NLEV).permute(0, 3, 1, 2).float()
        e = torch.einsum("bchw,ce->behw", oh, self.emb.weight)
        return self.head(torch.cat([self.conv(e).flatten(1), cond], 1))


def tv32(raw):
    """TV on a soft distribution built from the continuous output, as in s32_tv5."""
    mu_l = raw.clamp(0, 15)
    lv = torch.arange(NLEV, dtype=torch.float32)
    logits = -((lv[None, None, :] - mu_l[:, :, None]) ** 2) / (2 * 0.35 ** 2)
    p = F.softmax(logits, -1).reshape(-1, 4, 8, NLEV)
    return (p[:, 1:] - p[:, :-1]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()


def speckle(imgs):
    out = []
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
        out.append(same / pairs)
    return float(np.mean(out))


def train_one(k):
    d = os.path.join(OUT, A.tag or f"dim{k}")
    os.makedirs(d, exist_ok=True)
    print()
    print("=" * 90)
    print(f"k = {k}   ({k} -> {A.hid} -> " + ("32" if A.layers==1 else f"{A.hid} -> 32") + f")   steps {A.steps}  lam {A.lam}")
    print("=" * 90)
    torch.manual_seed(0)
    C = torch.from_numpy(ZZ[:, :k].copy())
    G = Gen(k, A.hid, A.layers)
    D = Dis(k)
    optG = torch.optim.Adam(G.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    rng = np.random.default_rng(0)
    t0 = time.time()
    hist = []
    for step in range(1, A.steps + 1):
        bi = tr[rng.integers(0, len(tr), A.batch)]
        cb = C[bi]
        with torch.no_grad():
            f = G(cb).clamp(0, 15).round().long()
        rl = D(RIMG[bi], cb)
        fl = D(torch.from_numpy(layout.image_from_left_half(f.numpy())).long(), cb)
        loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) + \
                 F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
        optD.zero_grad(); loss_d.backward(); optD.step()

        raw = G(cb)
        pred = raw.clamp(0, 15)
        f = pred.round().long()
        fl = D(torch.from_numpy(layout.image_from_left_half(f.numpy())).long(), cb)
        loss_g = A.adv * F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))
        loss_g = loss_g + F.mse_loss(pred, TGT[bi]) + A.lam * tv32(raw)
        # CONTRAST TERM (training-stage fix for washed-out faces)
        # Plain MSE has its optimum at the conditional MEAN, so when the input does not pin
        # down how contrasty the face should be, the safest prediction is an average-contrast
        # face. Measured on the true coordinates, dim4 had a contrast slope of 0.684 -- it
        # produced only 68% of the contrast the face should have. This term penalises that
        # shrinkage directly: match the per-sample spread of the output to the target's.
        if A.contrast_weight > 0:
            sp = pred.std(1)
            st_ = TGT[bi].std(1)
            loss_g = loss_g + A.contrast_weight * ((sp - st_) ** 2).mean()
        optG.zero_grad(); loss_g.backward(); optG.step()

        if step % A.eval_every == 0 or step == A.steps:
            with torch.no_grad():
                bv = va[:1000]
                out = G(C[bv]).clamp(0, 15)
                dec = out.round().long()
                acc = float((dec == RTRUE[bv]).float().mean())
                ag = speckle(layout.image_from_left_half(dec.numpy()))
            rec = {"step": step, "acc": round(acc, 4), "agree": round(ag, 4),
                   "sec": round(time.time() - t0, 1)}
            hist.append(rec)
            print(f"  step {step:5d}  acc {acc:.4f}  agree {ag*100:5.1f}%  "
                  f"({rec['sec']:.0f}s)", flush=True)
    params = sum(p.numel() for p in G.parameters())
    macs = (k * A.hid + A.hid * 32) if A.layers == 1 else (k * A.hid + A.hid * A.hid + A.hid * 32)
    torch.save({"G": G.state_dict(), "k": k,
                "args": {"steps": A.steps, "hid": A.hid, "layers": A.layers, "lam": A.lam, "adv": A.adv,
                         "mode": "dimsweep", "k": k}},
               os.path.join(d, "final.pt"))
    json.dump({"k": k, "params": params, "multiplies": macs, "history": hist},
              open(os.path.join(d, "summary.json"), "w"), indent=2)
    return {"k": k, "params": params, "multiplies": macs,
            "acc": hist[-1]["acc"], "agree": hist[-1]["agree"]}


rows = []
for k in [int(s) for s in A.dims.split(",") if s.strip()]:
    rows.append(train_one(k))

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"  {'k':>3} {'multiplies':>11} {'accuracy':>9} {'agreement':>10}")
for r in rows:
    print(f"  {r['k']:3d} {r['multiplies']:11,d} {r['acc']:9.4f} {r['agree']*100:9.1f}%")
print(f"  reference PCA-22 (s32_tv5): 79,360 multiplies, acc 0.7719, agree 72.8%")
# each k is normally trained in its OWN process -- they run in parallel across the 32 cores,
# so the ladder summary must not be one shared file or the processes clobber each other
tag = "_".join(str(r["k"]) for r in rows)
json.dump(rows, open(os.path.join(OUT, f"ladder_{tag}.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, f'ladder_{tag}.json')}")
print("DIMSWEEP_DONE")
