#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Colour stage-2 training: the experiment matrix from docs/COLOUR_EXPERIMENT_DESIGN.md

    E1  stage2( gray only )                    proof-of-failure control
    E2  stage2( gray + colour PCA 4 )
    E3  stage2( gray + colour PCA 8 )          the proposed design
    E4  stage2( gray + colour PCA 16 )
    E5  joint( 4 gray PCs + colour PCA 8 )     does one step beat two?
    E6  colour PCA 8 only                      no gray input

THE GRAY INPUT IS STAGE 1'S PREDICTION, NOT THE TRUE GRAY
At inference stage 2 only ever sees what stage 1 produced, so it is trained on that too. Using
the true gray would train it on inputs it can never receive, and the error of stage 1 (about
40% exact match on levels) is large enough that the mismatch would hurt.

THE TARGET IS THE MIRRORED 32-POSITION COLOUR (96 values)
Colour turned out to be 98.7% mirror-symmetric, even more than luminance, so the compact
target is faithful and halves the output layer.
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

A = argparse.ArgumentParser()
A.add_argument("--steps", type=int, default=12000)
A.add_argument("--batch", type=int, default=256)
A.add_argument("--hid", type=int, default=128)
A.add_argument("--lr", type=float, default=3e-4)
A.add_argument("--threads", type=int, default=8)
A.add_argument("--only", type=str, default="")
A = A.parse_args()
torch.set_num_threads(A.threads)

print("=" * 100)
print(f"COLOUR STAGE-2 TRAINING   steps={A.steps} hid={A.hid} steps-consistent across variants")
print("=" * 100)

# ---------------------------------------------------------------- data
z = np.load(os.path.join(OUT, "cache.npz"))
gray32 = z["gray32"].astype(np.float32)          # (N,32) 0..15 true
pc4 = z["pc4"]                                   # (N,4)
rgb96 = z["rgb96"].astype(np.float32)            # (N,96) 0..15 true, mirrored 32 positions
cpc = z["cpc"]                                   # (N,24)
N = len(gray32)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
print(f"rows {N:,}   train {len(tr):,}   val {len(va):,}")

# ---------------------------------------------------------------- stage 1, frozen
gj = json.load(open(os.path.join(WEB, "pca22_dim4x_model.json"), encoding="utf-8"))
layers = gj["gen"]["layers"]
seq = []
for i, L in enumerate(layers):
    W = torch.tensor(L["W"], dtype=torch.float32)
    b = torch.tensor(L["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq.append(lin)
    if i != len(layers) - 1:
        seq.append(nn.ReLU())
G1 = nn.Sequential(*seq).eval()
for p in G1.parameters():
    p.requires_grad_(False)


def stage1_gray(pcs):
    """The full deployed path: network -> clip -> per-face min-max stretch -> round -> /15."""
    with torch.no_grad():
        v = torch.clamp(G1(torch.tensor(pcs, dtype=torch.float32)), 0, 15)
        lo = v.min(1, keepdim=True)[0]; hi = v.max(1, keepdim=True)[0]
        v = torch.clamp((v - lo) / (hi - lo).clamp_min(1e-6) * 15, 0, 15)
        v = torch.round(v)
    return v.numpy()


def stretch_true(g):
    lo = g.min(1, keepdims=True); hi = g.max(1, keepdims=True)
    return np.clip((g - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)


G_pred = stage1_gray(pc4)                        # (N,32) 0..15 integers, what stage 2 receives
exact = float((G_pred == stretch_true(gray32)).mean())
print(f"stage 1 reproduces the stretched true gray exactly on {100*exact:.1f}% of positions")
print("  (stage 2 is trained on stage 1's output, so this mismatch is part of the task)")

# the target gray is the same stretched true gray, used for the consistency metric
g_true_s = stretch_true(gray32)
G_IN = (G_pred / 15.0).astype(np.float32)        # (N,32) normalised input


# ---------------------------------------------------------------- variants
def build_inputs(name, k):
    if name == "A":
        return np.concatenate([G_IN, cpc[:, :k]], 1) if k else G_IN.copy()
    if name == "B":                              # joint: gray PCs + colour PCs
        return np.concatenate([pc4, cpc[:, :k]], 1)
    if name == "C":                              # colour only
        return cpc[:, :k].copy()
    raise ValueError(name)


VARIANTS = [
    ("E1", "A", 0, "灰阶 only（证伪对照）"),
    ("E2", "A", 4, "灰阶 + 颜色4"),
    ("E3", "A", 8, "灰阶 + 颜色8（主方案）"),
    ("E4", "A", 16, "灰阶 + 颜色16"),
    ("E5", "B", 8, "联合：4灰度PC + 颜色8"),
    ("E6", "C", 8, "颜色8 only（无灰度）"),
    ("E8", "C", 16, "颜色16 only（生成用对照）"),
]
if A.only:
    keep = set(A.only.split(","))
    VARIANTS = [v for v in VARIANTS if v[0] in keep]

Y = torch.tensor(rgb96 / 15.0)
tr_t = torch.tensor(tr, dtype=torch.long)
va_t = torch.tensor(va, dtype=torch.long)


def train_one(tag, scheme, k, note):
    Xin = build_inputs(scheme, k).astype(np.float32)
    d_in = Xin.shape[1]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(d_in, A.hid), nn.ReLU(),
        nn.Linear(A.hid, A.hid), nn.ReLU(),
        nn.Linear(A.hid, 96))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr)
    Xt = torch.tensor(Xin)
    t0 = time.time()
    for step in range(A.steps):
        idx = tr_t[torch.from_numpy(rng.integers(0, len(tr_t), A.batch))]
        opt.zero_grad()
        out = net(Xt[idx])
        loss = nn.functional.mse_loss(out, Y[idx])
        loss.backward()
        opt.step()
        if (step + 1) % 4000 == 0:
            with torch.no_grad():
                vl = nn.functional.mse_loss(net(Xt[va_t]), Y[va_t]).item()
            print(f"    {tag} step {step+1:6d}  train {loss.item():.5f}  val {vl:.5f}  "
                  f"{time.time()-t0:.0f}s")
    net.eval()
    with torch.no_grad():
        pred = net(Xt[va_t]).numpy()
    return net, Xin, pred, time.time() - t0


def metrics(pred_va, idx_va):
    """Everything from the design's section 5, computed on the validation rows."""
    p = np.clip(pred_va, 0, 1) * 15.0
    t = rgb96[idx_va]
    mae = float(np.abs(p - t).mean())
    pr = p.reshape(-1, 32, 3); trr = t.reshape(-1, 32, 3)
    # HSV from 0..15 channels
    def hsv(a):
        mx = a.max(-1); mn = a.min(-1)
        v = mx / 15.0
        s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        d = np.maximum(mx - mn, 1e-9)
        r_, g_, b_ = a[..., 0], a[..., 1], a[..., 2]
        h = np.zeros_like(mx)
        m1 = mx == r_; h[m1] = ((g_ - b_)[m1] / d[m1]) % 6
        m2 = (mx == g_) & ~m1; h[m2] = ((b_ - r_)[m2] / d[m2]) + 2
        m3 = (mx == b_) & ~m1 & ~m2; h[m3] = ((r_ - g_)[m3] / d[m3]) + 4
        return h / 6.0, s, v
    hp, sp_, vp = hsv(pr)
    ht, st, vt = hsv(trr)
    # circular hue distance, weighted by how saturated both are (hue is meaningless when grey)
    dh = np.abs(hp - ht); dh = np.minimum(dh, 1 - dh)
    w = (sp_ > 0.15) & (st > 0.15)
    hue_err = float(dh[w].mean()) if w.sum() else float("nan")
    hue_all = float(dh.mean())
    sat_err = float(np.abs(sp_ - st).mean())
    # gray consistency: luminance of the predicted colour vs what stage 1 gave
    lum_p = 0.299 * pr[..., 0] + 0.587 * pr[..., 1] + 0.114 * pr[..., 2]
    lum_t = 0.299 * trr[..., 0] + 0.587 * trr[..., 1] + 0.114 * trr[..., 2]
    g1 = G_pred[idx_va]
    gray_consistency = float(np.abs(lum_p - g1).mean())
    return {"rgb_mae": mae, "hue_err": hue_err, "hue_err_all": hue_all,
            "sat_err": sat_err, "gray_consistency": gray_consistency,
            "hue_weighted_share": float(w.mean())}


results = {}
for tag, scheme, k, note in VARIANTS:
    print()
    print(f"  {tag}: {note}   input dim = {build_inputs(scheme, k).shape[1]}")
    net, Xin, pred, secs = train_one(tag, scheme, k, note)
    m = metrics(pred, va)
    m.update({"note": note, "scheme": scheme, "k": k, "d_in": int(Xin.shape[1]),
              "seconds": round(secs, 1),
              "params": int(sum(p.numel() for p in net.parameters()))})
    results[tag] = m
    print(f"    -> RGB MAE {m['rgb_mae']:.3f}   hue err {m['hue_err']:.3f}   "
          f"sat err {m['sat_err']:.3f}   gray cons {m['gray_consistency']:.2f}")
    torch.save({"state": net.state_dict(), "d_in": int(Xin.shape[1]), "k": k,
                "scheme": scheme, "hid": A.hid},
               os.path.join(OUT, f"stage2_{tag}.pt"))

# one file per run, explicit utf-8: six parallel trainers used to overwrite a shared file and
# json.dump without encoding wrote GBK on Windows, which is not valid UTF-8
_tag = (A.only or "all").replace(",", "_")
json.dump(results, open(os.path.join(OUT, f"stage2_results_{_tag}.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"  {'id':3s} {'variant':28s} {'in':>3s} {'params':>7s} {'RGB MAE':>8s} "
      f"{'色相误差':>9s} {'饱和误差':>9s} {'灰度一致':>9s}")
print("  " + "-" * 82)
for tag, m in results.items():
    print(f"  {tag:3s} {m['note']:28s} {m['d_in']:3d} {m['params']:7d} "
          f"{m['rgb_mae']:8.3f} {m['hue_err']:9.3f} {m['sat_err']:9.3f} "
          f"{m['gray_consistency']:9.2f}")
json.dump({"quant_floor": 0.2049}, open(os.path.join(OUT, "floor.json"), "w"), indent=2)
print()
print("  4-bit quantisation floor on RGB MAE = 0.205; no model can go below it.")
print("COLOUR_STAGE2_DONE")
