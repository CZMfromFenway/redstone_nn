#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Recompute the colour metrics from the saved checkpoints and write ONE utf-8 results file.

WHY THIS EXISTS
Six trainer processes each wrote results/stage2_results.json on finishing. Beyond the race
(last writer wins, so only E1 survived), the dump had no encoding= argument, so on Windows the
Chinese note fields were encoded as GBK and the file is not valid UTF-8 -- the same mistake
that already corrupted the dimension-sweep manifest once.

The weights themselves are fine (one file per variant, no collision), so the metrics are
recomputed here with one writer and explicit encoding. The trainers have been changed to write
per-variant files for the same reason.
"""
import json
import math
import os
import sys
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

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

z = np.load(os.path.join(OUT, "cache.npz"))
gray32 = z["gray32"].astype(np.float32)
pc4 = z["pc4"]
rgb96 = z["rgb96"].astype(np.float32)
cpc = z["cpc"]
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]

# stage 1, frozen, same as the trainer
gj = json.load(open(os.path.join(WEB, "pca22_dim4x_model.json"), encoding="utf-8"))
seq = []
L_ = gj["gen"]["layers"]
for i, L in enumerate(L_):
    W = torch.tensor(L["W"], dtype=torch.float32)
    b = torch.tensor(L["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq.append(lin)
    if i != len(L_) - 1:
        seq.append(nn.ReLU())
G1 = nn.Sequential(*seq).eval()
for p in G1.parameters():
    p.requires_grad_(False)


def gray_path(pcs):
    with torch.no_grad():
        v = torch.clamp(G1(torch.tensor(pcs, dtype=torch.float32)), 0, 15)
        lo = v.min(1, keepdim=True)[0]; hi = v.max(1, keepdim=True)[0]
        v = torch.clamp((v - lo) / (hi - lo).clamp_min(1e-6) * 15, 0, 15)
        return torch.round(v).numpy()


G_pred = gray_path(pc4)
G_IN = (G_pred / 15.0).astype(np.float32)
va_t = va


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


def metrics(p, t, g1):
    mae = float(np.abs(p - t).mean())
    pr = p.reshape(-1, 32, 3); trr = t.reshape(-1, 32, 3)
    hp, sp_, vp = hsv(pr)
    ht, st, vt = hsv(trr)
    dh = np.abs(hp - ht); dh = np.minimum(dh, 1 - dh)
    w = (sp_ > 0.15) & (st > 0.15)
    hue = float(dh[w].mean()) if w.sum() else float("nan")
    sat = float(np.abs(sp_ - st).mean())
    lum_p = 0.299 * pr[..., 0] + 0.587 * pr[..., 1] + 0.114 * pr[..., 2]
    gcons = float(np.abs(lum_p - g1).mean())
    # per-channel MAE, to see whether R, G or B is the hard one
    maes = [float(np.abs(pr[..., c] - trr[..., c]).mean()) for c in range(3)]
    return {"rgb_mae": mae, "hue_err": hue, "hue_err_all": float(dh.mean()),
            "sat_err": sat, "gray_consistency": gcons,
            "mae_rgb": maes, "hue_weighted_share": float(w.mean())}


NOTES = {"E1": "灰阶 only (证伪对照)", "E2": "灰阶 + 颜色4", "E3": "灰阶 + 颜色8 (主方案)",
         "E4": "灰阶 + 颜色16", "E5": "联合: 4灰度PC + 颜色8", "E6": "颜色8 only (无灰度)"}

res = {}
for tag in ("E1", "E2", "E3", "E4", "E5", "E6"):
    fp = os.path.join(OUT, f"stage2_{tag}.pt")
    if not os.path.exists(fp):
        continue
    m = torch.load(fp, weights_only=False)
    sch, k, hid, d_in = m["scheme"], m["k"], m["hid"], m["d_in"]
    if sch == "A":
        x = G_IN[va_t] if k == 0 else np.concatenate([G_IN[va_t], cpc[va_t][:, :k]], 1)
    elif sch == "B":
        x = np.concatenate([pc4[va_t], cpc[va_t][:, :k]], 1)
    else:
        x = cpc[va_t][:, :k]
    net = nn.Sequential(nn.Linear(d_in, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                        nn.Linear(hid, 96))
    net.load_state_dict(m["state"]); net.eval()
    with torch.no_grad():
        p = np.clip(net(torch.tensor(x.astype(np.float32))).numpy() * 15.0, 0, 15)
    met = metrics(p, rgb96[va_t], G_pred[va_t])
    met.update({"note": NOTES[tag], "scheme": sch, "k": k, "d_in": int(d_in),
                "hid": int(hid), "params": int(sum(q.numel() for q in net.parameters()))})
    res[tag] = met

# one writer, explicit encoding
p = os.path.join(OUT, "stage2_results.json")
json.dump(res, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("=" * 100)
print("COLOUR STAGE 2 -- RESULTS")
print("=" * 100)
print(f"  {'id':3s} {'variant':24s} {'in':>3s} {'params':>7s} {'RGB MAE':>8s} "
      f"{'色相':>6s} {'饱和':>6s} {'灰度一致':>8s}  R/G/B MAE")
print("  " + "-" * 96)
for tag in res:
    m = res[tag]
    rgb = "/".join(f"{v:.2f}" for v in m["mae_rgb"])
    print(f"  {tag:3s} {m['note']:24s} {m['d_in']:3d} {m['params']:7d} {m['rgb_mae']:8.3f} "
          f"{m['hue_err']:6.3f} {m['sat_err']:6.3f} {m['gray_consistency']:8.2f}  {rgb}")

print()
print("=" * 100)
print("THE FOUR PRE-REGISTERED CRITERIA")
print("=" * 100)
e1, e3, e5 = res.get("E1"), res.get("E3"), res.get("E5")
if e1 and e3:
    drop = 1 - e3["hue_err"] / max(e1["hue_err"], 1e-9)
    print(f"  1. E3 色相误差 < 0.10            {e3['hue_err']:.3f}      "
          f"{'PASS' if e3['hue_err'] < 0.10 else 'FAIL'}")
    print(f"  2. E3 比 E1 色相改善 > 40%       {100*drop:5.1f}%     "
          f"{'PASS' if drop > 0.40 else 'FAIL'}")
    print(f"  4. E3 灰度一致 < 1.5 级          {e3['gray_consistency']:.2f}      "
          f"{'PASS' if e3['gray_consistency'] < 1.5 else 'FAIL'}")
if e5 and e3:
    gap = abs(e3["hue_err"] - e5["hue_err"]) / max(min(e3["hue_err"], e5["hue_err"]), 1e-9)
    print(f"  3. E3 与 E5 差距 < 20%           {100*gap:5.1f}%     "
          f"{'PASS' if gap < 0.20 else 'FAIL'}")
print()
print(f"  RGB 4-bit quantisation floor = 0.205 (no model can beat it)")
print(f"  best RGB MAE = {min(m['rgb_mae'] for m in res.values()):.3f}")
print(f"wrote {p}")
print("RECOMPUTE_DONE")
