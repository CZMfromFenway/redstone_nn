#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Colour generator with the advisor-critic recipe that sharpened the grayscale line.

WHY THIS SHOULD HELP
The colour model so far is pure MSE, whose optimum is the conditional mean -- visibly mushy at
24x (9.6 luminance levels against the grayscale generator's 12.7). The grayscale generator got
its crispness from MSE + an adversarial term + total variation, and that recipe is reused here.

THE USER'S STANDARDS, AS LOSS TERMS
Their three criteria, measured on their own 204 labels, separate their best faces from 中/差:

    main-colour separation      AUC 0.859    best 13.50   bad 10.10
    edge sharpness (p90)        AUC 0.838    best 14.27   bad  9.78
    fewer distinct colours      AUC 0.718    best 20.4    bad 23.3
    lower mean saturation       AUC 0.65     best 0.432   bad  0.544
    a lot of near-grey          AUC 0.731    best 0.285   bad  0.129

Only two of those are cheap differentiable terms:
    sat_gap   || E[saturation] - E[saturation of the target] ||   -- keeps colour from inflating
    sep_gap   || E[pairwise colour distance] - same on target ||  -- keeps the three colours apart
The rest (edge sharpness, distinct-colour count) come from the adversarial + TV pair and are
measured after training, not optimised directly.

VARIANTS
    C2   adv + MSE + TV                    the grayscale recipe transplanted
    C3   C2 + saturation and separation matching
    C4   C2 with a stronger TV weight       for smoothness, since TV still helped before

Every variant is judged against the REAL data's statistics, not against another model.
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
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour")
OUTG = os.path.join(OUT, "gan")
os.makedirs(OUTG, exist_ok=True)

A = argparse.ArgumentParser()
A.add_argument("--steps", type=int, default=8000)
A.add_argument("--batch", type=int, default=256)
A.add_argument("--hid", type=int, default=128)
A.add_argument("--lrG", type=float, default=3e-4)
A.add_argument("--lrD", type=float, default=2e-4)
A.add_argument("--tv", type=float, default=5.0)
A.add_argument("--adv", type=float, default=1.0)
A.add_argument("--stats", type=float, default=0.0, help="weight on saturation/separation matching")
A.add_argument("--tag", default="C2")
A.add_argument("--threads", type=int, default=8)
A = A.parse_args()
torch.set_num_threads(A.threads)

z = np.load(os.path.join(OUT, "cache.npz"))
rgb96 = z["rgb96"].astype(np.float32)          # (N,96) 0..15
cpc = z["cpc"]                                 # (N,24)
N = len(rgb96)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
print("=" * 100)
print(f"COLOUR GAN  tag={A.tag}  steps={A.steps}  tv={A.tv} adv={A.adv} stats={A.stats}")
print("=" * 100)
print(f"rows {N:,}  train {len(tr):,}  val {len(va):,}")

LUMW = torch.tensor([0.299, 0.587, 0.114])


def to_img(v):
    """(B,96) 0..15 -> (B,3,8,8), mirrored within each row."""
    a = v.view(-1, 32, 3)
    img = torch.zeros(a.shape[0], 3, 8, 8, dtype=v.dtype, device=v.device)
    rows = torch.tensor([p % 8 for p in range(32)])
    cols = torch.tensor([p // 8 for p in range(32)])
    mcols = 7 - cols
    img[:, :, rows, cols] = a.permute(0, 2, 1)
    img[:, :, rows, mcols] = a.permute(0, 2, 1)
    return img


def sat_of(x):
    """x: (B, ... ,3) 0..15 -> saturation in 0..1."""
    mx = x.max(-1)[0]; mn = x.min(-1)[0]
    return torch.where(mx > 1e-6, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))


def stats_of(v):
    """the two quantities the user's standard cares about, as differentiable scalars."""
    a = v.view(-1, 32, 3)
    s = sat_of(a).mean()
    d = torch.cdist(a, a)
    iu = torch.triu_indices(32, 32, offset=1)
    sep = d[:, iu[0], iu[1]].mean()
    return s, sep


def tv32(raw):
    """total variation on the 8x8x3 image, the same smoothness term as the grayscale recipe."""
    img = to_img(raw.clamp(0, 15))
    dh = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    dv = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    return dh + dv


class G(nn.Module):
    def __init__(self, k=8, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 96))

    def forward(self, c):
        return self.net(c).clamp(0, 15)


class D(nn.Module):
    def __init__(self, k=8, c=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, c, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 2, c * 2, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 2 * 16 + k, 64),
                                 nn.LeakyReLU(0.2, True), nn.Linear(64, 1))

    def forward(self, img, c):
        return self.head(torch.cat([self.conv(img).flatten(1), c], 1))


T = torch.tensor(rgb96)          # 0..15 units, as in the grayscale recipe
CT = torch.tensor(cpc[:, :8])
tr_t = torch.tensor(tr, dtype=torch.long)
va_t = torch.tensor(va, dtype=torch.long)
with torch.no_grad():
    tstats = stats_of(T[va_t] * 15.0)
print(f"  target saturation {tstats[0]:.3f}   target separation {tstats[1]:.3f}")

torch.manual_seed(0)
Gm = G(8, A.hid)
Dm = D(8)
optG = torch.optim.Adam(Gm.parameters(), lr=A.lrG, betas=(0.5, 0.999))
optD = torch.optim.Adam(Dm.parameters(), lr=A.lrD, betas=(0.5, 0.999))
rng = np.random.default_rng(0)
t0 = time.time()
for step in range(A.steps):
    bi = tr_t[torch.from_numpy(rng.integers(0, len(tr_t), A.batch))]
    cb = CT[bi]
    # ---- discriminator
    with torch.no_grad():
        f = Gm(cb)
    rl = Dm(to_img(T[bi] * 15.0).float(), cb)
    fl = Dm(to_img(f).float(), cb)
    loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) + \
             F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
    optD.zero_grad(); loss_d.backward(); optD.step()
    # ---- generator
    out = Gm(cb)
    fl = Dm(to_img(out).float(), cb)
    loss_g = A.adv * F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))
    loss_g = loss_g + F.mse_loss(out, T[bi])
    loss_g = loss_g + A.tv * tv32(out) / 15.0   # TV stays on 0..1
    if A.stats > 0:
        s, sep = stats_of(out)
        loss_g = loss_g + A.stats * ((s - tstats[0]).abs() + (sep - tstats[1]).abs() / 15.0)
    optG.zero_grad(); loss_g.backward(); optG.step()
    if (step + 1) % 2000 == 0:
        with torch.no_grad():
            vl = F.mse_loss(Gm(CT[va_t]), T[va_t]).item()
            s, sep = stats_of(Gm(CT[va_t]))
        print(f"    {A.tag} step {step+1:6d}  D {loss_d.item():.3f}  G {loss_g.item():.4f}  "
              f"val {vl:.5f}  sat {s:.3f}  sep {sep:.2f}  {time.time()-t0:.0f}s")

Gm.eval()
with torch.no_grad():
    pv = Gm(CT[va_t]).numpy()
torch.save({"G": Gm.state_dict(), "k": 8, "hid": A.hid,
            "args": {"tag": A.tag, "steps": A.steps, "tv": A.tv, "adv": A.adv,
                     "stats": A.stats, "hid": A.hid}},
           os.path.join(OUTG, f"{A.tag}.pt"))


def report(p):
    p = np.clip(p, 0, 15)
    t = rgb96[va_t]
    mae = float(np.abs(p - t).mean())
    a = p.reshape(-1, 32, 3); b = t.reshape(-1, 32, 3)
    def sat(x):
        mx = x.max(-1); mn = x.min(-1)
        return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    def sep(x):
        d = np.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=-1)
        iu = np.triu_indices(32, 1)
        return float(d[:, iu[0], iu[1]].mean())
    img = np.zeros((len(p), 8, 8, 3))
    img[:, [q % 8 for q in range(32)], [q // 8 for q in range(32)]] = a
    img[:, [q % 8 for q in range(32)], [7 - q // 8 for q in range(32)]] = a
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    ed = np.concatenate([dh.ravel(), dv.ravel()])
    return {"rgb_mae": mae, "sat": float(sat(a).mean()), "sep": sep(a),
            "edge_p90": float(np.percentile(ed, 90)), "edge_mean": float(ed.mean())}


rep = report(pv)
print()
print("=" * 100)
print(f"RESULT  {A.tag}")
print("=" * 100)
print(f"  RGB MAE {rep['rgb_mae']:.3f}   sat {rep['sat']:.3f} (real 0.319)   "
      f"sep {rep['sep']:.2f} (real {'?'})")
print(f"  edge p90 {rep['edge_p90']:.2f}   edge mean {rep['edge_mean']:.2f}")
json.dump(rep, open(os.path.join(OUTG, f"{A.tag}_metrics.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("COLOUR_GAN_DONE")
