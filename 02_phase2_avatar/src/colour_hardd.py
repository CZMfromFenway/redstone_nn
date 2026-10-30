#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Hard discriminator input: the straight-through estimator the user proposed.

THE PROBLEM IT FIXES
The previous adversarial run fed the discriminator the EXPECTED colour, sum_k p_k c_k, which is a
blurred average. Real avatars are always one hard palette colour per position, so the
discriminator could separate real from fake on blurriness alone. It sat at 0.85 accuracy for the
whole run and never approached the 0.5 where a GAN is actually learning.

That also explains what the term actually did: it pushed the soft output toward saturation, which
is why edge and colour count rose while row structure and eye contrast did not move.

THE FIX
Forward pass uses only the mode; the backward pass keeps the softmax gradient. Written out:

    q = argmax(p)                       a one-hot vector, no gradient
    fake = one_hot(q) + p - p.detach()  value = one_hot, gradient = dp
    colour = fake @ palette

This is the user's "raise to a power and renormalise, keep only the mode" with a gradient path
bolted on. The optional temperature T sharpens p before the argmax, so T -> 0 is a plain step.

WHAT TO WATCH
If D accuracy falls from 0.85 toward 0.5-0.7, the discriminator has lost its shortcut and is
being forced to look at content. If it stays at 0.85, the shortcut was not the only problem.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour", "hardd")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114], dtype=np.float32)

A = argparse.ArgumentParser()
A.add_argument("--tag", default="run")
A.add_argument("--steps", type=int, default=36000)
A.add_argument("--hid", type=int, default=128)
A.add_argument("--adv", type=float, default=0.3)
A.add_argument("--temp", type=float, default=0.1)
A.add_argument("--dhid", type=int, default=256)
A.add_argument("--dlr", type=float, default=3e-4)
A.add_argument("--spectral", type=int, default=0)
A.add_argument("--seed", type=int, default=0)
A.add_argument("--log-every", type=int, default=4000)
A = A.parse_args()

z = np.load(os.path.join(ROOT, "results", "colour", "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb96)
tr = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["train"]
POS = rgb96.reshape(N, 32, 3)
CT = torch.tensor(cpc[:, :8])
POS_T = torch.tensor(POS.astype(np.float32))

palj = json.load(open(os.path.join(WEB, "colour_pal8_model.json"), encoding="utf-8"))
PAL = np.array(palj["palette"], dtype=np.float64)
K = len(PAL)
PAL_T = torch.tensor(PAL.astype(np.float32))
dd = ((POS[:, :, None, :] - PAL[None, None, :, :]) ** 2).sum(-1)
YIDX = torch.tensor(dd.argmin(-1).reshape(N, 32), dtype=torch.long)

torch.manual_seed(A.seed)
net = nn.Sequential(nn.Linear(8, A.hid), nn.ReLU(), nn.Linear(A.hid, A.hid), nn.ReLU(),
                    nn.Linear(A.hid, 32 * K))
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

disc = nn.Sequential(nn.Linear(96, A.dhid), nn.LeakyReLU(0.2),
                     nn.Linear(A.dhid, A.dhid // 2), nn.LeakyReLU(0.2),
                     nn.Linear(A.dhid // 2, 1))
if A.spectral:
    disc = nn.Sequential(*[nn.utils.spectral_norm(m) if isinstance(m, nn.Linear) else m
                           for m in disc])
dopt = torch.optim.Adam(disc.parameters(), lr=A.dlr)


def hard_forward(logits):
    """One-hot in the forward pass, softmax gradient in the backward pass."""
    p = F.softmax(logits / A.temp, -1)
    idx = p.argmax(-1, keepdim=True)
    hard = torch.zeros_like(p).scatter_(-1, idx, 1.0)
    return hard + p - p.detach(), p


tr_t = torch.tensor(tr, dtype=torch.long)
rng = np.random.default_rng(A.seed)
PROBE = np.clip(np.random.default_rng(999).standard_normal((2000, 8)), -3.5, 3.5).astype(np.float32)


def evaluate():
    net.eval()
    with torch.no_grad():
        lg = net(torch.tensor(PROBE)).view(-1, 32, K)
        # measure collapse in the HARD output, since that is what gets rendered
        idx = lg.argmax(-1).numpy()
        faces = PAL[idx].reshape(len(PROBE), 96)
    net.train()
    a = np.clip(faces, 0, 15).reshape(len(PROBE), 32, 3)
    lu = a @ LUM
    rows = np.stack([lu[:, [c * 8 + r for c in range(4)]].mean(1) for r in range(8)], 1)
    eye = np.median(lu[:, [c * 8 + r for r in (5, 6) for c in range(4)]], 1) \
        - lu[:, [c * 8 + r for r in (3, 4) for c in (1, 2)]].min(1)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    edge = float(np.percentile(np.concatenate([dh.reshape(len(a), -1),
                                              dv.reshape(len(a), -1)], 1), 90, axis=1).mean())
    ncol = float(np.mean([len({tuple(np.rint(c).astype(int)) for c in a[i]})
                          for i in range(len(a))]))
    keys = {tuple(np.rint(f).astype(int)) for f in faces}
    return {"row_spread": float(rows.std(1).mean()),
            "row_jump": float(np.abs(np.diff(rows, axis=1)).max(1).mean()),
            "eye": float(eye.mean()), "edge": edge, "ncol": ncol,
            "distinct": float(len(keys) / len(faces))}


print(f"=== {A.tag}: steps={A.steps} adv={A.adv} T={A.temp} spectral={A.spectral} "
      f"dlr={A.dlr} ===")
t0 = time.time()
log = []
for step in range(A.steps):
    bi = tr_t[torch.from_numpy(rng.integers(0, len(tr_t), 256))]
    lg = net(CT[bi]).view(-1, 32, K)
    ce = F.cross_entropy(lg.reshape(-1, K), YIDX[bi].reshape(-1))

    hard, p = hard_forward(lg)
    exp_c = (p @ PAL_T).reshape(-1, 96)              # soft, only to show the gap
    fake = (hard @ PAL_T).reshape(-1, 96)            # hard, what the discriminator sees
    real = POS_T[bi].reshape(-1, 96)

    with torch.no_grad():
        d_acc = float((disc(real) > 0).float().mean())
        d_acc = 0.5 * (d_acc + float((disc(fake) < 0).float().mean()))
    d_real = disc(real); d_fake = disc(fake.detach())
    d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) \
           + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    dopt.zero_grad(); d_loss.backward(); dopt.step()

    adv = F.binary_cross_entropy_with_logits(disc(fake), torch.ones(len(fake), 1))
    loss = ce + A.adv * adv
    opt.zero_grad(); loss.backward(); opt.step()

    if (step + 1) % A.log_every == 0:
        ev = evaluate()
        rec = {"step": step + 1, "ce": float(ce), "adv": float(adv), "d_acc": d_acc, **ev}
        log.append(rec)
        hard_gap = float((fake - exp_c).abs().mean())
        print(f"  {step+1:6d}  ce {ce:.3f}  adv {float(adv):.3f}  D_acc {d_acc:.3f}  "
              f"软硬差 {hard_gap:.3f}  rowvar {ev['row_spread']:.2f}  eye {ev['eye']:.2f}  "
              f"edge {ev['edge']:.2f}  ncol {ev['ncol']:.2f}  distinct {ev['distinct']:.3f}")

ev = evaluate()
json.dump({"tag": A.tag, "steps": A.steps, "hid": A.hid, "adv": A.adv, "temp": A.temp,
           "spectral": A.spectral, "dlr": A.dlr, "seed": A.seed,
           "final": ev, "log": log, "seconds": time.time() - t0},
          open(os.path.join(OUT, f"{A.tag}.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
torch.save({"G": net.state_dict(), "palette": PAL.tolist(), "args": vars(A)},
           os.path.join(OUT, f"{A.tag}.pt"))
print(f"  final {ev}")
print(f"  {time.time()-t0:.0f}s")
print("HARDD_DONE")
