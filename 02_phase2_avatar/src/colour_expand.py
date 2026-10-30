#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Expanded training and a diagnosis of where the adversarial term helps or breaks.

TWO QUESTIONS
    1  does simply training longer / wider raise colour yield?
    2  does the adversarial term actually work, and where does it fail?

GAN FAILURE MODES THAT ARE MEASURED, NOT ASSUMED
    D accuracy        if the discriminator separates real from generated with ease, the generator
                      is nowhere near the data and the gradient it receives is uninformative
    mode collapse     the number of distinct faces among 1000 samples. Collapse shows up here
                      long before it is visible in the loss
    metric drift      row spread, eye contrast, colour count and edge at each checkpoint, so a
                      model that improves one by destroying another is caught

A NOTE ON WHAT "REAL" MEANS HERE
The discriminator sees real avatars from the dataset and generated faces. Both are 32 colours.
The generated sample is the EXPECTED colour (softmax-weighted), not the argmax, so gradients flow;
that makes the fake distribution smoother than the real one, which is itself a reason the
discriminator can win too easily. This is reported, not hidden.
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
OUT = os.path.join(ROOT, "results", "colour", "expand")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114], dtype=np.float32)
LUMT = torch.tensor(LUM)
ROW = torch.tensor([p % 8 for p in range(32)])
EYE = torch.tensor([c * 8 + r for r in (3, 4) for c in (1, 2)])
SKIN = torch.tensor([c * 8 + r for r in (5, 6) for c in range(4)])

A = argparse.ArgumentParser()
A.add_argument("--tag", default="run")
A.add_argument("--steps", type=int, default=48000)
A.add_argument("--hid", type=int, default=128)
A.add_argument("--adv", type=float, default=0.0)
A.add_argument("--seed", type=int, default=0)
A.add_argument("--log-every", type=int, default=4000)
A = A.parse_args()

z = np.load(os.path.join(ROOT, "results", "colour", "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb96)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr = sp["train"]
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
disc = None
if A.adv > 0:
    disc = nn.Sequential(nn.Linear(96, 256), nn.LeakyReLU(0.2), nn.Linear(256, 128),
                         nn.LeakyReLU(0.2), nn.Linear(128, 1))
    dopt = torch.optim.Adam(disc.parameters(), lr=3e-4)

tr_t = torch.tensor(tr, dtype=torch.long)
rng = np.random.default_rng(A.seed)
PROBE = np.clip(np.random.default_rng(999).standard_normal((2000, 8)), -3.5, 3.5).astype(np.float32)


def evaluate():
    net.eval()
    with torch.no_grad():
        lg = net(torch.tensor(PROBE)).view(-1, 32, K)
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
    edge = float(np.percentile(np.concatenate([dh.reshape(len(a), -1), dv.reshape(len(a), -1)], 1), 90, axis=1).mean())
    ncol = float(np.mean([len({tuple(np.rint(c).astype(int)) for c in a[i]}) for i in range(len(a))]))
    # mode collapse: distinct faces and the mean pairwise distance between them
    keys = {tuple(np.rint(f).astype(int)) for f in faces}
    uniq = len(keys) / len(faces)
    return {"row_spread": float(rows.std(1).mean()),
            "row_jump": float(np.abs(np.diff(rows, axis=1)).max(1).mean()),
            "eye": float(eye.mean()), "edge": edge, "ncol": ncol,
            "distinct": float(uniq)}


print(f"=== {A.tag}: steps={A.steps} hid={A.hid} adv={A.adv} seed={A.seed} ===")
t0 = time.time()
log = []
tr_t_np = tr
for step in range(A.steps):
    bi = tr_t[torch.from_numpy(rng.integers(0, len(tr_t), 256))]
    lg = net(CT[bi]).view(-1, 32, K)
    ce = F.cross_entropy(lg.reshape(-1, K), YIDX[bi].reshape(-1))
    loss = ce
    d_acc = float("nan")
    gadv = 0.0
    if disc is not None:
        exp_c = (F.softmax(lg, -1) @ PAL_T).reshape(-1, 96)
        real = POS_T[bi].reshape(-1, 96)
        with torch.no_grad():
            d_acc = float(((disc(real) > 0).float().mean() +
                           (disc(exp_c) < 0).float().mean()) / 2)
        d_real = disc(real); d_fake = disc(exp_c.detach())
        d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) \
               + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        dopt.zero_grad(); d_loss.backward(); dopt.step()
        adv = F.binary_cross_entropy_with_logits(disc(exp_c), torch.ones(len(exp_c), 1))
        gadv = float(adv)
        loss = ce + A.adv * adv
    opt.zero_grad(); loss.backward(); opt.step()

    if (step + 1) % A.log_every == 0:
        ev = evaluate()
        rec = {"step": step + 1, "ce": float(ce), "adv": gadv, "d_acc": d_acc, **ev}
        log.append(rec)
        print(f"  {step+1:6d}  ce {ce:.3f}  " +
              (f"adv {gadv:.3f}  D_acc {d_acc:.3f}  " if disc is not None else "") +
              f"rowvar {ev['row_spread']:.2f}  rowjump {ev['row_jump']:.2f}  "
              f"eye {ev['eye']:.2f}  edge {ev['edge']:.2f}  ncol {ev['ncol']:.2f}  "
              f"distinct {ev['distinct']:.3f}")

ev = evaluate()
res = {"tag": A.tag, "steps": A.steps, "hid": A.hid, "adv": A.adv, "seed": A.seed,
       "final": ev, "log": log, "seconds": time.time() - t0}
json.dump(res, open(os.path.join(OUT, f"{A.tag}.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
torch.save({"G": net.state_dict(), "palette": PAL.tolist(), "args": vars(A)},
           os.path.join(OUT, f"{A.tag}.pt"))
print(f"  final: {ev}")
print(f"  {time.time()-t0:.0f}s  wrote {A.tag}.json / .pt")
print("EXPAND_DONE")
