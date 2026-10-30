#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
The screening at lam=0.05 moved the speckle by almost nothing. Two hypotheses:

  H1 the penalty weight is simply too small -> sweep lam over orders of magnitude
  H2 the training/evaluation MISMATCH is the problem. Training uses a hard Gumbel
     sample, but the pictures are judged on the ARGMAX. Sampling adds speckle the
     generator was never told to avoid, so the argmax may be fine while the sample is
     noisy -- or vice versa. We therefore evaluate BOTH and also train with the argmax
     path so the objective matches what the user looks at.

H2 also suggests a stronger mechanism than any smoothness penalty: make the CE target
the model's OWN current decision fed back (a form of self-distillation towards a
confident, stable answer). That is not tested here; the lam sweep decides first.

Every configuration is a short pilot, same data, same seed, so the comparison is fair.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)

AP = argparse.ArgumentParser()
AP.add_argument("--steps", type=int, default=700)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--lams", default="0.05,0.5,2.0,8.0")
AP.add_argument("--penalty", default="tv_probs")
AP.add_argument("--decode", default="sample", choices=["sample", "mode"],
                help="which decoding the CE/adversarial sees during training")
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--tag", default="lam")
A = AP.parse_args()
torch.set_num_threads(A.threads)
OUT = os.path.join(ROOT, "results", "coherence", A.tag)
os.makedirs(OUT, exist_ok=True)

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
mask2 = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
         gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[mask2]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
COND = torch.tensor(((X - mu[0]) @ comps.T) / sd[0], dtype=torch.float32)
FULL_T = torch.from_numpy(full.astype(np.int64))
RTRUE = FULL_T[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, cond):
        return self.net(cond).view(-1, 32, NLEV)


class Dis(nn.Module):
    def __init__(self, c=64, emb=8, cdim=22):
        super().__init__()
        self.emb = nn.Embedding(NLEV, emb)
        self.conv = nn.Sequential(
            nn.Conv2d(emb, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 64 + cdim, 64),
                                  nn.LeakyReLU(0.2, True), nn.Linear(64, 1))
    def forward(self, idx_img, cond):
        oh = F.one_hot(idx_img, NLEV).permute(0, 3, 1, 2).float()
        e = torch.einsum("bchw,ce->behw", oh, self.emb.weight)
        return self.head(torch.cat([self.conv(e).flatten(1), cond], 1))


def pen_tv_probs(lg):
    p = F.softmax(lg, -1).reshape(-1, 4, 8, NLEV)
    return (p[:, 1:] - p[:, :-1]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()


def speckle_np(imgs):
    n = imgs.shape[0]
    agree, iso, blobs = [], [], []
    for k in range(n):
        g = imgs[k]
        pairs = same = 0; i = 0
        for r in range(8):
            for c in range(8):
                ns = 0
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r, c] == g[rr, cc]:
                            same += 1; ns += 1
                if ns == 0: i += 1
        agree.append(same / pairs); iso.append(i)
        seen = np.zeros((8, 8), bool); cnt = 0
        for r in range(8):
            for c in range(8):
                if seen[r, c]: continue
                lv = g[r, c]; st = [(r, c)]; seen[r, c] = True
                while st:
                    rr, cc = st.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        y, x = rr + dr, cc + dc
                        if 0 <= y < 8 and 0 <= x < 8 and not seen[y, x] and g[y, x] == lv:
                            seen[y, x] = True; st.append((y, x))
                cnt += 1
        blobs.append(cnt)
    return float(np.mean(agree)), float(np.mean(iso)), float(np.mean(blobs))


REF = speckle_np(full[va[:300]])
print(f"REAL reference: agree {REF[0]*100:.1f}%  iso {REF[1]:.2f}  blobs {REF[2]:.1f}")
print(f"training decode = {A.decode}   penalty = {A.penalty}")
print()
print(f"{'lam':>7s} {'agree_smp':>10s} {'iso_smp':>8s} {'blobs_smp':>10s} "
      f"{'agree_mod':>10s} {'iso_mod':>8s} {'acc_mode':>9s} {'acc_smp':>8s} {'sec':>6s}")
print("-" * 100)

rows = []
for lam in [float(x) for x in A.lams.split(",")]:
    G = Gen(); D = Dis()
    optG = torch.optim.Adam(G.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        bi = tr[rng.integers(0, len(tr), A.batch)]
        cb = COND[bi]
        with torch.no_grad():
            lg0 = G.logits(cb)
            f = (F.gumbel_softmax(lg0, 1.0, hard=True).argmax(-1) if A.decode == "sample"
                 else lg0.argmax(-1))
        rl = D(FULL_T[bi], cb)
        fl = D(layout.image_from_left_half(f).long(), cb)
        loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) + \
                 F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
        optD.zero_grad(); loss_d.backward(); optD.step()

        lg = G.logits(cb)
        f = (F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1) if A.decode == "sample"
             else lg.argmax(-1))
        fl = D(layout.image_from_left_half(f).long(), cb)
        loss_g = F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))
        tgt = RTRUE[bi]
        loss_g = loss_g + F.cross_entropy(lg.reshape(-1, NLEV), tgt.reshape(-1))
        if lam > 0:
            loss_g = loss_g + lam * pen_tv_probs(lg)
        optG.zero_grad(); loss_g.backward(); optG.step()

    G.eval()
    with torch.no_grad():
        bv = va[:400]
        cb = COND[bv]
        lg = G.logits(cb)
        mod = lg.argmax(-1).numpy()
        smp = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1).numpy()
        Rt = RTRUE[bv]
        acc_m = float((torch.from_numpy(mod) == Rt).float().mean())
        acc_s = float((torch.from_numpy(smp) == Rt).float().mean())
    a_s, i_s, b_s = speckle_np(layout.image_from_left_half(smp))
    a_m, i_m, b_m = speckle_np(layout.image_from_left_half(mod))
    sec = time.time() - t0
    rows.append({"lam": lam, "agree_sample": a_s, "iso_sample": i_s, "blobs_sample": b_s,
                 "agree_mode": a_m, "iso_mode": i_m, "blobs_mode": b_m,
                 "acc_mode": acc_m, "acc_sample": acc_s, "seconds": round(sec, 1)})
    print(f"{lam:7.2f} {a_s*100:9.1f}% {i_s:8.2f} {b_s:10.1f} {a_m*100:9.1f}% {i_m:8.2f} "
          f"{acc_m:9.4f} {acc_s:8.4f} {sec:6.0f}")

print()
print("=" * 100)
print("READING")
print("=" * 100)
best = max(rows, key=lambda r: r["agree_mode"])
print(f"  best neighbour-agreement (mode) at lam={best['lam']}: {best['agree_mode']*100:.1f}% "
      f"vs real {REF[0]*100:.1f}%")
print(f"  accuracy at that setting: {best['acc_mode']:.4f}")
base = [r for r in rows if r["lam"] == 0.0]
if base:
    print(f"  lam=0 baseline agreement {base[0]['agree_mode']*100:.1f}%  "
          f"acc {base[0]['acc_mode']:.4f}")
print("  If agreement barely moves with lam, the penalty is not the binding constraint:")
print("  the model is under-fit, and its per-position distributions are too flat for any")
print("  smoothness term to help.")
json.dump({"reference": {"agree": REF[0], "iso": REF[1], "blobs": REF[2]},
           "rows": rows, "steps": A.steps, "decode": A.decode, "penalty": A.penalty},
          open(os.path.join(OUT, "lam_sweep.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'lam_sweep.json')}")
print("LAM_SWEEP_DONE")
