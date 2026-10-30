#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Two experiments the user asked for, in one place so they share data and metrics:

  --mode gray32 : the generator emits 32 CONTINUOUS brightness values (regression),
                  decoded to a 16-level image by nearest level. This is the "shrink the
                  output from 512 to 32" test, and the first step of the planned
                  progressive network shrink.
  --mode color16: the 512-logit architecture but over the 16-COLOUR palette instead of
                  16 grayscale levels. Tests whether the 512 output is adequate once the
                  target is colour, where the class order is arbitrary.

Both are trained with the coherence penalty (the configuration the night's sweeps found
best) so the comparison isolates the output representation, not the speckle fix.

The gray32 arm supports two objectives:
  --soft-sigma 0   : MSE against the level index divided by 15 (regression)
  --soft-sigma > 0 : cross-entropy of a discretised prediction, `soft` only in the
                     DECODING (nearest level) -- i.e. plain regression, reported as such
  --gauss-target   : target is a Gaussian kernel over levels, so the model can express
                     "level 6.4"; decoded by argmax of the resulting distribution

Metrics are the same as every other run so the numbers are comparable.
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
AP.add_argument("--tag", required=True)
AP.add_argument("--mode", default="gray32", choices=["gray32", "color16"])
AP.add_argument("--steps", type=int, default=8000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--hid", type=int, default=256)
AP.add_argument("--layers", type=int, default=2, help="hidden layers (network shrink test)")
AP.add_argument("--lam", type=float, default=5.0)
AP.add_argument("--gauss-target", action="store_true",
                help="gray32 only: train against a Gaussian-kernel target over levels")
AP.add_argument("--tv32", action="store_true",
                help="gray32 only: apply the spatial-coherence (TV) penalty to a soft "
                     "distribution built from the continuous output, so the 32-output "
                     "architecture gets the SAME smoothness treatment the 512-logit model "
                     "had. Without this the comparison is unfair: the first 32-output run "
                     "reached acc 0.7273 but only 46.6%% neighbour agreement, against "
                     "60.7%% for the 512 model with lam=5.")
AP.add_argument("--sigma", type=float, default=0.35, help="Gaussian target width (levels)")
AP.add_argument("--adv-weight", type=float, default=1.0)
AP.add_argument("--eval-every", type=int, default=2000)
AP.add_argument("--threads", type=int, default=10)
A = AP.parse_args()
torch.set_num_threads(A.threads)
OUT = os.path.join(ROOT, "results", "shrink", A.tag)
os.makedirs(OUT, exist_ok=True)
print(f"tag={A.tag} mode={A.mode} steps={A.steps} hid={A.hid} layers={A.layers} "
      f"lam={A.lam} gauss_target={A.gauss_target}")

# ---------------------------------------------------------------- data
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full_gray = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
COND_ALL = torch.tensor(((X - mu) @ comps.T) / sd[0], dtype=torch.float32)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]

# for colour16 we need palette-index images aligned with the same rows. The clean
# custom-palette set is the same 923,975 rows, filtered by the same symmetry mask.
if A.mode == "color16":
    lab = np.load(os.path.join(ROOT, "data", "faces8_v2.npy")).astype(np.int64)   # (N,64)
    l4 = lab.reshape(-1, 8, 8)
    kn2 = (l4[:, layout.POS_ROW, layout.POS_COL] ==
           l4[:, layout.POS_ROW, layout.MIRROR_COL]).mean(axis=1) >= 0.95
    idx_col = np.flatnonzero(kn2)
    print(f"colour data: {len(idx_col):,} rows pass the symmetry filter")
    gray_c = (rgb[idx_col].astype(np.float64) * LUM).sum(-1)
    Xc = gray_c.reshape(len(idx_col), -1)
    muc = Xc.mean(0, keepdims=True)
    subc = np.random.default_rng(0).choice(len(Xc), min(60000, len(Xc)), replace=False)
    Uc, Sc, Vtc = np.linalg.svd(Xc[subc] - muc, full_matrices=False)
    compsc = Vtc[:22]
    sdc = (((Xc - muc) @ compsc.T).std(0, keepdims=True) + 1e-6)
    COND_ALL = torch.tensor(((Xc - muc) @ compsc.T) / sdc[0], dtype=torch.float32)
    IMG_ALL = torch.from_numpy(l4[idx_col].reshape(len(idx_col), 8, 8))
    # targets are the right half of the COLOUR image, canonical order
    RTRUE = IMG_ALL[:, layout.POS_ROW, layout.MIRROR_COL]
    print(f"colour condition {tuple(COND_ALL.shape)}  images {tuple(IMG_ALL.shape)}")
else:
    IMG_ALL = torch.from_numpy(full_gray.astype(np.int64))
    RTRUE = IMG_ALL[:, layout.POS_ROW, layout.MIRROR_COL]

print(f"targets {tuple(RTRUE.shape)}  range {int(RTRUE.min())}..{int(RTRUE.max())}")


class Gen(nn.Module):
    """hidden layers configurable so the shrink test can be run on smaller nets."""
    def __init__(self, hid, layers, out_dim, cdim=22):
        super().__init__()
        seq = [nn.Linear(cdim, hid), nn.ReLU()]
        for _ in range(max(0, layers - 1)):
            seq += [nn.Linear(hid, hid), nn.ReLU()]
        seq += [nn.Linear(hid, out_dim)]
        self.net = nn.Sequential(*seq)
        self.out_dim = out_dim
    def forward(self, cond):
        return self.net(cond)


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


REF = speckle_np(IMG_ALL[va[:400]].numpy().reshape(-1, 8, 8))
print(f"REAL reference: agree {REF[0]*100:.1f}%  iso {REF[1]:.2f}  blobs {REF[2]:.1f}")

# ---------------------------------------------------------------- model
OUT_DIM = 32 if A.mode == "gray32" else 32 * NLEV
G = Gen(A.hid, A.layers, OUT_DIM)
D = Dis()
nG = sum(p.numel() for p in G.parameters())
print(f"G params {nG:,}  output dim {OUT_DIM} "
      f"({'32 continuous values' if A.mode=='gray32' else '32x16 logits'})")
optG = torch.optim.Adam(G.parameters(), lr=3e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))


def decode_and_loss(raw, tgt):
    """Returns (predicted levels (B,32) long, generator loss term)."""
    if A.mode == "color16":
        lg = raw.view(-1, 32, NLEV)
        p = F.softmax(lg, -1).reshape(-1, 4, 8, NLEV)
        tv = (p[:, 1:] - p[:, :-1]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
        return lg, tv
    # gray32: raw is (B,32) continuous in "level" units
    # A soft distribution over levels is built from the continuous value; that lets the
    # SAME spatial-coherence penalty be applied as for the 512-logit architecture, so
    # the two representations can be compared fairly.
    SIG = 0.35
    mu_l = raw.clamp(0, 15)                                   # (B,32)
    lv = torch.arange(NLEV, dtype=torch.float32, device=raw.device)
    soft_logits = -((lv[None, None, :] - mu_l[:, :, None]) ** 2) / (2 * SIG ** 2)
    tv = None
    if A.tv32:
        p = F.softmax(soft_logits, -1).reshape(-1, 4, 8, NLEV)
        tv = (p[:, 1:] - p[:, :-1]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
    if A.gauss_target:
        tgt_soft = torch.exp(-((lv[None, None, :] - tgt[:, :, None].float()) ** 2)
                             / (2 * A.sigma ** 2))
        tgt_soft = tgt_soft / tgt_soft.sum(-1, keepdim=True).clamp_min(1e-9)
        ce = -(tgt_soft * F.log_softmax(soft_logits.reshape(-1, NLEV), -1)
               .reshape(-1, 32, NLEV)).sum(-1).mean()
        obj = ce
    else:
        obj = F.mse_loss(mu_l, tgt.float())
    if tv is not None:
        obj = obj + A.lam * tv
    return soft_logits if A.tv32 else None, obj


hist = []
rng = np.random.default_rng(0)
t0 = time.time()
for step in range(1, A.steps + 1):
    bi = tr[rng.integers(0, len(tr), A.batch)]
    cb = COND_ALL[bi]
    tgt = RTRUE[bi]
    # --- D step
    with torch.no_grad():
        raw = G(cb)
        if A.mode == "color16":
            f = raw.view(-1, 32, NLEV).argmax(-1)
        else:
            f = raw.clamp(0, 15).round().long()
    rl = D(IMG_ALL[bi], cb)
    fl = D(layout.image_from_left_half(f).long(), cb)
    loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) + \
             F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
    optD.zero_grad(); loss_d.backward(); optD.step()
    # --- G step
    raw = G(cb)
    if A.mode == "color16":
        lg, tv = decode_and_loss(raw, tgt)
        f = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1)
        fl = D(layout.image_from_left_half(f).long(), cb)
        ce = F.cross_entropy(lg.reshape(-1, NLEV), tgt.reshape(-1))
        loss_g = A.adv_weight * F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl)) \
                 + ce + A.lam * tv
    else:
        _, obj = decode_and_loss(raw, tgt)
        f = raw.clamp(0, 15).round().long()
        fl = D(layout.image_from_left_half(f).long(), cb)
        loss_g = A.adv_weight * F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl)) + obj
    optG.zero_grad(); loss_g.backward(); optG.step()

    if step % A.eval_every == 0 or step == A.steps:
        with torch.no_grad():
            bv = va[:400]
            raw_v = G(COND_ALL[bv])
            if A.mode == "color16":
                dec = raw_v.view(-1, 32, NLEV).argmax(-1)
            else:
                dec = raw_v.clamp(0, 15).round().long()
            Rt = RTRUE[bv]
            acc = float((dec == Rt).float().mean())
            err = float((dec.float() - Rt.float()).abs().mean())
            rm = D(IMG_ALL[bv], COND_ALL[bv]).squeeze(1)
            fm = D(layout.image_from_left_half(dec).long(), COND_ALL[bv]).squeeze(1)
            dacc = float((torch.cat([rm, fm]) > 0).float()
                         .eq(torch.cat([torch.ones_like(rm), torch.zeros_like(fm)])).float().mean())
        a_, i_, b_ = speckle_np(layout.image_from_left_half(dec))
        rec = {"step": step, "acc": round(acc, 4), "mae_levels": round(err, 3),
               "d_acc": round(dacc, 4), "agree": round(a_, 4), "iso": round(i_, 3),
               "blobs": round(b_, 2), "sec": round(time.time() - t0, 1),
               "params": nG, "mode": A.mode, "layers": A.layers, "hid": A.hid}
        hist.append(rec)
        print(f"  step {step:5d}  acc {acc:.4f}  MAE {err:.3f} levels  Dacc {dacc:.3f}  "
              f"agree {a_*100:5.1f}%  iso {i_:5.2f}  blobs {b_:5.1f}  ({rec['sec']:.0f}s)",
              flush=True)

torch.save({"G": G.state_dict(), "args": vars(A), "n_params": nG},
           os.path.join(OUT, "final.pt"))
json.dump({"args": vars(A), "n_params": nG, "reference": {"agree": REF[0], "iso": REF[1],
                                                          "blobs": REF[2]},
           "history": hist}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"\n  REAL: agree {REF[0]*100:.1f}%  iso {REF[1]:.2f}  blobs {REF[2]:.1f}")
print(f"wrote {OUT}")
print("SHRINK_DONE")
