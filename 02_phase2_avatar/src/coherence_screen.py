#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Which spatial-coherence penalty actually removes the speckle?

The user's complaint: generated faces are 麻麻赖赖 (speckled) while real ones form
larger connected patches of one colour. Measured earlier: real 60.5% neighbour
agreement / 6.80 isolated pixels; generated (mode) 48.6% / 12.13.

This script ranks candidate penalties by training the generator briefly with each and
measuring the speckle metrics plus how much accuracy is lost. Candidates:

  none       : the current model (CE + adversarial only)
  tv_logits  : total variation on the per-position logits
  tv_probs   : total variation on the softmax probabilities
  kl_neigh   : symmetric KL between neighbouring positions' distributions
  ce_smooth  : CE against a spatially blurred target (helps the model agree locally)
  comp       : a "connected component" reward via multi-scale box filters on the
               probability field -- explicitly rewards large same-level patches
  downsample : consistency between the full-resolution output and a 2x-downsampled
               re-encoding (coarse structure must match fine structure)

Each is trained for a SHORT pilot (a few hundred steps) on a small data slice, so the
ranking is a screening test, not the final answer. The winner then gets a long run.
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
AP.add_argument("--penalties", default="none,tv_logits,tv_probs,kl_neigh,ce_smooth,comp,downsample")
AP.add_argument("--steps", type=int, default=700)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--lam", type=float, default=0.05, help="penalty weight")
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--tag", default="screen")
A = AP.parse_args()
torch.set_num_threads(A.threads)

OUT = os.path.join(ROOT, "results", "coherence", A.tag)
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data + PCA condition
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
mask = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
        gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[mask]
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
print(f"data: train {len(tr):,}  val {len(va):,}   condition dim 22")


class Gen(nn.Module):
    """condition only (no z): the no-noise variant proved both simpler and smoother."""
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
        self.cdim = cdim
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


# ---------------------------------------------------------------- the penalties
def to_probs(lg, tau=1.0):
    return F.softmax(lg / tau, -1)


def pen_tv_logits(lg):
    p = to_probs(lg)                                  # (B,32,16)
    B = p.shape[0]
    im = p.reshape(B, 4, 8, NLEV)                     # canonical p = col*8 + row -> (col,row)
    d1 = (im[:, 1:] - im[:, :-1]).abs().mean()         # along columns
    d2 = (im[:, :, 1:] - im[:, :, :-1]).abs().mean()   # along rows
    return d1 + d2


def pen_tv_probs(lg):
    return pen_tv_logits(lg)


def pen_kl_neigh(lg):
    p = to_probs(lg).reshape(-1, 4, 8, NLEV)
    q = p.clamp_min(1e-8)
    k1 = (q[:, 1:] * (q[:, 1:].log() - q[:, :-1].log())).mean()
    k2 = (q[:, :, 1:] * (q[:, :, 1:].log() - q[:, :, :-1].log())).mean()
    return 0.5 * (k1 + k2)


def pen_ce_smooth(lg, target):
    """CE against a target blurred across neighbouring positions."""
    t = F.one_hot(target, NLEV).float().reshape(-1, 4, 8, NLEV)
    # blur along both spatial axes, keeping the distribution normalised
    tb = t.clone()
    tb[:, 1:] = 0.5 * (tb[:, 1:] + t[:, :-1])
    tb[:, :, 1:] = 0.5 * (tb[:, :, 1:] + t[:, :, :-1])
    tb = tb / tb.sum(-1, keepdim=True).clamp_min(1e-8)
    lp = F.log_softmax(lg.reshape(-1, 4, 8, NLEV), -1)
    return -(tb * lp).sum(-1).mean()


def pen_comp(lg):
    """Reward large same-level patches: apply box filters to the probability field at
    several scales; a position whose own level dominates its neighbourhood is rewarded."""
    p = to_probs(lg).reshape(-1, 4, 8, NLEV)
    B = p.shape[0]
    own = p.max(-1).values                              # (B,4,8) confidence
    # a 3x3 box smooth of "own level probability" per level, then take the matching level
    idxmax = p.argmax(-1)                               # (B,4,8)
    oh = F.one_hot(idxmax, NLEV).permute(0, 3, 1, 2).float()   # (B,16,4,8)
    k = torch.ones(1, 1, 3, 3, device=oh.device) / 9.0
    sm = F.conv2d(oh.reshape(-1, 1, 4, 8), k, padding=1).reshape(B, NLEV, 4, 8)
    match = sm.gather(1, idxmax.unsqueeze(1)).squeeze(1)        # (B,4,8) local agreement
    # penalty = 1 - local agreement  => pushes towards large uniform patches
    return (1.0 - match).mean()


def pen_downsample(lg):
    """Coarse/fine consistency: the 4x8 probability field, 2x downsampled along rows,
    should agree with the average of the pairs it came from."""
    p = to_probs(lg).reshape(-1, 4, 8, NLEV)
    pairs = p.reshape(-1, 4, 4, 2, NLEV)
    coarse = pairs.mean(3)                              # (B,4,4,16)
    up = coarse.repeat_interleave(2, dim=2)             # (B,4,8,16)
    return (p - up).abs().mean()


PEN = {
    "none": None,
    "tv_logits": pen_tv_logits,
    "tv_probs": pen_tv_probs,
    "kl_neigh": pen_kl_neigh,
    "ce_smooth": pen_ce_smooth,
    "comp": pen_comp,
    "downsample": pen_downsample,
}

# ---------------------------------------------------------------- speckle metrics
def speckle_np(imgs):
    n = imgs.shape[0]
    agree, iso, blobs = [], [], []
    for k in range(n):
        g = imgs[k]
        pairs = same = 0
        i = 0
        for r in range(8):
            for c in range(8):
                ns = 0
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r, c] == g[rr, cc]:
                            same += 1; ns += 1
                if ns == 0:
                    i += 1
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


REF_IMGS = full[va[:300]]
ref = speckle_np(REF_IMGS)
print(f"\nREAL reference: agree {ref[0]*100:.1f}%  iso {ref[1]:.2f}  blobs {ref[2]:.1f}")


def evaluate(G):
    G.eval()
    with torch.no_grad():
        bv = va[:400]
        cond = COND[bv]
        lg = G.logits(cond)
        pr = F.softmax(lg, -1)
        mode = lg.argmax(-1).numpy()
        smp = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1).numpy()
        Rt = RTRUE[bv]
        acc_m = float((torch.from_numpy(mode) == Rt).float().mean())
        acc_s = float((torch.from_numpy(smp) == Rt).float().mean())
        ent = float((-(pr * pr.clamp_min(1e-12).log()).sum(-1)).mean())
    img = layout.image_from_left_half(mode)
    a, i, b = speckle_np(img)
    G.train()
    return {"agree": a, "iso": i, "blobs": b, "acc_mode": acc_m, "acc_sample": acc_s,
            "entropy": ent}


# ---------------------------------------------------------------- screening loop
results = []
for name in A.penalties.split(","):
    name = name.strip()
    if name not in PEN:
        print(f"  unknown penalty {name}; skipping"); continue
    G = Gen(); D = Dis()
    optG = torch.optim.Adam(G.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        bi = tr[rng.integers(0, len(tr), A.batch)]
        cb = COND[bi]
        # D step
        with torch.no_grad():
            f = G.logits(cb).argmax(-1)
        rl = D(FULL_T[bi], cb)
        fl = D(layout.image_from_left_half(f).long(), cb)
        loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) + \
                 F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
        optD.zero_grad(); loss_d.backward(); optD.step()
        # G step
        lg = G.logits(cb)
        f = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1)
        fl = D(layout.image_from_left_half(f).long(), cb)
        loss_g = F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))
        tgt = RTRUE[bi]
        loss_g = loss_g + F.cross_entropy(lg.reshape(-1, NLEV), tgt.reshape(-1))
        if PEN[name] is not None:
            p = (PEN[name](lg, tgt) if name == "ce_smooth" else PEN[name](lg))
            loss_g = loss_g + A.lam * p
        optG.zero_grad(); loss_g.backward(); optG.step()
    ev = evaluate(G)
    ev["penalty"] = name
    ev["seconds"] = round(time.time() - t0, 1)
    results.append(ev)
    # distance to the real distribution, as a single "how speckled" score
    d_speck = abs(ev["agree"] - ref[0]) + abs(ev["iso"] - ref[1]) / 10.0
    ev["speckle_gap"] = float(d_speck)
    print(f"  {name:11s} agree {ev['agree']*100:5.1f}%  iso {ev['iso']:5.2f}  "
          f"blobs {ev['blobs']:5.1f}  acc(mode) {ev['acc_mode']:.4f}  "
          f"entropy {ev['entropy']:.3f}  ({ev['seconds']:.0f}s)")

results.sort(key=lambda r: r["speckle_gap"])
print()
print("=" * 100)
print("RANKED BY CLOSENESS TO THE REAL SPECKLE PROFILE (lower gap = smoother, real-like)")
print("=" * 100)
print(f"  real reference          agree {ref[0]*100:5.1f}%  iso {ref[1]:5.2f}  blobs {ref[2]:5.1f}")
for r in results:
    print(f"  {r['penalty']:11s} gap {r['speckle_gap']:.4f}   agree {r['agree']*100:5.1f}%  "
          f"iso {r['iso']:5.2f}  acc {r['acc_mode']:.4f}")
json.dump({"reference": {"agree": ref[0], "iso": ref[1], "blobs": ref[2]},
           "results": results, "steps": A.steps, "lam": A.lam},
          open(os.path.join(OUT, "screening.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'screening.json')}")
print("COHERENCE_SCREEN_DONE")
