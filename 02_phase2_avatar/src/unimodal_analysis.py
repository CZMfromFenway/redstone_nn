#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
THE KEY QUESTION: are the 16 output channels, ordered by luminance, UNIMODAL?

If they are, then the 512 logits carry far less information than they appear to: a
distribution that is one peak plus a monotone fall-off can be described by a couple of
numbers, and the whole output could plausibly collapse to 32 continuous brightness
values without losing much. If they are multi-modal, 16-way categorical output is
genuinely necessary and the 512 is irreducible.

Measured, for the best model, over held-out faces:
  1. strict unimodality : probability falls monotonically on both sides of the argmax
  2. relaxed unimodality: allows an extra step of slack within a tolerance
  3. peak count         : local maxima holding at least 5% of the mass
  4. entropy vs 2.773 (uniform) and vs the ceiling implied by the mode
  5. RECONSTRUCTION TEST -- the decisive one. Fit p(level) as a function only of the
     mode (averaged over the data) and measure the KL divergence to the true
     distributions. Low KL => a single number per position really would suffice.

The reconstruction test matters more than the peak counts, because "nearly unimodal but
skewed differently in different contexts" would still need more than one number.
"""
import json, os
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

TAG = "final_lam5_20k"
RUN = os.path.join(ROOT, "results", "coherence", TAG)
OUT = os.path.join(ROOT, "results", "unimodal")
os.makedirs(OUT, exist_ok=True)

ck = torch.load(os.path.join(RUN, "final.pt"), weights_only=False)
a = ck["args"]


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, cond):
        return self.net(cond).view(-1, 32, NLEV)


G = Gen(a["hid"], 22)
G.load_state_dict(ck["G"]); G.eval()
print(f"model: {TAG}  steps={a['steps']}  lam={a['lam']}  decode={a['decode']}")
print(f"gates: 22 -> {a['hid']} -> {a['hid']} -> 512")

# ---------------------------------------------------------------- condition + truth
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
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]

N = 4000
bv = va[:N]
COND = torch.tensor(((X[bv] - mu[0]) @ comps.T) / sd[0], dtype=torch.float32)
with torch.no_grad():
    lg = G.logits(COND)
    P = F.softmax(lg, -1).numpy()                     # (N,32,16)
print(f"\nanalysed {N} held-out faces x 32 positions = {N*32} distributions")

UNIF = np.log(NLEV)
ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(-1)     # (N,32) nats
mode = P.argmax(-1)                                    # (N,32)
maxp = P.max(-1)

# ---------------------------------------------------------------- 1/2 peak structure
def strict_unimodal(p):
    m = int(p.argmax())
    left = p[:m + 1]
    right = p[m:]
    return bool(np.all(np.diff(left) >= -1e-9) and np.all(np.diff(right) <= 1e-9))

def relaxed_unimodal(p, tol=0.02):
    """Allow small reversals: the sequence may dip by at most `tol` (absolute
    probability) before rising again. Absorbs quantisation of a smooth hump."""
    m = int(p.argmax())
    for seq in (p[:m + 1][::-1], p[m:]):
        for i in range(1, len(seq)):
            if seq[i] > seq[i - 1] + tol:
                return False
    return True

strict = np.array([[strict_unimodal(P[i, j]) for j in range(32)] for i in range(N)])
relax = np.array([[relaxed_unimodal(P[i, j]) for j in range(32)] for i in range(N)])

def npeaks(p, thresh=0.05):
    """local maxima holding at least `thresh` of the mass"""
    peaks = 0
    for c in range(NLEV):
        lo = p[c - 1] if c > 0 else 0.0
        hi = p[c + 1] if c < NLEV - 1 else 0.0
        if p[c] >= lo and p[c] >= hi and p[c] > thresh:
            peaks += 1
    return peaks

peaks = np.array([[npeaks(P[i, j]) for j in range(32)] for i in range(N)])

print()
print("=" * 96)
print("1. PEAK STRUCTURE")
print("=" * 96)
print(f"  strictly unimodal (monotone fall-off both sides) : {strict.mean()*100:5.1f}%")
print(f"  relaxed unimodal (allows <=0.02 reversals)       : {relax.mean()*100:5.1f}%")
print(f"  distributions with >=2 peaks holding >=5% mass   : {(peaks>=2).mean()*100:5.1f}%")
print(f"  mean number of such peaks                        : {peaks.mean():.2f}")
print(f"  entropy: mean {ent.mean():.3f} nats   (uniform = {UNIF:.3f}, "
      f"so {ent.mean()/UNIF*100:.1f}% of the uniform ceiling)")
print(f"  mean max-probability {maxp.mean():.3f}")

# how far does the mode sit from the distribution's own centre of mass?
levels = np.arange(NLEV)
com = (P * levels[None, None, :]).sum(-1)
print(f"  mean |argmax - centre of mass| = {np.abs(mode - com).mean():.2f} levels")

# ---------------------------------------------------------------- 3 the decisive test
print()
print("=" * 96)
print("2. RECONSTRUCTION TEST: can ONE number per position reproduce the distribution?")
print("=" * 96)
print("  Fit q(level | mode) from the data (the average shape at each mode) and measure")
print("  the KL divergence against the true p(level | context). Compare with two controls:")
print("    * a context-blind baseline: q = the global average over all positions")
print("    * the entropy of p itself: KL cannot be smaller than 0, and KL = H(p) - I;")
print("      if KL is a large share of H(p), the mode carries little of the information.")
print()

# q(level | mode): shapes learnt on a fitting half, evaluated on the other half
half = N // 2
qtab = np.zeros((NLEV, NLEV))
for c in range(NLEV):
    sel = mode[:half] == c
    if sel.sum() > 0:
        qtab[c] = P[:half][sel].mean(0)
    else:
        qtab[c] = np.full(NLEV, 1.0 / NLEV)
qtab = np.clip(qtab, 1e-9, 1)
qtab /= qtab.sum(1, keepdims=True)

Ptest = P[half:]
q_mode = qtab[mode[half:]]                    # (N/2,32,16) reconstruction from the mode
kl_mode = (Ptest * (np.log(np.clip(Ptest, 1e-12, 1)) - np.log(q_mode))).sum(-1)

q_glob = np.clip(P[half:].mean(axis=(0, 1)), 1e-9, 1)
q_glob = q_glob / q_glob.sum()
kl_glob = (Ptest * (np.log(np.clip(Ptest, 1e-12, 1)) - np.log(q_glob))).sum(-1)

Htest = -(Ptest * np.log(np.clip(Ptest, 1e-12, 1))).sum(-1)
print(f"  KL( p || q(level|mode) )        = {kl_mode.mean():.4f} nats")
print(f"  KL( p || global average shape ) = {kl_glob.mean():.4f} nats   (context-blind)")
print(f"  entropy H(p)                    = {Htest.mean():.4f} nats")
print(f"  -> the mode+shape model removes "
      f"{(Htest.mean()-kl_mode.mean())/Htest.mean()*100:.1f}% of the entropy;")
print(f"     the context-blind baseline removes "
      f"{(Htest.mean()-kl_glob.mean())/Htest.mean()*100:.1f}%.")

# a second, harder version of the same test: quantile-of-level vs level, i.e. is the
# distribution determined by where it sits on the brightness axis ALONE?
print()
print("  also: how well does a 2-parameter (mode, confidence) fit do?")
# regress the distribution on (mode, maxp) by binning
bins_m = mode[half:].ravel()
bins_c = np.clip((maxp[half:].ravel() * 5).astype(int), 0, 4)
Pt = Ptest.reshape(-1, NLEV)
q2 = np.zeros_like(Pt)
for m_ in range(NLEV):
    for c_ in range(5):
        sel = (bins_m == m_) & (bins_c == c_)
        if sel.sum() > 20:
            q2[sel] = Pt[sel].mean(0)
        else:
            q2[sel] = qtab[m_]
q2 = np.clip(q2, 1e-9, 1); q2 /= q2.sum(1, keepdims=True)
kl2 = (Pt * (np.log(np.clip(Pt, 1e-12, 1)) - np.log(q2))).sum(-1).mean()
print(f"  KL with a (mode, confidence) fit = {kl2:.4f} nats "
      f"(removes {(Htest.mean()-kl2)/Htest.mean()*100:.1f}% of the entropy)")

# ---------------------------------------------------------------- 4 collapsed-output proxy
print()
print("=" * 96)
print("3. WHAT IF THE OUTPUT WERE 32 CONTINUOUS VALUES? (proxy)")
print("=" * 96)
# Project each distribution onto a scalar in two ways and see how well the argmax of the
# reconstructed distribution agrees with the true mode.
mean_val = (P * levels[None, None, :]).sum(-1)
agree_mean = float((np.round(mean_val) == mode).mean())
print(f"  using the distribution MEAN rounded as the level reproduces the argmax in "
      f"{agree_mean*100:.1f}% of positions")
print("  (a continuous 32-value output would decode by rounding or nearest-palette;")
print("   this is the accuracy ceiling of that scheme if the mean were predicted exactly)")
json.dump({
    "tag": TAG, "n": N,
    "strict_unimodal_frac": float(strict.mean()),
    "relaxed_unimodal_frac": float(relax.mean()),
    "multipeak_frac": float((peaks >= 2).mean()),
    "mean_peaks": float(peaks.mean()),
    "entropy_mean": float(ent.mean()),
    "entropy_uniform": float(UNIF),
    "maxprob_mean": float(maxp.mean()),
    "kl_mode_shape": float(kl_mode.mean()),
    "kl_global_shape": float(kl_glob.mean()),
    "kl_mode_conf": float(kl2),
    "entropy_true": float(Htest.mean()),
    "agree_round_mean_with_mode": agree_mean,
}, open(os.path.join(OUT, "unimodal_analysis.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'unimodal_analysis.json')}")
print("UNIMODAL_DONE")
