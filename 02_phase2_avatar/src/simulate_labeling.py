#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
SIMULATION: with only ~100 manual labels, does preference-guided CEM actually improve
the sampler? This tests the MACHINERY, using a measurable proxy instead of the user's
eye, so the answer is available before the user spends 100 judgements.

Proxy for "好看": the Hamming distance from the generated 8x8 face to the nearest real
face in a held-out pool. Lower = more face-like. This is not the user's taste, but it is
(a) measurable, (b) strongly related to what the user complained about (structure and
speckle), and (c) the property that would have to improve for the output to look better.

Protocol, repeated for several seeds so the result is not one lucky run:
    * start from the base sampler N(0,1)^22
    * in rounds of 20, draw 20 candidates, score them with the proxy, refit the CEM
      diagonal Gaussian on the elite (best 50%), never below an exploration floor
    * after 20/40/60/80/100 labels, estimate the proxy score of a fresh batch drawn
      from the current sampler
    * compare with a control that keeps sampling from the base distribution

Reports the improvement in "mean distance to nearest real face" and in the share of
candidates that beat the base distribution's median, which is the practical question:
how much better does the sampler get after N labels?
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16

AP = argparse.ArgumentParser()
AP.add_argument("--model-tag", default="final_lam5_20k")
AP.add_argument("--rounds", type=int, default=5)
AP.add_argument("--per-round", type=int, default=20)
AP.add_argument("--elite-frac", type=float, default=0.5)
AP.add_argument("--prior", type=float, default=24.0)
AP.add_argument("--sd-floor", type=float, default=0.55)
AP.add_argument("--eps", type=float, default=0.15)
AP.add_argument("--seeds", default="0,1,2")
AP.add_argument("--eval-n", type=int, default=400)
AP.add_argument("--dims", type=int, default=22,
                help="search only in the top-k principal components; freezes the rest "
                     "at the base N(0,1). Tests whether the search succeeds when the "
                     "effective dimension is low enough for the label budget.")
A = AP.parse_args()

OUT = os.path.join(ROOT, "results", "preference_sim")
os.makedirs(OUT, exist_ok=True)

# ---- model -------------------------------------------------------------------------
class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)

run = os.path.join(ROOT, "results", "coherence", A.model_tag)
ck = torch.load(os.path.join(run, "final.pt"), weights_only=False)
G = Gen(ck["args"]["hid"], 22); G.load_state_dict(ck["G"]); G.eval()
print(f"model {A.model_tag}: {ck['args']['steps']} steps, lam={ck['args']['lam']}")

# ---- held-out REAL pool: the proxy measures distance to THIS ------------------------
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
POOL = full[va[:3000]].reshape(3000, -1).astype(np.int16)


def score_batch(Z, chunk=100):
    """proxy score: negative mean distance to the nearest real face (higher = better)"""
    Zt = torch.tensor(np.asarray(Z, dtype=np.float32))
    with torch.no_grad():
        dec = G.logits(Zt).argmax(-1).numpy()
    img = layout.image_from_left_half(dec).reshape(len(Z), -1).astype(np.int16)
    d = np.empty(len(img))
    for s in range(0, len(img), chunk):
        blk = img[s:s + chunk]
        d[s:s + chunk] = (blk[:, None, :] != POOL[None, :, :]).sum(2).min(1)
    return -d          # higher is better


def speckle_of(Z):
    Zt = torch.tensor(np.asarray(Z, dtype=np.float32))
    with torch.no_grad():
        dec = G.logits(Zt).argmax(-1).numpy()
    imgs = layout.image_from_left_half(dec)
    agree = []
    for g in imgs:
        pairs = same = 0
        for r in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r, c] == g[rr, cc]:
                            same += 1
        agree.append(same / pairs)
    return float(np.mean(agree))


def base_median(seed=0):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.standard_normal((A.eval_n, 22)), -3.5, 3.5)
    return float(np.median(score_batch(Z)))


def sample_from(dist, n, rng):
    if dist is None:
        return np.clip(rng.standard_normal((n, 22)), -3.5, 3.5)
    Z = dist["mean"][None, :] + dist["sd"][None, :] * rng.standard_normal((n, 22))
    use_base = rng.random(n) < A.eps
    Z[use_base] = rng.standard_normal((use_base.sum(), 22))
    return np.clip(Z, -3.5, 3.5)


def refit(Z, S, prior, sd_floor, dims=22):
    k = max(2, int(len(Z) * A.elite_frac))
    idx = np.argsort(-S)[:k]
    E = Z[idx]
    n = len(E)
    m = E.mean(0)
    s = E.std(0) if n > 1 else np.ones(22)
    w = n / (n + prior)
    m = w * m
    s = np.sqrt(w * (s ** 2) + (1 - w) * 1.0)
    s = np.maximum(s, sd_floor)
    if dims < 22:
        # freeze every component past the first `dims` at the base N(0,1), so the search
        # only spends its label budget on the dimensions that matter
        m = m.copy(); s = s.copy()
        m[dims:] = 0.0
        s[dims:] = 1.0
    return {"mean": m, "sd": s}


print()
print("=" * 100)
print("SIMULATION: preference-guided CEM with a measurable proxy (distance to real faces)")
print("=" * 100)
print("proxy = negative Hamming distance from the generated face to the nearest real face")
print("        (higher is better; a face-like output scores higher)")
print()

rows = []
for seed in [int(s) for s in A.seeds.split(",")]:
    rng = np.random.default_rng(seed)
    med = base_median(seed)
    dist = None
    traj = []
    labelled = 0
    for rd in range(A.rounds):
        Z = sample_from(dist, A.per_round, rng)
        S = score_batch(Z)
        labelled += A.per_round
        dist = refit(Z, S, A.prior, A.sd_floor, A.dims)
        # evaluate the CURRENT sampler on a fresh batch
        Ze = sample_from(dist, A.eval_n, np.random.default_rng(seed * 1000 + rd))
        Se = score_batch(Ze)
        frac_better = float((Se > med).mean())
        traj.append({"labels": labelled, "mean_proxy": float(Se.mean()),
                     "frac_above_base_median": frac_better,
                     "speckle_agree": speckle_of(Ze)})
        print(f"  seed {seed}  labels {labelled:3d}  mean proxy {Se.mean():7.2f}  "
              f"share beating the base median {frac_better*100:5.1f}%  "
              f"agree {traj[-1]['speckle_agree']*100:.1f}%")
    # control: the base sampler, evaluated the same number of times
    Zc = np.clip(np.random.default_rng(seed + 999).standard_normal((A.eval_n, 22)), -3.5, 3.5)
    Sc = score_batch(Zc)
    ctrl = {"mean_proxy": float(Sc.mean()),
            "frac_above_base_median": float((Sc > med).mean()),
            "speckle_agree": speckle_of(Zc)}
    print(f"  seed {seed}  BASE      mean proxy {ctrl['mean_proxy']:7.2f}  "
          f"share beating the base median {ctrl['frac_above_base_median']*100:5.1f}%  "
          f"agree {ctrl['speckle_agree']*100:.1f}%")
    print()
    rows.append({"seed": seed, "trajectory": traj, "control": ctrl, "base_median": med})

print("=" * 100)
print("SUMMARY")
print("=" * 100)
last = [r["trajectory"][-1] for r in rows]
ctrl = [r["control"] for r in rows]
print(f"  after {A.rounds*A.per_round} labels:")
print(f"    mean proxy        {np.mean([x['mean_proxy'] for x in last]):7.2f} "
      f"(base {np.mean([x['mean_proxy'] for x in ctrl]):7.2f})  "
      f"gain {np.mean([x['mean_proxy'] for x in last]) - np.mean([x['mean_proxy'] for x in ctrl]):+.2f}")
print(f"    share beating base median "
      f"{np.mean([x['frac_above_base_median'] for x in last])*100:5.1f}% "
      f"(base {np.mean([x['frac_above_base_median'] for x in ctrl])*100:5.1f}%)")
print(f"    neighbour agreement "
      f"{np.mean([x['speckle_agree'] for x in last])*100:.1f}% "
      f"(base {np.mean([x['speckle_agree'] for x in ctrl])*100:.1f}%)")
print()
print("READING")
print("  * 'share beating the base median' is the practical number: with no improvement")
print("    it stays near 50%. Each point above 50 is a candidate the user would more")
print("    often prefer, if the proxy matches their taste.")
print("  * the proxy is distance-to-real-faces, NOT the user's aesthetic judgement.")
print("    It tests whether the CEM machinery can climb a measurable objective at this")
print("    label budget; it cannot promise the same gain for human preference.")
json.dump({"args": vars(A), "rows": rows}, open(os.path.join(OUT, "sim.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'sim.json')}")
print("SIM_DONE")
