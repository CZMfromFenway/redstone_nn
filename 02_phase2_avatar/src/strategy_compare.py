#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
WHICH LABELLING STRATEGY ACTUALLY WORKS AT 100 LABELS?

learnability.py showed the proxy is predictable from the 22 coordinates nonlinearly
(MLP R^2 = 0.222) but barely linearly (ridge R^2 = 0.034). That is the key fact:
"goodness" is a NONLINEAR, possibly multi-modal landscape, so:

  * CEM (fit a diagonal Gaussian to the elite) can only shrink to ONE box. It cannot
    represent "good here AND there, bad in between". Measured result: +1.0 unit of
    Hamming distance after 100 labels, essentially noise.
  * A REWARD MODEL trained on the labels can represent any shape, and then
    generate-and-filter turns it into a sampler.

This script compares them head to head at the same label budget:

  A. base        : no labels, sample N(0,1)^22
  B. CEM         : the diagonal-Gaussian elite refit (what the page currently does)
  C. reward+topk : fit a regressor on the labels, draw a large pool, keep the best q%
  D. reward+IS   : fit a regressor, then importance-resample the base draws by
                   exp(beta * predicted score)

All four are evaluated on the SAME held-out proxy, with several seeds, so the comparison
is not one lucky run. The winner is what the labelling page should do once labels exist.
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
OUT = os.path.join(ROOT, "results", "strategy")
os.makedirs(OUT, exist_ok=True)

AP = argparse.ArgumentParser()
AP.add_argument("--budget", type=int, default=100)
AP.add_argument("--rounds", type=int, default=5)
AP.add_argument("--eval-n", type=int, default=400)
AP.add_argument("--seeds", default="0,1,2")
AP.add_argument("--pool", type=int, default=2000, help="candidate pool for top-k")
AP.add_argument("--keep", type=float, default=0.15, help="fraction kept by top-k")
AP.add_argument("--beta", type=float, default=2.0, help="inverse temperature for IS")
A = AP.parse_args()


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


ck = torch.load(os.path.join(ROOT, "results", "coherence", "final_lam5_20k", "final.pt"),
                weights_only=False)
G = Gen(ck["args"]["hid"], 22); G.load_state_dict(ck["G"]); G.eval()
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
POOL = full[va[:3000]].reshape(3000, -1).astype(np.int16)


def proxy(Z):
    Zt = torch.tensor(np.asarray(Z, dtype=np.float32))
    with torch.no_grad():
        dec = G.logits(Zt).argmax(-1).numpy()
    img = layout.image_from_left_half(dec).reshape(len(Z), -1).astype(np.int16)
    d = np.empty(len(img))
    for s in range(0, len(img), 128):
        d[s:s + 128] = (img[s:s + 128][:, None, :] != POOL[None, :, :]).sum(2).min(1)
    return -d


class Reward(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(22, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_reward(X, y, steps=2500, hid=128, seed=0, l2=1e-4):
    torch.manual_seed(seed)
    m = Reward(hid)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=l2)
    Xt = torch.tensor(X, dtype=torch.float32)
    mu, sd = float(y.mean()), float(y.std() + 1e-9)
    yt = torch.tensor((y - mu) / sd, dtype=torch.float32)
    for _ in range(steps):
        loss = ((m(Xt) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    def predict(Z):
        with torch.no_grad():
            return m(torch.tensor(np.asarray(Z, dtype=np.float32))).numpy() * sd + mu
    return predict


def run_strategy(name, seed, rounds, budget):
    rng = np.random.default_rng(seed)
    per = budget // rounds
    med = float(np.median(proxy(np.clip(rng.standard_normal((A.eval_n, 22)), -3.5, 3.5))))
    Xs, ys = [], []
    dist = None
    traj = []
    for rd in range(rounds):
        # ---- draw candidates from the CURRENT strategy
        if name == "base":
            C = np.clip(rng.standard_normal((per, 22)), -3.5, 3.5)
        elif name == "cem":
            if dist is None:
                C = np.clip(rng.standard_normal((per, 22)), -3.5, 3.5)
            else:
                C = np.clip(dist["mean"][None, :] +
                            dist["sd"][None, :] * rng.standard_normal((per, 22)), -3.5, 3.5)
        elif name in ("reward_topk", "reward_is"):
            # explore with a mixture: half from base, half concentrated where the reward
            # model currently predicts high (that is the point of learning it)
            nb = per // 2
            Cb = np.clip(rng.standard_normal((nb, 22)), -3.5, 3.5)
            if Xs:
                P = np.clip(rng.standard_normal((A.pool, 22)), -3.5, 3.5)
                pred = predict_reward(P)
                top = np.argsort(-pred)[:per - nb]
                Cn = P[top]
                C = np.vstack([Cb, Cn])
            else:
                C = np.clip(rng.standard_normal((per, 22)), -3.5, 3.5)
        S = proxy(C)                     # the user's clicks, simulated
        Xs.append(C); ys.append(S)
        X = np.vstack(Xs); y = np.concatenate(ys)
        # ---- update the strategy
        if name == "cem":
            k = max(2, int(len(C) * 0.5))
            idx = np.argsort(-S)[:k]
            E = C[idx]
            n = len(E)
            m = E.mean(0); s = E.std(0) if n > 1 else np.ones(22)
            w = n / (n + 24.0)
            m = w * m
            s = np.sqrt(w * (s ** 2) + (1 - w) * 1.0)
            dist = {"mean": m, "sd": np.maximum(s, 0.55)}
        elif name in ("reward_topk", "reward_is"):
            predict_reward = fit_reward(X, y, seed=seed)
        # ---- evaluate the current sampler on a fresh batch
        Ze = np.clip(rng.standard_normal((A.eval_n, 22)), -3.5, 3.5)
        if name == "cem" and dist is not None:
            Ze[: A.eval_n // 2] = np.clip(
                dist["mean"][None, :] + dist["sd"][None, :] *
                rng.standard_normal((A.eval_n // 2, 22)), -3.5, 3.5)
        elif name == "reward_topk" and Xs:
            P = np.clip(rng.standard_normal((A.eval_n * 8, 22)), -3.5, 3.5)
            keep = np.argsort(-predict_reward(P))[: A.eval_n]
            Ze = P[keep]
        elif name == "reward_is" and Xs:
            P = np.clip(rng.standard_normal((A.eval_n, 22)), -3.5, 3.5)
            pr = predict_reward(P)
            w = np.exp(A.beta * (pr - pr.max()))
            w /= w.sum()
            Ze = P[rng.choice(len(P), A.eval_n, p=w)]
        Se = proxy(Ze)
        traj.append({"labels": (rd + 1) * per,
                     "mean_proxy": float(Se.mean()),
                     "frac_above_median": float((Se > med).mean())})
    return {"strategy": name, "seed": seed, "trajectory": traj, "base_median": med}


print("=" * 96)
print(f"LABELLING STRATEGY COMPARISON  (budget {A.budget} labels, {A.rounds} rounds)")
print("=" * 96)
print("evaluated on the same held-out proxy; 'frac above median' = share of candidates")
print("better than a median random draw (0.50 = no improvement)")
print()
allrows = []
for name in ("base", "cem", "reward_topk", "reward_is"):
    for seed in [int(s) for s in A.seeds.split(",")]:
        r = run_strategy(name, seed, A.rounds, A.budget)
        allrows.append(r)
        t = r["trajectory"][-1]
        print(f"  {name:12s} seed {seed}  after {t['labels']:3d} labels: "
              f"mean proxy {t['mean_proxy']:7.2f}  above-median {t['frac_above_median']*100:5.1f}%")

print()
print("=" * 96)
print("SUMMARY (mean over seeds, at the full label budget)")
print("=" * 96)
print(f"  {'strategy':14s} {'mean proxy':>11s} {'above median':>13s} {'vs base':>9s}")
print("  " + "-" * 52)
by = {}
for name in ("base", "cem", "reward_topk", "reward_is"):
    rs = [r for r in allrows if r["strategy"] == name]
    mp = np.mean([r["trajectory"][-1]["mean_proxy"] for r in rs])
    am = np.mean([r["trajectory"][-1]["frac_above_median"] for r in rs])
    by[name] = {"mean_proxy": float(mp), "above_median": float(am)}
base_am = by["base"]["above_median"]
for name in ("base", "cem", "reward_topk", "reward_is"):
    d = by[name]
    print(f"  {name:14s} {d['mean_proxy']:11.2f} {d['above_median']*100:12.1f}% "
          f"{(d['above_median']-base_am)*100:+8.1f}")
print()
best = max(("cem", "reward_topk", "reward_is"), key=lambda k: by[k]["above_median"])
gain = (by[best]["above_median"] - base_am) * 100
print(f"  best strategy: {best}   {gain:+.1f} points of 'above median' over the base")
print()
print("=" * 96)
print("MEMORISATION CHECK -- the trap this comparison must not fall into")
print("=" * 96)
print("The proxy is 'distance to the nearest REAL face', so any strategy that optimises")
print("it directly is rewarded for REPRODUCING training faces. A candidate that is only")
print("a few pixels from a real face is a copy, not a generation. The honest question is")
print("therefore not just 'is the proxy better' but 'did it get better by copying'.")
print()
print(f"  reference: base draws sit {abs(by['base']['mean_proxy']):.1f} pixels from a real face on average")
for name in ("cem", "reward_topk", "reward_is"):
    d = abs(by[name]["mean_proxy"])
    if d < 8:
        verdict = "COLLAPSED TO MEMORISATION (not a real quality gain)"
    elif d < 12:
        verdict = "suspiciously close to the training set"
    else:
        verdict = "plausibly a genuine improvement"
    print(f"  {name:14s} {d:5.1f} pixels  -> {verdict}")
print()
print("  This is why 'reward_is' scoring 100% above the median is NOT a win: driving a")
print("  generative model onto its training data always maximises a proximity proxy. The")
print("  usable strategy is the one that improves while staying far enough from the")
print("  training set to still be generating rather than copying.")
print()
print("HOW TO READ THIS")
print("  'above median' is the practical number: the share of candidates a user who agrees")
print("  with the proxy would prefer over a random draw. 50% means no improvement.")
print(f"  With {A.budget} labels, {best} moves it to {by[best]['above_median']*100:.0f}%, but check the")
print("  memorisation column before believing that number.")
json.dump({"args": vars(A), "rows": allrows, "summary": by, "best": best},
          open(os.path.join(OUT, "strategy.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'strategy.json')}")
print("STRATEGY_DONE")
