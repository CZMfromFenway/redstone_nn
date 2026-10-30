#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Filtering the colour pipeline, measured WITHOUT circularity.

THE MISTAKE THE PREVIOUS RUN MADE
It filtered with a scorer and then evaluated with the SAME scorer, so keeping the top half
guaranteed that every survivor scored above threshold and the answer came out 100%. That is
the same trap as reporting in-sample accuracy.

THE FIX
Two disjoint scorers, repeated over random splits of the labels:
    filter scorer   trained on 100 labels, used to rank and keep
    eval scorer     trained on the other 53 labels, used to judge the survivors
Nothing the filter sees contributes to the judgement. Ten splits, averaged.

The generated faces are cached once and reused across splits, since they do not depend on the
labels -- only the two scorers do.
"""
import json
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
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114])

full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
Ng = len(full)
Xg = full.reshape(Ng, -1).astype(np.float64)
mu_g = Xg.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(Ng, 60000, replace=False)
_, _, Vt = np.linalg.svd(Xg[sub] - mu_g, full_matrices=False)
comps = Vt[:22]
sd_g = (((Xg - mu_g) @ comps.T).std(0, keepdims=True) + 1e-6)[0]

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
Xl = np.array([r["pca"][:4] for r in lab], dtype=np.float32)
gl = np.array([r["grade"] for r in lab])
y = (gl >= 2).astype(float)
mu_l = Xl.mean(0); sd_l = Xl.std(0) + 1e-9
XN = ((Xl - mu_l) / sd_l).astype(np.float32)


def fit_scorer(idx, seed=0, steps=700):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
    t = torch.tensor(XN[idx])
    v = torch.tensor(y[idx], dtype=torch.float32)[:, None]
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
        opt.step()
    net.eval()
    return net


def apply(net, kn):
    with torch.no_grad():
        return net(torch.tensor(((kn - mu_l) / sd_l).astype(np.float32))).numpy().ravel()


M = 12000
CACHE = os.path.join(OUT, f"colour_knobs_{M}.npz")
if os.path.exists(CACHE):
    zc = np.load(CACHE)
    K = {k: zc[k] for k in zc.files}
    print(f"loaded cached knobs for {M:,} faces")
else:
    K = {}
    for tag in ("E6", "E8"):
        fp = os.path.join(OUT, f"stage2_{tag}.pt")
        if not os.path.exists(fp):
            continue
        m = torch.load(fp, weights_only=False)
        net = nn.Sequential(nn.Linear(m["d_in"], m["hid"]), nn.ReLU(),
                            nn.Linear(m["hid"], m["hid"]), nn.ReLU(), nn.Linear(m["hid"], 96))
        net.load_state_dict(m["state"]); net.eval()
        rng = np.random.default_rng(21)
        Z = rng.standard_normal((M, m["k"])).astype(np.float32)
        with torch.no_grad():
            rgbv = np.clip(net(torch.tensor(Z)).numpy() * 15.0, 0, 15)
        a = rgbv.reshape(M, 32, 3)
        img = np.zeros((M, 8, 8, 3), dtype=np.float32)
        img[:, layout.POS_ROW, layout.POS_COL] = a
        img[:, layout.POS_ROW, layout.MIRROR_COL] = a
        lum = (img @ LUM).reshape(M, 64)
        K[tag] = ((((lum.astype(np.float64) - mu_g) @ comps.T) / sd_g)[:, :4]).astype(np.float32)
        print(f"  {tag}: {M:,} faces generated, knobs computed")
    np.savez_compressed(CACHE, **K)

SPLITS = 10
N = len(XN)
print()
print("=" * 100)
print(f"FAIR EVALUATION: filter scorer on 100 labels, eval scorer on 53, {SPLITS} splits")
print("=" * 100)
print(f"  {'variant':8s} {'keep':>6s} {'eval-judged acceptable':>23s} {'sd':>7s} {'PC1 mean':>9s}")
print("  " + "-" * 60)
res = {}
for tag, kn in K.items():
    res[tag] = {}
    for keep in (1.0, 0.50, 0.25, 0.10):
        vals = []
        for s in range(SPLITS):
            rng = np.random.default_rng(100 + s)
            perm = rng.permutation(N)
            A, B = perm[:100], perm[100:]
            fsc = fit_scorer(A, seed=s)
            esc = fit_scorer(B, seed=s + 500)
            sf = apply(fsc, kn)
            se = apply(esc, kn)
            if keep >= 1.0:
                sel = np.arange(len(kn))
            else:
                sel = np.argsort(-sf)[:int(len(kn) * keep)]
            vals.append(float(np.mean(se[sel] > 0)))
        res[tag][str(keep)] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
        print(f"  {tag:8s} {int(keep*100):5d}% {100*np.mean(vals):22.1f}% "
              f"{100*np.std(vals):7.1f}%" if keep < 1 else
              f"  {tag:8s} {int(keep*100):5d}% {100*np.mean(vals):22.1f}% "
              f"{100*np.std(vals):7.1f}%", end="")
        print(f" {kn[np.argsort(-sf)[:int(len(kn)*max(keep,1e-9))]][:,1].mean():+9.3f}"
              if keep < 1 else f" {kn[:,1].mean():+9.3f}")
    print()

print("  the PC1 column is the mechanical reason the filter helps: it pushes the survivors")
print("  toward the negative side, which is where the user's acceptable faces live.")
print("  THIS IS STILL A MODEL JUDGEMENT on colour faces, which have no human labels.")

json.dump(res, open(os.path.join(OUT, "colour_filter_fair.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'colour_filter_fair.json')}")
print("COLOUR_FILTER_FAIR_DONE")
