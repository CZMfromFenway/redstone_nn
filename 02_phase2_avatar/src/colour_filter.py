#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Apply the established seed-filtering to the COLOUR pipeline.

WHY THIS SHOULD HELP
The colour generator's luminance structure sits almost exactly at the average face
(PC1 mean +0.05), while the grayscale generator's sits at PC1 -0.44. PC1-negative is the
direction the user's labels reward (it is both the 类二次元 direction and the acceptable one),
so the gray path gets a free advantage the colour path does not.

The fix is the same one already validated for grayscale: draw many seeds, score each by the
luminance knobs of the face it would produce, keep the best fraction. Here the scoring has to
happen AFTER generation, because the knobs depend on the produced face, not on the seed. That
is more expensive than the grayscale case, where the knobs ARE the seed, but it is still far
cheaper than generating and judging every face.

Reported: the predicted acceptable rate before and after filtering, at 50% and 25% keeps.
This is a MODEL JUDGEMENT, not a measured human rate.
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
WEB = os.path.join(ROOT, "webapp")
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
g = np.array([r["grade"] for r in lab])
mu_l = Xl.mean(0); sd_l = Xl.std(0) + 1e-9
torch.manual_seed(1)
sc_acc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(sc_acc.parameters(), lr=0.02, weight_decay=1e-3)
t = torch.tensor((Xl - mu_l) / sd_l)
v = torch.tensor((g >= 2).astype(float), dtype=torch.float32)[:, None]
for _ in range(900):
    opt.zero_grad()
    nn.functional.binary_cross_entropy_with_logits(sc_acc(t), v).backward()
    opt.step()
sc_acc.eval()


def knobs(lum):
    return ((((lum.astype(np.float64) - mu_g) @ comps.T) / sd_g)[:, :4]).astype(np.float32)


def score(kn):
    with torch.no_grad():
        return sc_acc(torch.tensor((kn - mu_l) / sd_l)).numpy().ravel()


M = 20000
print("=" * 100)
print("FILTERED COLOUR PIPELINE: score the face's luminance, keep the best")
print("=" * 100)
print(f"  generating {M:,} colour faces per variant (this is the expensive step)")
print()
print(f"  {'variant':18s} {'keep':>6s} {'predicted acceptable':>21s} {'PC1 mean':>9s}")
print("  " + "-" * 60)
out = {}
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
    kn = knobs((img @ LUM).reshape(M, 64))
    sc = score(kn)
    base = float(np.mean(sc > 0))
    print(f"  {tag:18s} {100:5d}% {100*base:20.1f}% {kn[:,1].mean():+9.3f}")
    out[tag] = {"keep": {}, "base": base}
    for keep in (0.50, 0.25, 0.10):
        k = int(M * keep)
        sel = np.argsort(-sc)[:k]
        r = float(np.mean(sc[sel] > 0))
        out[tag]["keep"][str(keep)] = r
        print(f"  {'':18s} {int(keep*100):5d}% {100*r:20.1f}% {kn[sel][:,1].mean():+9.3f}")
    print()

print("  the filter is applied AFTER generation because the knobs describe the produced face,")
print("  not the seed. Cost is one extra forward pass per face plus a comparison.")
print("  these are MODEL JUDGEMENTS; the real rate needs labels on colour faces.")

json.dump(out, open(os.path.join(OUT, "colour_filter.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'colour_filter.json')}")
print("COLOUR_FILTER_DONE")
