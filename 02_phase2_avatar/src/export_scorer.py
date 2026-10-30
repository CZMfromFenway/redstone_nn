#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Export the 4->16->1 selection scorer to JSON so the browser can pre-score seeds.

This is the same scorer used in the progress sheet and in the headroom measurements: a 4->16->1
network trained on the round-5 style labels with "acceptable" as the target (grade >= 2).

It also records the threshold at the median of the score distribution over N(0,1)^4 inputs, which
is the quantity that would be burned into hardware. The browser does not need the threshold --
it draws n candidates and keeps the best -- but shipping it keeps the two implementations
describing the same thing.
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
WEB = os.path.join(ROOT, "webapp")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
torch.set_num_threads(8)

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in lab], dtype=np.float32)
y = np.array([1.0 if r["grade"] >= 2 else 0.0 for r in lab], dtype=np.float32)
mu = X.mean(0); sd = X.std(0) + 1e-9
print(f"training the scorer on {len(lab)} labels (acceptable vs 中/差)")

torch.manual_seed(1)
net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
t = torch.tensor((X - mu) / sd); v = torch.tensor(y)[:, None]
for _ in range(900):
    opt.zero_grad()
    nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
    opt.step()
net.eval()

W0 = net[0].weight.detach().numpy()
b0 = net[0].bias.detach().numpy()
W2 = net[2].weight.detach().numpy()
b2 = net[2].bias.detach().numpy()

# fold the standardisation into layer 1: W' = W/sd, b' = b - W.(mu/sd)
W0f = W0 / sd
b0f = b0 - (W0 @ (mu / sd))
Z = np.clip(np.random.default_rng(0).standard_normal((4000, 4)), -3.5, 3.5).astype(np.float32)
with torch.no_grad():
    a = net(torch.tensor(((Z - mu) / sd))).numpy().ravel()
h = np.maximum(0, Z @ W0f.T + b0f)
b = (h @ W2.T + b2).ravel()
print(f"  folding the standardisation into layer 1: max error {np.abs(a-b).max():.2e}")

pool = np.clip(np.random.default_rng(7).standard_normal((200000, 4)), -3.5, 3.5).astype(np.float32)
hp = np.maximum(0, pool @ W0f.T + b0f)
sp = (hp @ W2.T + b2).ravel()
thr = {q: float(np.quantile(sp, q)) for q in (0.50, 0.65, 0.75, 0.85)}
print(f"  threshold at the median: {thr[0.50]:+.5f}   (accept half -> 2 draws per kept seed)")

out = {
    "type": "scorer",
    "in_dim": 4,
    "layers": [
        {"W": W0f.tolist(), "b": b0f.tolist(), "relu": True},
        {"W": W2.tolist(), "b": b2.tolist(), "relu": False},
    ],
    "thresholds": {str(k): v for k, v in thr.items()},
    "meta": {
        "note": "4->16->1, trained on round-5 style labels, target = acceptable (grade >= 2)",
        "n_labels": len(lab),
        "standardisation": "folded into layer 1; feed raw N(0,1) knobs",
        "mults": int(4 * 16 + 16),
        "usage": "score a seed before generating; draw n = 1/keep candidates and keep the best",
    },
}
fp = os.path.join(WEB, "scorer_accept.json")
json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
print(f"\nwrote {fp}  ({os.path.getsize(fp)/1e3:.0f} KB)")
print(f"  {out['meta']['mults']} multiplies per scored seed")
print("SCORER_EXPORT_DONE")
