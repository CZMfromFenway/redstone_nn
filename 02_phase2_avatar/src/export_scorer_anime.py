#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Export the 类二次元 probe as a scorer the browser can apply.

The existing scorer_accept.json predicts "acceptable" (any of the three styles). This one
predicts "类二次元" specifically, trained on the same round-5 style labels with grade==4 as the
positive class. Both are 4 -> 16 -> 1 over the same four knobs, so any PCA4 model can use either.

The standardisation is folded into layer 1, so the page feeds raw N(0,1) knobs.
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
Y = np.array([1.0 if r["grade"] == 4 else 0.0 for r in lab], dtype=np.float32)
print(f"training the 类二次元 probe on {len(lab)} labels "
      f"({int(Y.sum())} positive = {100*Y.mean():.1f}%)")

mu = X.mean(0); sd = X.std(0) + 1e-9
torch.manual_seed(2)
net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
t = torch.tensor((X - mu) / sd)
v = torch.tensor(Y)[:, None]
for _ in range(900):
    opt.zero_grad()
    nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
    opt.step()
net.eval()

W0 = net[0].weight.detach().numpy(); b0 = net[0].bias.detach().numpy()
W2 = net[2].weight.detach().numpy(); b2 = net[2].bias.detach().numpy()
W0f = W0 / sd
b0f = b0 - (W0 @ (mu / sd))

Z = np.clip(np.random.default_rng(0).standard_normal((4000, 4)), -3.5, 3.5).astype(np.float32)
with torch.no_grad():
    ref = net(torch.tensor(((Z - mu) / sd))).numpy().ravel()
folded = (np.maximum(0, Z @ W0f.T + b0f) @ W2.T + b2).ravel()
print(f"  folding the standardisation into layer 1: max error {np.abs(ref-folded).max():.2e}")

pool = np.clip(np.random.default_rng(7).standard_normal((200000, 4)), -3.5, 3.5).astype(np.float32)
sp = (np.maximum(0, pool @ W0f.T + b0f) @ W2.T + b2).ravel()
thr = {q: float(np.quantile(sp, q)) for q in (0.50, 0.65, 0.75, 0.85)}
print(f"  threshold at the median: {thr[0.50]:+.5f} "
      f"(that keeps {100*(1-0.50):.0f}% of draws)")

out = {
    "type": "scorer",
    "in_dim": 4,
    "target": "anime",
    "layers": [
        {"W": W0f.tolist(), "b": b0f.tolist(), "relu": True},
        {"W": W2.tolist(), "b": b2.tolist(), "relu": False},
    ],
    "thresholds": {str(k): v for k, v in thr.items()},
    "meta": {
        "note": "4->16->1, target = 类二次元 (grade 4 in the round-5 style labels)",
        "n_labels": len(lab), "positive_rate": float(Y.mean()),
        "standardisation": "folded into layer 1; feed raw N(0,1) knobs",
        "mults": int(4 * 16 + 16),
    },
}
fp = os.path.join(WEB, "scorer_anime.json")
json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
print(f"\nwrote {fp}  ({os.path.getsize(fp)/1e3:.0f} KB, {out['meta']['mults']} multiplies)")
print("ANIME_SCORER_DONE")
