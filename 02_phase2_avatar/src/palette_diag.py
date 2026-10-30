#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Why is the K=8 output less saturated than the real data, and what exactly should be changed?

Two very different causes give the same symptom:

    (a) the PALETTE itself is under-saturated, i.e. k-means picked dull colours, so no choice of
        indices can fix it -- the eight values have to change
    (b) the palette is fine but the CLASSIFIER prefers the dull entries, so the eight values are
        fine and the fix is in training

This separates them by measuring, for both the real data and the generated faces:
    - which palette entries are used, and how often
    - the saturation each entry actually contributes (its saturation times its share)
    - the saturation the real data would get if it used the palette optimally

If the real data's achievable saturation with this palette is close to 0.32, the palette is fine
and the classifier is the problem.
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
PAL = os.path.join(OUT, "palette")
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def sat_of(x):
    mx = x.max(-1); mn = x.min(-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


z = np.load(os.path.join(OUT, "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
va = sp["val"]
K = 8

ck = torch.load(os.path.join(PAL, f"P{K}.pt"), weights_only=False)
C = ck["palette"]
hid = ck["hid"]


class PM(nn.Module):
    def __init__(self, hid, K):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.head = nn.Linear(hid, 32 * K)
        self.K = K

    def forward(self, c):
        return self.head(self.body(c)).view(-1, 32, self.K)


PM.K = K
net = PM(hid, K)
net.load_state_dict(ck["state"]); net.eval()

pos_real = rgb96.reshape(-1, 32, 3)
d = ((pos_real[:, :, None, :] - C[None, None, :, :]) ** 2).sum(-1)
idx_real = d.argmin(-1)
with torch.no_grad():
    CT = torch.tensor(cpc[:, :8])
    idx_gen = net(CT).argmax(-1).numpy()

sat_entry = sat_of(C)
share_real = np.array([(idx_real == j).mean() for j in range(K)])
share_gen = np.array([(idx_gen == j).mean() for j in range(K)])

print("=" * 104)
print("K=8 PALETTE DIAGNOSTIC: is the palette dull, or is the classifier choosing dull colours?")
print("=" * 104)
print(f"  {'#':>2s} {'RGB':>22s} {'亮度':>6s} {'饱和':>6s} {'真实占比':>9s} {'模型占比':>9s} "
      f"{'真实贡献':>9s} {'模型贡献':>9s}")
print("  " + "-" * 92)
order = np.argsort(-C.mean(1))
for j in order:
    r, g, b = C[j]
    lp = 0.299 * r + 0.587 * g + 0.114 * b
    print(f"  {j:2d} ({r:5.2f},{g:5.2f},{b:5.2f}) {lp:6.2f} {sat_entry[j]:6.2f} "
          f"{100*share_real[j]:8.1f}% {100*share_gen[j]:8.1f}% "
          f"{sat_entry[j]*share_real[j]:9.3f} {sat_entry[j]*share_gen[j]:9.3f}")
print("  " + "-" * 92)
sr = float((sat_entry * share_real).sum())
sg = float((sat_entry * share_gen).sum())
print(f"  {'加权平均饱和':>34s} {sr:25.3f} {sg:9.3f}")
print()
print(f"  真实数据自身的饱和 = {sat_of(pos_real.reshape(-1,3)).mean():.3f}")
print(f"  用这个调色板、按真实数据的分布选择 -> 可达 {sr:.3f}")
print(f"  模型实际选择 -> {sg:.3f}")
print()
gap_pal = abs(sr - 0.323)
gap_model = abs(sr - sg)
print(f"  调色板本身造成的差 = {gap_pal:.3f}")
print(f"  模型选择造成的差   = {gap_model:.3f}")
if gap_pal > gap_model:
    print("  -> 主要是调色板偏灰：需要改这 8 个值。")
else:
    print("  -> 主要是分类器偏爱低饱和条目：调色板可以不动，改训练即可。")

# what would an optimally saturated palette look like, keeping the same luminance ladder?
print()
print("=" * 104)
print("OPTIMISED 8-COLOUR PALETTE")
print("=" * 104)
# Refit with a saturation target: weight the k-means objective by how much each pixel needs
# saturation, so dull colours are chosen less readily. Concretely: push each centroid away from
# grey along its own chroma direction until the palette's weighted saturation matches the real
# data's, keeping luminance and hue fixed. That is monotone, cheap, and cannot invent new hues.
target = 0.323
C2 = C.copy()
for scale in np.arange(0.8, 3.01, 0.02):
    D = C2.mean(1, keepdims=True)
    Cp = np.clip(D + (C - D) * scale, 0, 15)
    s = float((sat_of(Cp) * share_real).sum())
    if s >= target:
        break
Cnew = Cp
print(f"  把每个颜色沿自身色相方向拉离灰轴，亮度不变，直到加权饱和达到 {target}")
print(f"  缩放系数 {scale:.2f}")
print(f"  {'#':>2s} {'原值':>22s} {'新值':>22s} {'原饱和':>7s} {'新饱和':>7s}")
print("  " + "-" * 70)
for j in order:
    print(f"  {j:2d} ({C[j][0]:5.2f},{C[j][1]:5.2f},{C[j][2]:5.2f}) "
          f"({Cnew[j][0]:5.2f},{Cnew[j][1]:5.2f},{Cnew[j][2]:5.2f}) "
          f"{sat_entry[j]:7.2f} {sat_of(Cnew[j]):7.2f}")
print()
print(f"  新调色板的加权饱和 = {float((sat_of(Cnew) * share_real).sum()):.3f}")
np.save(os.path.join(PAL, "palette_k8_optimised.npy"), Cnew)
json.dump({"K": K, "old": C.tolist(), "new": Cnew.tolist(), "scale": float(scale),
           "share_real": share_real.tolist(), "sat_old": sat_entry.tolist()},
          open(os.path.join(PAL, "palette_k8.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(PAL, 'palette_k8.json')}")
print("PALETTE_DIAG_DONE")
