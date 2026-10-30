#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Visual check on the input-distribution measurement.

The numbers said a UNIFORM input distribution produces smoother faces (63.3% neighbour
agreement vs 42.4%) that sit closer to real faces (Hamming 18.80 vs 33.27). Both of those
look like improvements, which is suspicious: uniform inputs have a LARGER standard
deviation than the Gaussian ones the model was trained on, so they push the network
further into saturation, and saturation usually means degenerate output that happens to
score well on naive metrics.

This renders each input distribution so the claim can be judged rather than trusted.
"""
import json, os, sys
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)
OUT = os.path.join(ROOT, "results", "redstone")
os.makedirs(OUT, exist_ok=True)


def font(sz):
    for c in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", c)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


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


def faces(cond):
    with torch.no_grad():
        dec = G.logits(torch.tensor(cond, dtype=torch.float32)).argmax(-1).numpy()
    return layout.image_from_left_half(dec)


N = 16
rng = np.random.default_rng(0)


def make(kind):
    if kind == "G":
        return rng.standard_normal((N, 22)), "连续高斯（训练分布）"
    if kind == "Q8":
        g = np.clip(rng.standard_normal((N, 22)), -3.5, 3.5)
        return (np.round((g + 3.5) / 7.0 * 255.0) / 255.0 * 7.0 - 3.5), "高斯量化到 8 位"
    if kind == "U":
        return rng.uniform(-3.5, 3.5, (N, 22)), "均匀 [-3.5, 3.5]"
    if kind.startswith("B"):
        n = int(kind[1:])
        s = rng.integers(0, 2, (N, 22, n)).sum(2).astype(np.float64)
        return (s - n / 2.0) / sqrt(n / 4.0), f"二项分布（{n} 个二值投掷器求和）"
    raise ValueError(kind)


KINDS = ["G", "Q8", "B4", "B8", "B16", "U"]
TILE, PAD, LAB = 40, 6, 20
COLS = N
fnt = font(14)
W = COLS * (TILE + PAD) + PAD
H = len(KINDS) * (TILE + PAD + LAB) + PAD + 26
cv = Image.new("RGB", (W, H), (22, 24, 28))
dr = ImageDraw.Draw(cv)
dr.text((PAD, 7), "输入分布不同，输出长什么样", font=font(16), fill=(228, 232, 240))

for r, k in enumerate(KINDS):
    c, name = make(k)
    f = faces(c)
    y = 28 + r * (TILE + PAD + LAB)
    for col in range(COLS):
        t = np.kron(LEVELS[f[col]], np.ones((TILE // 8, TILE // 8), dtype=np.uint8))
        cv.paste(Image.fromarray(np.clip(t, 0, 255).astype(np.uint8), mode="L").convert("RGB"),
                 (PAD + col * (TILE + PAD), y))
    dr.text((PAD, y + TILE + 3), name, font=fnt, fill=(170, 182, 198))

cv.save(os.path.join(OUT, "INPUT_DISTRIBUTIONS.png"))
print(f"wrote {os.path.join(OUT, 'INPUT_DISTRIBUTIONS.png')}  {cv.size}")

# quantify how many DISTINCT images each distribution makes at a coarser level, which is
# what a viewer would notice, plus the fraction of fully-blank or fully-flat faces
print()
print(f"  {'distribution':34s} {'distinct 2x2 patterns / 400':>28s} {'flat faces':>11s}")
print("  " + "-" * 78)
rows = []
for k in KINDS:
    c, name = make(k) if False else (None, None)
for k in KINDS:
    rng2 = np.random.default_rng(1)
    def mk(k):
        if k == "G":
            return rng2.standard_normal((400, 22))
        if k == "Q8":
            g = np.clip(rng2.standard_normal((400, 22)), -3.5, 3.5)
            return np.round((g + 3.5) / 7.0 * 255.0) / 255.0 * 7.0 - 3.5
        if k == "U":
            return rng2.uniform(-3.5, 3.5, (400, 22))
        if k.startswith("B"):
            n = int(k[1:])
            s = rng2.integers(0, 2, (400, 22, n)).sum(2).astype(np.float64)
            return (s - n / 2.0) / sqrt(n / 4.0)
    cc = mk(k)
    ff = faces(cc)
    pats = set()
    flat = 0
    for g in ff:
        cell = np.round(g.reshape(2, 4, 2, 4).mean(axis=(1, 3)) / 15 * 3).astype(int)
        pats.add(tuple(cell.reshape(-1).tolist()))
        if g.std() < 1.0:
            flat += 1
    label = {"G": "连续高斯（训练分布）", "Q8": "高斯量化到 8 位",
             "B4": "二项 4 位", "B8": "二项 8 位", "B16": "二项 16 位",
             "U": "均匀 [-3.5,3.5]"}[k]
    rows.append({"kind": k, "distinct_2x2": len(pats), "flat": flat})
    print(f"  {label:34s} {len(pats):28d} {flat:11d}")

json.dump({"rows": rows}, open(os.path.join(OUT, "input_dist_visual.json"), "w"), indent=2)
print("INPUT_DIST_VIS_DONE")
