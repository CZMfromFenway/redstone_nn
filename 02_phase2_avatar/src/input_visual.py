#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Visual check on the claim that a handful of random inputs is enough.

Measurement B said that randomising only the top 2 components still yields 374 distinct
faces out of 400, with neighbour agreement 61.8% (smoother than real faces at 60.5%).
That is a strong claim for the redstone budget, so it needs to be LOOKED at, not just
counted: "distinct" only means pixel-different, and a set of subtly different average
faces would satisfy it while being useless.

This renders, side by side, what you actually get when only the top 1 / 2 / 3 / 8
components are randomised, all the way up to all 22, so the diversity can be judged.
"""
import os, sys
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


# influence order measured in input_cost.py
ORDER = [0, 1, 3, 5, 6, 12, 2, 4, 7, 9, 10, 14, 8, 11, 13, 17, 15, 16, 20, 21, 18, 19]
KS = [1, 2, 3, 8, 22]
COLS = 16
TILE = 40
PAD = 6
LAB = 20

fnt = font(14)
W = COLS * (TILE + PAD) + PAD
H = len(KS) * (TILE + PAD + LAB) + PAD + 34
cv = Image.new("RGB", (W, H), (22, 24, 28))
dr = ImageDraw.Draw(cv)
dr.text((PAD, 8), "只随机化前 k 个主成分时，实际得到的脸", font=font(16), fill=(228, 232, 240))

rng = np.random.default_rng(0)
for r, k in enumerate(KS):
    c = np.zeros((COLS, 22), dtype=np.float32)
    for pc in ORDER[:k]:
        c[:, pc] = rng.standard_normal(COLS)
    f = faces(c)
    y = 30 + r * (TILE + PAD + LAB)
    for col in range(COLS):
        tile = np.kron(LEVELS[f[col]],
                       np.ones((TILE // 8, TILE // 8), dtype=np.uint8))
        tile = np.clip(tile, 0, 255).astype(np.uint8)
        img = Image.fromarray(tile, mode="L").convert("RGB")
        cv.paste(img, (PAD + col * (TILE + PAD), y))
    dr.text((PAD, y + TILE + 3), f"k={k}  (只转动 {k} 个旋钮)", font=fnt,
            fill=(170, 182, 198))

cv.save(os.path.join(OUT, "INPUT_FEW_KNOBS.png"))
print(f"wrote {os.path.join(OUT, 'INPUT_FEW_KNOBS.png')}  {cv.size}")

# also a larger version of just k=2 and k=3, so quality is visible
for k in (2, 3):
    c = np.zeros((12, 22), dtype=np.float32)
    for pc in ORDER[:k]:
        c[:, pc] = rng.standard_normal(12)
    f = faces(c)
    big = Image.new("RGB", (12 * (72 + 6) + 6, 72 + 30), (22, 24, 28))
    d2 = ImageDraw.Draw(big)
    for col in range(12):
        tile = np.kron(LEVELS[f[col]], np.ones((9, 9), dtype=np.uint8))
        img = Image.fromarray(np.clip(tile, 0, 255).astype(np.uint8), mode="L").convert("RGB")
        big.paste(img, (6 + col * 78, 6))
    d2.text((6, 82), f"k={k}: 只用 {k} 个随机数生成的 12 张不同的脸", font=font(15),
            fill=(200, 210, 225))
    big.save(os.path.join(OUT, f"INPUT_K{k}_BIG.png"))
    print(f"wrote INPUT_K{k}_BIG.png  {big.size}")
print("INPUT_VIS_DONE")
