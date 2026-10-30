#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Is the contrast-loss model cheating by freezing a few pixels at the extremes?

THE SUSPICION
The user noticed that in the contrast-remedy sheet every face from the contrast-loss variant
seems to have the same three very dark pixels near the top-left corner. That would be a
degenerate solution: the loss only asks for a wider SPREAD, and the cheapest way to widen a
spread is to pin one pixel at 0 and another at 15 forever. It raises the standard deviation
without making the face any better -- and a human reading it as "fixed dark blobs" is
exactly right to distrust it.

THE TEST
For each variant, sample many faces and record, for every pixel of the 8x8 image, the mean
and the standard deviation ACROSS samples. A pixel that never changes has std ~0. A pixel
that is always black has mean ~0 as well. Then:
    * list the frozen pixels (std < 0.35 levels)
    * count how many are frozen AND dark (mean < 2)
    * compare against the baseline and the k=22 reference
If the contrast variant has several frozen dark pixels and the others do not, the remedy is
a hack, not an improvement.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "dimsweep")
WEB = os.path.join(ROOT, "webapp")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_any(tag=None, jf=None):
    if jf:
        j = json.load(open(os.path.join(WEB, jf), encoding="utf-8"))
        layers = j["gen"]["layers"]
        seq = []
        for i, L in enumerate(layers):
            W = torch.tensor(L["W"], dtype=torch.float32)
            b = torch.tensor(L["b"], dtype=torch.float32)
            lin = nn.Linear(W.shape[1], W.shape[0])
            with torch.no_grad():
                lin.weight.copy_(W); lin.bias.copy_(b)
            seq.append(lin)
            if i != len(layers) - 1:
                seq.append(nn.ReLU())
        m = nn.Sequential(*seq); m.eval()
        return m, int(len(layers[0]["W"][0]))
    ck = torch.load(os.path.join(OUT, tag, "final.pt"), weights_only=False)
    st = ck.get("G") or ck.get("S")
    st = {(k.replace("net.", "", 1) if k.startswith("net.") else k): v for k, v in st.items()}
    idx = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    seq = []
    for j in idx:
        W = st[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.load_state_dict(st); m.eval()
    return m, int(st[f"{idx[0]}.weight"].shape[1])


VARIANTS = [("dim4 baseline", None, "pca22_dim4_model.json"),
            ("dim4 + contrast loss", "dim4_contrast", None),
            ("dim4 + stretch", "dim4_stretch", None),
            ("dim22 (reference)", None, "pca22_dim22_model.json")]

N = 4000
rng = np.random.default_rng(31)
print("=" * 100)
print("FROZEN-PIXEL CHECK")
print("=" * 100)
print("  A pixel with std across samples ~0 never changes. If it is also dark, it is a")
print("  pinned black hole. Real faces have no frozen pixels.")
print()
FROZEN = 0.35
res = {}
for label, tag, jf in VARIANTS:
    net, k = load_any(tag, jf)
    Z = np.clip(rng.standard_normal((N, k)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        dec = np.clip(np.rint(net(torch.tensor(Z)).numpy()), 0, 15).astype(int)
    imgs = layout.image_from_left_half(dec)          # (N,8,8)
    per_mean = imgs.mean(0)
    per_std = imgs.std(0)
    frozen = per_std < FROZEN
    frozen_dark = frozen & (per_mean < 2.0)
    res[label] = {"frozen": int(frozen.sum()), "frozen_dark": int(frozen_dark.sum()),
                  "per_mean": per_mean.tolist(), "per_std": per_std.tolist()}
    print(f"  {label:22s} frozen pixels {int(frozen.sum()):2d}/64   of which dim/dark "
          f"{int(frozen_dark.sum()):2d}   frozen positions: "
          f"{[tuple(int(v) for v in p) for p in np.argwhere(frozen)][:8]}")

# real faces for the same measurement
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
rimg = full[:N]
print(f"  {'real faces':22s} frozen pixels {int((rimg.std(0) < FROZEN).sum()):2d}/64")

# ---------------------------------------------------------------- maps
labels = list(res)
TILE, PAD = 72, 10
W = 130 + len(labels) * (TILE + PAD) + PAD
H = 40 + 2 * (TILE + 26) + 20
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(17), font(11), font(9)
dr.text((8, 8), "每个位置跨样本的均值（上）与标准差（下）", font=f_t, fill=(234, 238, 244))
dr.text((8, 28), "标准差那一行发黑的位置＝几乎不随输入变化", font=f_s, fill=(150, 162, 178))
for i, lab in enumerate(labels):
    x = 130 + i * (TILE + PAD)
    m = np.array(res[lab]["per_mean"])
    s = np.array(res[lab]["per_std"])
    for j, (arr, scale, y0) in enumerate(((m, 15.0, 40), (s, 4.0, 40 + TILE + 26))):
        a = np.clip(arr / scale * 255, 0, 255)
        t = np.kron(a, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"), (x, y0))
    dr.text((8, 40 + TILE // 2), "均值", font=f_l, fill=(150, 162, 178))
    dr.text((8, 40 + TILE + 26 + TILE // 2), "标准差", font=f_l, fill=(150, 162, 178))
    dr.text((x, 40 + 2 * TILE + 32), lab.split(" ")[0], font=f_s, fill=(206, 214, 226))
p = os.path.join(OUT, "FROZEN_PIXELS.png")
cv.save(p)
print(f"\nwrote {p}")

# ---------------------------------------------------------------- worst offenders
print()
print("  the darkest frozen positions per variant (row, col) with mean over samples:")
for lab in labels:
    m = np.array(res[lab]["per_mean"])
    s = np.array(res[lab]["per_std"])
    idx = np.dstack(np.unravel_index(np.argsort(m.ravel()), m.shape))[0]
    worst = [(int(r), int(c), round(float(m[r, c]), 2), round(float(s[r, c]), 2))
             for r, c in idx[:3]]
    print(f"    {lab:22s} " + "  ".join(f"({r},{c}) mean {mm} sd {ss}" for r, c, mm, ss in worst))

json.dump(res, open(os.path.join(OUT, "frozen_pixels.json"), "w"), indent=2)
print("FROZEN_CHECK_DONE")
