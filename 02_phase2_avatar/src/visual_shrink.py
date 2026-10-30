#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Show what the two new architectures actually produce: the 16-colour / 512-logit model and
the 32-channel continuous-output model.

The two cannot be compared pixel-for-pixel (one predicts palette indices, the other a
continuous brightness which is then rounded to a level), so this script produces a
separate sheet for each with the same layout:

    row 1  real hold-out faces
    row 2  the model's MODE output      (what the redstone/argmax path would draw)
    row 3  the model's SAMPLED output   (what a stochastic decoder would draw)

and, for the 32-channel model, a fourth row showing the CONTINUOUS value before rounding,
rendered at higher precision by mapping 0..15 onto 0..255 directly. That is the whole
point of the architecture: the continuous value carries sub-level information that the
512-logit model throws away.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)

AP = argparse.ArgumentParser()
AP.add_argument("--tag", default="s32_tv5")
AP.add_argument("--n", type=int, default=24)
AP.add_argument("--cols", type=int, default=12)
API = AP.parse_args()

OUT = os.path.join(ROOT, "results", "shrink", "VISUAL")
os.makedirs(OUT, exist_ok=True)
PALETTE = np.load(os.path.join(ROOT, "data", "palette_avatar16_v2.npy")).astype(np.uint8)


def _cjk_font(size=13):
    """PIL's default bitmap font has no CJK glyphs, so Chinese labels render as boxes.
    Try the Windows fonts, and fall back to English labels if none is present."""
    from PIL import ImageFont
    for cand in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", cand)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size), True
            except Exception:
                pass
    return ImageFont.load_default(), False


def Sheet(rows, cols, tile=56, labels=None):
    n = len(rows[0])
    nrow = len(rows)
    pad = 6
    font, cjk = _cjk_font(13)
    if labels and not cjk:
        # no CJK font available: use the ASCII transliteration instead of boxes
        labels = [l.encode("ascii", "replace").decode() for l in labels]
    lab = 18 if labels else 0
    W = cols * (tile + pad) + pad
    H = nrow * (tile + pad + lab) + pad
    cv = Image.new("RGB", (W, H), (22, 24, 28))
    dr = ImageDraw.Draw(cv)
    for r in range(nrow):
        for c in range(cols):
            if c >= n:
                break
            x = pad + c * (tile + pad)
            y = pad + r * (tile + pad + lab)
            im = rows[r][c]
            if im.ndim == 2:
                big = np.kron(np.clip(im, 0, 255).astype(np.uint8),
                              np.ones((tile // 8, tile // 8), dtype=np.uint8))
                big = np.stack([big] * 3, -1)
            else:
                big = np.repeat(np.repeat(np.clip(im, 0, 255).astype(np.uint8),
                                          tile // 8, 0), tile // 8, 1)
            cv.paste(Image.fromarray(big, mode="RGB"), (x, y))
        if labels:
            dr.text((pad, pad + r * (tile + pad + lab) + tile + 2), labels[r],
                    fill=(170, 180, 195), font=font)
    return cv


class GenGray32(nn.Module):
    def __init__(self, hid=256, out=32, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, out))
    def forward(self, c):
        return self.net(c)


class GenColor(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def forward(self, c):
        return self.net(c).view(-1, 32, NLEV)


def load_pca_and_split(colour):
    """Rebuild exactly the condition the model was trained with.

    IMPORTANT: the grayscale canonical arrays (L0_full_idx.npy, L0_split.npz) are ALREADY
    the symmetry-filtered subset of 321,219 rows, so the validation indices refer to that
    subset and must not be re-applied to the full 923,975-row array. The colour path
    filters for itself (398,474 rows, a different mask because it compares palette
    indices rather than luminance) and is indexed by the same split, which is exactly
    what shrink_32out.py did during training.
    """
    rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
    LUM = np.array([0.299, 0.587, 0.114])
    gray = (rgb.astype(np.float64) * LUM).sum(-1)
    va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
    if colour:
        lab = np.load(os.path.join(ROOT, "data", "faces8_v2.npy")).astype(np.int64)
        l4 = lab.reshape(-1, 8, 8)
        mask = (l4[:, layout.POS_ROW, layout.POS_COL] ==
                l4[:, layout.POS_ROW, layout.MIRROR_COL]).mean(axis=1) >= 0.95
        idx = np.flatnonzero(mask)
        g = (rgb[idx].astype(np.float64) * LUM).sum(-1)
        imgs = l4[idx]
    else:
        # already filtered: use it directly
        imgs = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
        gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
        mask = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
                gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
        g = gray[mask]
        assert len(g) == len(imgs), f"gray rows {len(g)} vs imgs {len(imgs)}"
    X = g.reshape(len(g), -1)
    mu = X.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
    U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
    comps = Vt[:22]
    sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
    cond = ((X - mu) @ comps.T) / sd[0]
    return cond, imgs, va


# =====================================================================================
# 1. the 16-COLOUR model
# =====================================================================================
run = os.path.join(ROOT, "results", "shrink", "c16_512")
if os.path.exists(os.path.join(run, "final.pt")):
    ck = torch.load(os.path.join(run, "final.pt"), weights_only=False)
    Gc = GenColor(ck["args"]["hid"]); Gc.load_state_dict(ck["G"]); Gc.eval()
    cond, imgs, va = load_pca_and_split(colour=True)
    print(f"colour model: {ck['args']['steps']} steps, {ck['n_params']:,} params, "
          f"out 32x16, {len(cond):,} training rows")
    bv = va[:API.n]
    with torch.no_grad():
        lg = Gc(torch.tensor(cond[bv], dtype=torch.float32))
        mode = lg.argmax(-1).numpy()
        smp = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1).numpy()
    real = imgs[bv]
    gmode = layout.image_from_left_half(mode)
    gsmp = layout.image_from_left_half(smp)
    rows = [[PALETTE[real[i]] for i in range(API.n)],
            [PALETTE[gmode[i]] for i in range(API.n)],
            [PALETTE[gsmp[i]] for i in range(API.n)]]
    sh = Sheet(rows, API.cols, tile=56,
               labels=["真实彩色", "生成·取众数", "生成·采样"])
    p1 = os.path.join(OUT, "COLOR16_real_mode_sample.png")
    sh.save(p1)
    print(f"  wrote {os.path.basename(p1)}  {sh.size}")
    # a large single row of just the mode output, so quality is easy to judge
    sh2 = Sheet([rows[1]], 12, tile=72)
    p1b = os.path.join(OUT, "COLOR16_mode_big.png")
    sh2.save(p1b); print(f"  wrote {os.path.basename(p1b)}  {sh2.size}")
    # per-position accuracy, for context in the report
    acc = float((mode == imgs[bv][:, layout.POS_ROW, layout.MIRROR_COL]).mean())
    print(f"  held-out accuracy on these {API.n} faces: {acc:.4f}")

# =====================================================================================
# 2. the 32-CHANNEL continuous model
# =====================================================================================
for tag in (API.tag, "s32_gauss"):
    run = os.path.join(ROOT, "results", "shrink", tag)
    if not os.path.exists(os.path.join(run, "final.pt")):
        print(f"skip {tag}: no checkpoint")
        continue
    ck = torch.load(os.path.join(run, "final.pt"), weights_only=False)
    Gt = GenGray32(ck["args"]["hid"]); Gt.load_state_dict(ck["G"]); Gt.eval()
    cond, imgs, va = load_pca_and_split(colour=False)
    print(f"\n{tag}: {ck['args']['steps']} steps, {ck['n_params']:,} params, "
          f"out 32 continuous, lam={ck['args']['lam']} tv32={ck['args'].get('tv32')}")
    bv = va[:API.n]
    with torch.no_grad():
        raw = Gt(torch.tensor(cond[bv], dtype=torch.float32)).numpy()
    raw = np.clip(raw, 0, 15)
    disc = np.round(raw).astype(np.int64)
    real = imgs[bv]
    gdisc = layout.image_from_left_half(disc)
    # the continuous value, rendered at full 8-bit precision: this is the information the
    # 512-logit model cannot express
    cont = layout.image_from_left_half(raw)
    rows = [[LEVELS[real[i]] for i in range(API.n)],
            [LEVELS[gdisc[i]] for i in range(API.n)],
            [np.clip(cont[i] / 15.0 * 255.0, 0, 255) for i in range(API.n)]]
    sh = Sheet(rows, API.cols, tile=56,
               labels=["真实灰度(16级)", "生成·四舍五入到16级", "生成·连续值未量化"])
    p2 = os.path.join(OUT, f"{tag}_real_disc_cont.png")
    sh.save(p2)
    print(f"  wrote {os.path.basename(p2)}  {sh.size}")
    sh2 = Sheet([rows[1]], 12, tile=72)
    p2b = os.path.join(OUT, f"{tag}_disc_big.png")
    sh2.save(p2b); print(f"  wrote {os.path.basename(p2b)}  {sh2.size}")
    tgt = real[:, layout.POS_ROW, layout.MIRROR_COL]
    acc = float((disc == tgt).mean())
    mae = float(np.abs(raw - tgt).mean())
    onint = float((np.abs(raw - np.round(raw)) < 0.1).mean())
    print(f"  accuracy {acc:.4f}   MAE {mae:.3f} levels   "
          f"share of outputs within 0.1 of an integer: {onint*100:.1f}%")
    print(f"  (if that share is high, the continuous output is really just an integer in "
          f"disguise)")

print(f"\nwrote sheets to {OUT}")
print("VISUAL_DONE")
