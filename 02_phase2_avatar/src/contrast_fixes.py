#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Two candidate fixes for the washed-out faces, measured, plus a reroll-yield metric.

WHY A FIX IS NEEDED AT ALL
Measured against real faces: contrast 11.20, 9.66% washed out. The dim4 arm produces
contrast 8.72 with 18.83% washed. A face that uses four grey levels cannot serve as an
avatar pixel grid, so the low-contrast tail is a real defect for this purpose even though it
is present in the real data too.

FIX A -- CONTRAST STRETCH
   For each generated face, shift and rescale so its own min/max land on 0/15. Costs nothing
   and cannot change the face's structure, only its scale.

FIX B -- REROLL
   Reject a face whose contrast is below a threshold and draw again. This is what a user
   actually does by hand, so it is legitimate -- provided the SAME rule is applied to every
   arm. The acceptance rate then becomes an honest per-arm quality metric: an arm that needs
   three rerolls for every usable face is worse even if the survivors look the same.

Both are measured per arm, on top of the same N(0,1) sampler the page uses.
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
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "dimsweep")


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_json_model(path):
    j = json.load(open(path, encoding="utf-8"))
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


def describe(dec, tag=""):
    per = dec.reshape(-1, 32).astype(float)
    con = per.max(1) - per.min(1)
    bri = per.mean(1)
    return {"contrast": float(con.mean()), "bright": float(bri.mean()),
            "washed": float(np.mean(con <= 5)), "flat": float(np.mean(con <= 2)),
            "dark": float(np.mean(bri < 3)), "bright_tail": float(np.mean(bri > 12))}


full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
real = full[:, layout.POS_ROW, layout.MIRROR_COL]
rs = describe(real)

rng = np.random.default_rng(11)
N = 20000
MODELS = [("dim22", "pca22_dim22_model.json", 22),
          ("dim8", "pca22_dim8_model.json", 8),
          ("dim4", "pca22_dim4_model.json", 4)]

print("=" * 104)
print("FIX A: PER-FACE CONTRAST STRETCH")
print("=" * 104)
print("  map each face's own [min,max] onto [0,15], then round")
print()
print(f"  {'arm':7s} {'':9s} {'contrast':>9s} {'washed%':>8s} {'flat%':>7s} {'dark%':>7s} "
      f"{'bright%':>8s}")
print("  " + "-" * 60)
print(f"  {'real':7s} {'':9s} {rs['contrast']:9.2f} {100*rs['washed']:7.2f}% "
      f"{100*rs['flat']:6.2f}% {100*rs['dark']:6.2f}% {100*rs['bright_tail']:7.2f}%")
out = {}
THRESH = 6.0
print()
for tag, fname, k in MODELS:
    net, _ = load_json_model(os.path.join(WEB, fname))
    Z = np.clip(rng.standard_normal((N, k)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        raw = net(torch.tensor(Z)).clamp(0, 15).numpy()
    base = np.clip(np.rint(raw), 0, 15).astype(int)
    d0 = describe(base)
    # stretch: (x - min) / (max - min) * 15
    lo = raw.min(1, keepdims=True)
    hi = raw.max(1, keepdims=True)
    span = np.maximum(hi - lo, 1e-6)
    stretched = np.clip(np.rint((raw - lo) / span * 15), 0, 15).astype(int)
    d1 = describe(stretched)
    # reroll yield: share of draws whose ORIGINAL contrast already passes the threshold
    con = base.reshape(-1, 32).max(1) - base.reshape(-1, 32).min(1)
    acc = float(np.mean(con >= THRESH))
    out[tag] = {"raw": d0, "stretched": d1, "accept_rate": acc}
    print(f"  {tag:7s} {'raw':9s} {d0['contrast']:9.2f} {100*d0['washed']:7.2f}% "
          f"{100*d0['flat']:6.2f}% {100*d0['dark']:6.2f}% {100*d0['bright_tail']:7.2f}%")
    print(f"  {'':7s} {'stretched':9s} {d1['contrast']:9.2f} {100*d1['washed']:7.2f}% "
          f"{100*d1['flat']:6.2f}% {100*d1['dark']:6.2f}% {100*d1['bright_tail']:7.2f}%")

print()
print("=" * 104)
print(f"FIX B: REROLL UNTIL CONTRAST >= {THRESH:.0f}  (the acceptance rate is itself a score)")
print("=" * 104)
print(f"  {'arm':7s} {'accept rate':>12s} {'rerolls per usable face':>26s}")
print("  " + "-" * 50)
for tag in out:
    a = out[tag]["accept_rate"]
    print(f"  {tag:7s} {100*a:11.2f}% {(1/a - 1) if a > 0 else float('inf'):25.2f}")

# ---------------------------------------------------------------- visual: raw vs stretched
TILE, PAD, COLS = 36, 3, 32
rows_img = []
for tag, fname, k in MODELS:
    net, _ = load_json_model(os.path.join(WEB, fname))
    Z = np.clip(np.random.default_rng(3).standard_normal((COLS, k)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        raw = net(torch.tensor(Z)).clamp(0, 15).numpy()
    lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
    span = np.maximum(hi - lo, 1e-6)
    st = np.clip(np.rint((raw - lo) / span * 15), 0, 15).astype(int)
    rows_img.append((tag + " raw", layout.image_from_left_half(
        np.clip(np.rint(raw), 0, 15).astype(int))))
    rows_img.append((tag + " 拉伸", layout.image_from_left_half(st)))

W = 130 + COLS * (TILE + PAD) + PAD
H = 46 + len(rows_img) * (TILE + 18) + 18
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l = font(18), font(11)
dr.text((8, 10), "拉伸前 / 后（每个模型 32 张同一批输入）", font=f_t, fill=(234, 238, 244))
y = 46
for name, imgs in rows_img:
    for i, g in enumerate(imgs):
        gg = np.clip(g / 15.0 * 255.0, 0, 255)
        t = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (130 + i * (TILE + PAD), y))
    dr.text((8, y + 10), name, font=f_l, fill=(110, 168, 254))
    y += TILE + 18
p = os.path.join(OUT, "CONTRAST_FIXES.png")
cv.save(p)
print(f"\nwrote {p}")

json.dump({"real": rs, "arms": out, "threshold": THRESH},
          open(os.path.join(OUT, "contrast_fixes.json"), "w"), indent=2)
print("CONTRAST_FIXES_DONE")
