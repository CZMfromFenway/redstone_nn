#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Why do so many samples look washed out? Measure contrast instead of guessing.

THE REPORT
The user says many faces look low-contrast ("overall grey") or overall too bright/dark, and
that this did not happen while labelling the earlier models. A low-contrast output is the
signature of regression to the conditional mean: the MSE optimum given an under-specified
condition is the average face, and the average face is neither bright nor dark and has
little spread.

So the measurement is per-face CONTRAST (max level - min level) and per-face MEAN, compared
against real faces, and against the earlier w128 model which the user labelled before and
did not complain about.

If contrast collapses as the input dimension shrinks, that confirms the conditional-mean
explanation. If even k=22 collapses, the cause is elsewhere.
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
os.makedirs(OUT, exist_ok=True)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)

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


def load_json_model(path):
    """Load an exported model the same way the page does: layers of {W, b}, ReLU between."""
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
    return m, int(layers[0]["W"][0].__len__()), j.get("meta", {})


def metrics(dec):
    """dec: (N,32) integer levels."""
    per = dec.reshape(-1, 32).astype(float)
    return {
        "contrast": per.max(1) - per.min(1),
        "mean": per.mean(1),
        "std": per.std(1),
    }


# real faces as the reference distribution
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
real = full[:, layout.POS_ROW, layout.MIRROR_COL]
mr = metrics(real)
print("=" * 100)
print("CONTRAST AND BRIGHTNESS: generated vs real")
print("=" * 100)
print(f"  real faces (n={len(real):,}): contrast mean {mr['contrast'].mean():5.2f} "
      f"median {np.median(mr['contrast']):4.1f} | brightness mean {mr['mean'].mean():5.2f} "
      f"| flat (contrast<=2) {100*np.mean(mr['contrast']<=2):5.2f}%")
print()

MODELS = [
    ("w128 (labelled before, no complaint)", "pca22_w128_128_model.json"),
    ("dim22 (same recipe, k=22)", "pca22_dim22_model.json"),
    ("dim8  (k=8)", "pca22_dim8_model.json"),
    ("dim4  (k=4)", "pca22_dim4_model.json"),
]
rng = np.random.default_rng(2024)
N = 4000
rows = []
print(f"  {'model':38s} {'k':>3s} {'contrast':>9s} {'median':>7s} {'bright':>7s} "
      f"{'flat%':>7s} {'washed%':>8s}")
print("  " + "-" * 88)
for label, fname in MODELS:
    p = os.path.join(WEB, fname)
    if not os.path.exists(p):
        print(f"  {label:38s} missing")
        continue
    net, k, meta = load_json_model(p)
    Z = np.clip(rng.standard_normal((N, k)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        raw = net(torch.tensor(Z))
    # both exports are continuous32: round to the nearest level
    dec = np.clip(np.rint(raw.numpy()), 0, 15).astype(int)
    m = metrics(dec)
    flat = float(np.mean(m["contrast"] <= 2))
    washed = float(np.mean(m["contrast"] <= 5))
    rows.append({"label": label, "k": k, "contrast": float(m["contrast"].mean()),
                 "median": float(np.median(m["contrast"])),
                 "bright": float(m["mean"].mean()), "flat": flat, "washed": washed,
                 "dec": dec})
    print(f"  {label:38s} {k:3d} {m['contrast'].mean():9.2f} "
          f"{np.median(m['contrast']):7.1f} {m['mean'].mean():7.2f} "
          f"{100*flat:6.2f}% {100*washed:7.2f}%")

print()
print("  flat   = contrast <= 2 levels (essentially one shade)")
print("  washed = contrast <= 5 levels (uses under a third of the range)")
print(f"  real faces: flat {100*np.mean(mr['contrast']<=2):.2f}%, "
      f"washed {100*np.mean(mr['contrast']<=5):.2f}%")

# does the input brightness axis actually drive output brightness?
print()
print("=" * 100)
print("DOES THE INPUT STILL CONTROL BRIGHTNESS?")
print("=" * 100)
print("  Correlating the output's mean level with PC1 of the input (the brightness axis).")
for r in rows:
    p = os.path.join(WEB, next(f for l, f in MODELS if l == r["label"]))
    net, k, meta = load_json_model(p)
    if k < 1:
        continue
    Z = np.clip(rng.standard_normal((N, k)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        dec = np.clip(np.rint(net(torch.tensor(Z)).numpy()), 0, 15)
    corr = float(np.corrcoef(Z[:, 0], dec.mean(1))[0, 1])
    print(f"  {r['label']:38s} corr(PC1, brightness) = {corr:+.3f}")

# ---------------------------------------------------------------- visual
if rows:
    TILE, PAD = 40, 3
    COLS = 32
    Wd = 140 + COLS * (TILE + PAD) + PAD
    Hd = 46 + len(rows) * (TILE + 24) + 20
    cv = Image.new("RGB", (Wd, Hd), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    f_t, f_l, f_s = font(18), font(12), font(8)
    dr.text((8, 10), "每个模型的 32 张随机样本 · 看对比度是否被压扁", font=f_t, fill=(234, 238, 244))
    dr.text((8, 32), "真实脸的对比度均值约 9.6 级", font=f_s, fill=(150, 162, 178))
    y = 46
    for r in rows:
        imgs = layout.image_from_left_half(r["dec"][:COLS])
        for i, g in enumerate(imgs):
            gg = np.clip(g / 15.0 * 255.0, 0, 255)
            t = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
            cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                     (140 + i * (TILE + PAD), y))
        dr.text((8, y + 8), r["label"].split(" (")[0], font=f_l, fill=(110, 168, 254))
        dr.text((8, y + 22), f"对比 {r['contrast']:.1f}", font=f_s, fill=(130, 142, 158))
        y += TILE + 24
    p = os.path.join(OUT, "CONTRAST_DIAGNOSIS.png")
    cv.save(p)
    print(f"\nwrote {p}")

json.dump([{k: v for k, v in r.items() if k != "dec"} for r in rows],
          open(os.path.join(OUT, "contrast.json"), "w"), indent=2)
print("CONTRAST_DONE")
