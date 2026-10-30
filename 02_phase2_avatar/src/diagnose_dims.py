#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Diagnose the low-k models before anyone labels them.

WHY THIS EXISTS
The k=4 and k=8 runs reported accuracy 0.053-0.057, which is BELOW the 1/16 = 0.0625
chance rate for matching a level, while reporting neighbour agreement of 78-80%, close to
real faces (77.6%). That combination is the signature of a degenerate solution: an output
that is smooth but unrelated to the request.

Accuracy alone cannot separate the two ways that can happen:
    (a) the model outputs ONE fixed face regardless of input (ignores the condition)
    (b) the model outputs smooth faces that vary with input but miss the target
So the decisive measurement is: hold the setup fixed and see whether the output MOVES when
the input changes.

Metrics reported per model
    spatial std      within-image spread; near 0 means a flat field
    input response   mean per-position std of the output ACROSS different inputs.
                     Near 0 means the model ignores its condition entirely -- case (a).
    distinct levels  how many of the 16 levels the output uses at all
    acc / agree      for comparison with the reference (s32_tv5: 0.7594 / 0.517)
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

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)
SRC = os.path.join(ROOT, "results", "dimsweep")
REF = os.path.join(ROOT, "results", "shrink", "s32_tv5")
OUT = os.path.join(ROOT, "results", "dimsweep")
os.makedirs(OUT, exist_ok=True)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_net(path):
    ck = torch.load(path, weights_only=False)
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


def agree(imgs):
    out = []
    for g in imgs:
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
        out.append(same / pairs)
    return float(np.mean(out))


# data
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.clip(np.rint(gray / 17.0), 0, 15).astype(np.int64)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
_, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sdv = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
ZZ = ((X - mu) @ comps.T) / sdv[0]
RTRUE = full[:, layout.POS_ROW, layout.MIRROR_COL]

MODELS = [("dim4", os.path.join(SRC, "dim4", "final.pt")),
          ("dim8", os.path.join(SRC, "dim8", "final.pt")),
          ("dim22", os.path.join(REF, "final.pt"))]

print("=" * 104)
print("WHY THE LOW-k MODELS LOOK WRONG")
print("=" * 104)
print(f"  {'model':7s} {'k':>3s} {'acc':>7s} {'agree':>7s} {'spatial sd':>11s} "
      f"{'input resp':>11s} {'levels used':>12s}")
print("  " + "-" * 62)
rows = []
N = 1000
bv = va[:N]
rng = np.random.default_rng(5)
RAND = np.clip(rng.standard_normal((512, 22)), -3.5, 3.5)
for tag, path in MODELS:
    if not os.path.exists(path):
        print(f"  {tag:7s} missing checkpoint")
        continue
    net, k = load_net(path)
    with torch.no_grad():
        out_true = net(torch.tensor(ZZ[bv][:, :k].astype(np.float32))).clamp(0, 15).numpy()
        out_rand = net(torch.tensor(RAND[:, :k].astype(np.float32))).clamp(0, 15).numpy()
    dec = out_true.round().astype(int)
    acc = float((dec == RTRUE[bv]).mean())
    ag = agree(layout.image_from_left_half(dec))
    spatial_sd = float(out_true.std(axis=1).mean())
    input_resp = float(out_rand.std(axis=0).mean())
    lv = len(set(out_rand.round().astype(int).reshape(-1).tolist()))
    rows.append({"tag": tag, "k": k, "acc": acc, "agree": ag,
                 "spatial_sd": spatial_sd, "input_resp": input_resp, "levels": lv})
    print(f"  {tag:7s} {k:3d} {acc:7.4f} {ag*100:6.1f}% {spatial_sd:11.3f} "
          f"{input_resp:11.3f} {lv:12d}")

print()
print("  reading: 'input resp' near 0 => the model ignores its condition (case a).")
print("           'input resp' healthy but acc low => smooth faces that miss (case b).")
print("           'spatial sd' near 0 => a flat field, which inflates agreement.")
print(f"  reference s32_tv5 at 12000 steps: acc 0.7594, agree 51.7%")

# visual
if rows:
    TILE, PAD = 34, 3
    COLS = 32
    W = 120 + COLS * (TILE + PAD) + PAD
    H = 50 + len(rows) * (TILE + 22) + 20
    cv = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    f_t, f_l, f_s = font(17), font(11), font(8)
    dr.text((6, 8), "低维模型的输出诊断（32 张随机输入）", font=f_t, fill=(232, 236, 242))
    dr.text((6, 30), "如果每一行看起来都一样，说明模型忽略了输入", font=f_s, fill=(150, 162, 178))
    y = 50
    for r in rows:
        net, k = load_net(next(p for t, p in MODELS if t == r["tag"]))
        with torch.no_grad():
            o = net(torch.tensor(RAND[:COLS, :k].astype(np.float32))).clamp(0, 15).numpy()
        imgs = layout.image_from_left_half(o.round().astype(int))
        for i, g in enumerate(imgs):
            gg = np.clip(g / 15.0 * 255.0, 0, 255)
            t = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
            cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                     (120 + i * (TILE + PAD), y))
        dr.text((6, y + 4), r["tag"], font=f_l, fill=(110, 168, 254))
        dr.text((6, y + 16), f"acc {r['acc']:.3f}", font=f_s, fill=(130, 142, 158))
        y += TILE + 22
    p = os.path.join(OUT, "DIM_DIAGNOSIS.png")
    cv.save(p)
    print(f"\nwrote {p}")
json.dump(rows, open(os.path.join(OUT, "diagnosis.json"), "w"), indent=2)
print("DIAGNOSE_DONE")
