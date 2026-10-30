#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Evaluate the training-stage contrast remedies against the plain k=4 baseline.

Three questions, three measurements:

1. DOES IT STILL KNOW THE FACE?  Feeding the TRUE coordinates, track
   corr(predicted contrast, true contrast) and the SLOPE. A slope near 1 means the output
   reproduces the face's spread; a slope well below 1 means it shrinks toward the mean. This
   separates "cannot know" (low correlation) from "shrinks anyway" (low slope).

2. IS THE WASHED TAIL GONE?  With the same random N(0,1) draws the page uses, count the
   share whose contrast falls to 5 levels or fewer.

3. DID IT COST STRUCTURE?  Neighbour agreement and per-pixel accuracy against the target,
   so a remedy that just makes faces noisy does not look like a win.
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


def load_ckpt(path):
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


full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
X = full.reshape(len(full), -1).astype(np.float64)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
_, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
ZZ = (((X - mu) @ comps.T) / sd[0]).astype(np.float32)
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
T32 = full[:, layout.POS_ROW, layout.MIRROR_COL].astype(np.float64)
true_con = T32.max(1) - T32.min(1)
bv = va[:4000]

# (label, checkpoint tag or None, exported json or None)
VARIANTS = [("dim4 (baseline)", None, "pca22_dim4_model.json"),
            ("dim4 + 对比度损失", "dim4_contrast", None),
            ("dim4 + 拉伸目标", "dim4_stretch", None),
            ("dim8 (参考)", None, "pca22_dim8_model.json"),
            ("dim22 (参考)", None, "pca22_dim22_model.json")]

rng = np.random.default_rng(21)
N = 20000
print("=" * 108)
print("TRAINING-STAGE CONTRAST REMEDIES (all at k=4 unless marked)")
print("=" * 108)
print(f"  {'variant':20s} {'corr':>6s} {'slope':>6s} {'pred con':>9s} {'washed%':>8s} "
      f"{'flat%':>7s} {'agree':>7s} {'acc':>7s}")
print("  " + "-" * 82)
rows = []
WEB = os.path.join(ROOT, "webapp")
for label, tag, jf in VARIANTS:
    if jf:
        jp = os.path.join(WEB, jf)
        if not os.path.exists(jp):
            print(f"  {label:20s} missing json")
            continue
        net, k = load_json_model(jp)
    else:
        p = os.path.join(OUT, tag, "final.pt")
        if not os.path.exists(p):
            print(f"  {label:20s} missing")
            continue
        net, k = load_ckpt(p)
    # 1. fed the TRUE coordinates
    with torch.no_grad():
        out_true = np.clip(net(torch.tensor(ZZ[bv][:, :k])).numpy(), 0, 15)
    pc = out_true.max(1) - out_true.min(1)
    corr = float(np.corrcoef(true_con[bv], pc)[0, 1])
    slope = float(np.polyfit(true_con[bv], pc, 1)[0])
    dec_t = np.rint(out_true).astype(int)
    acc = float((dec_t == T32[bv].astype(int)).mean())
    ag = agree(layout.image_from_left_half(dec_t))
    # 2. random draws, as the page samples
    Z = np.clip(rng.standard_normal((N, k)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        out_rand = np.clip(net(torch.tensor(Z)).numpy(), 0, 15)
    con = out_rand.max(1) - out_rand.min(1)
    washed = float(np.mean(con <= 5)); flat = float(np.mean(con <= 2))
    rows.append({"label": label, "tag": tag, "json": jf, "corr": corr, "slope": slope,
                 "pred_con": float(con.mean()), "washed": washed, "flat": flat,
                 "agree": ag, "acc": acc})
    print(f"  {label:20s} {corr:6.3f} {slope:6.3f} {con.mean():9.2f} {100*washed:7.2f}% "
          f"{100*flat:6.2f}% {ag*100:6.1f}% {acc:7.4f}")

print()
print("  real faces: contrast 11.20, washed 9.66%, flat 4.36%")
print("  slope near 1 = reproduces the face's own spread; well below 1 = shrinks to the mean")

# ---------------------------------------------------------------- visual
TILE, PAD, COLS = 36, 3, 32
have = [r for r in rows]
W = 160 + COLS * (TILE + PAD) + PAD
H = 46 + len(have) * (TILE + 20) + 18
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l = font(18), font(11)
dr.text((8, 10), "对比度三种修法（每行 32 张同一批随机输入）", font=f_t, fill=(234, 238, 244))
y = 46
for r in have:
    if r.get("json"):
        net, k = load_json_model(os.path.join(WEB, r["json"]))
    else:
        net, k = load_ckpt(os.path.join(OUT, r["tag"], "final.pt"))
    Z = np.clip(np.random.default_rng(4).standard_normal((COLS, k)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        dec = np.rint(np.clip(net(torch.tensor(Z)).numpy(), 0, 15)).astype(int)
    for i, g in enumerate(layout.image_from_left_half(dec)):
        gg = np.clip(g / 15.0 * 255.0, 0, 255)
        t = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (160 + i * (TILE + PAD), y))
    dr.text((8, y + 6), r["label"], font=f_l, fill=(110, 168, 254))
    dr.text((8, y + 20), f"发灰 {100*r['washed']:.1f}%", font=f_l, fill=(130, 142, 158))
    y += TILE + 20
p = os.path.join(OUT, "CONTRAST_REMEDIES.png")
cv.save(p)
print(f"\nwrote {p}")

json.dump(rows, open(os.path.join(OUT, "contrast_remedies.json"), "w"), indent=2)
print("CONTRAST_REMEDIES_DONE")
