#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Render the trained variants so the numbers can be judged visually.

The predicted accept rate is NOT trustworthy on its own: the probe is fitted on row spread, eye
contrast, mean neighbour difference and colour count, and C1 optimises the first two of those
directly. A high probe score therefore mostly proves that the optimiser hit its own target. Only
the picture can say whether the faces still read as faces.
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
OUT = os.path.join(ROOT, "results", "colour", "yield")
WEB = os.path.join(ROOT, "webapp")
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114])


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_ck(fn):
    ck = torch.load(os.path.join(OUT, fn), weights_only=False)
    st = ck["G"]; pal = np.array(ck["palette"])
    K = len(pal)
    idx = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    seq = []
    for j in idx:
        seq.append(nn.Linear(int(st[f"{j}.weight"].shape[1]), int(st[f"{j}.weight"].shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.load_state_dict(st); m.eval()
    return m, pal, K


def gen(m, pal, K, Z):
    with torch.no_grad():
        i = m(torch.tensor(Z)).view(-1, 32, K).argmax(-1).numpy()
    return pal[i].reshape(len(Z), 96)


def to_img(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


VARIANTS = [
    ("基线（无结构损失）", "ck_c1_0.0_0.pt"),
    ("C1 结构损失 λ=1", "ck_c1_1.0_1.pt"),
    ("C1 结构损失 λ=3", "ck_c1_3.0_2.pt"),
    ("C2 对抗 λ=0.3", "ck_c2_1.0_3.pt"),
]

# the shipped k8snap model, as the current best known good reference
snap = json.load(open(os.path.join(WEB, "colour_k8snap_model.json"), encoding="utf-8"))
PAL_SNAP = np.array(snap["palette"])
layers = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True} for L in snap["body"]]
layers.append({"W": np.array(snap["head"]["W"]), "b": np.array(snap["head"]["b"]),
               "relu": False})


def gen_snap(Z):
    v = Z
    for L in layers:
        v = v @ L["W"].T + L["b"]
        if L["relu"]:
            v = np.maximum(0, v)
    i = v.reshape(len(Z), 32, len(PAL_SNAP)).argmax(-1)
    return PAL_SNAP[i].reshape(len(Z), 96)


M = 20
rng = np.random.default_rng(21)
Z = np.clip(rng.standard_normal((M, 8)), -3.5, 3.5).astype(np.float32)

rows = [("k8snap 现役（训练后吸附）", gen_snap(Z))]
for name, fn in VARIANTS:
    if not os.path.exists(os.path.join(OUT, fn)):
        continue
    m, pal, K = load_ck(fn)
    rows.append((name, gen(m, pal, K, Z)))

TILE, PAD = 48, 5
W = 320 + M * (TILE + PAD) + 5
H = 104 + len(rows) * (TILE + 24) + 20
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "C1 / C2：结构损失与对抗损失的效果", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "同一批随机输入，逐列可比。第一行是现役方案作为参照。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "预测良率不可信——探针用的正是 C1 优化的量。只看图。", font=f_s,
        fill=(224, 192, 90))
y = 80
for name, faces in rows:
    dr.text((12, y + 6), name, font=f_l,
            fill=(110, 200, 130) if "k8snap" in name else (110, 168, 254))
    imgs = to_img(faces)
    for i in range(M):
        t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (320 + i * (TILE + PAD), y))
    y += TILE + 24
p = os.path.join(OUT, "C1C2_CMP.png")
cv.save(p)
print(f"wrote {p}  {cv.size}")
for name, faces in rows:
    a = np.clip(faces, 0, 15).reshape(len(faces), 32, 3)
    lu = a @ LUM
    rows_m = np.stack([lu[:, [c * 8 + r for c in range(4)]].mean(1) for r in range(8)], 1)
    print(f"  {name:22s} 行方差 {rows_m.std(1).mean():.2f}  "
          f"眼睛对比 {(np.median(lu[:, [c*8+r for r in (5,6) for c in range(4)]],1) - lu[:, [c*8+r for r in (3,4) for c in (1,2)]].min(1)).mean():5.2f}")
print("C1C2_FIG_DONE")
