#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
All the adversarial variants in one picture, with the diagnostic that predicts the verdict.

THE ARGUMENT THIS FIGURE MAKES
Round 12 measured the soft-adversarial model at 43.2% against 61.1% for the untouched baseline.
The hard-discriminator variants differ in how strongly the discriminator dominates, and the
spectral-normalised one sits at D accuracy 0.867 -- statistically the same as the 0.863 of the
already-tested soft run. If D accuracy is the thing that tracks the damage, then that variant
needs no separate blind test to be predicted.

That is a claim, so it is stated as one and the numbers are left visible for checking.
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
EXP = os.path.join(ROOT, "results", "colour", "expand")
HARD = os.path.join(ROOT, "results", "colour", "hardd")
OUT = os.path.join(ROOT, "results", "colour", "yield")
torch.set_num_threads(8)

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


def load(d, tag):
    p = os.path.join(d, f"{tag}.pt")
    if not os.path.exists(p):
        return None
    ck = torch.load(p, weights_only=False)
    st = ck["G"]; pal = np.array(ck["palette"]); K = len(pal)
    idx = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    seq = []
    for j in idx:
        seq.append(nn.Linear(int(st[f"{j}.weight"].shape[1]), int(st[f"{j}.weight"].shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.load_state_dict(st); m.eval()
    return m, pal, K


ROWS = [
    ("基线 base48k（无对抗，评分 61.1%）", EXP, "base48k", (110, 200, 130)),
    ("软对抗 adv48k（评分 43.2%）", EXP, "adv48k", (255, 150, 120)),
    ("硬对抗 T=0.1", HARD, "st_T010", (255, 130, 130)),
    ("硬对抗 T=0.03", HARD, "st_T003", (255, 165, 100)),
    ("硬对抗 + 谱归一化", HARD, "st_sn", (240, 200, 110)),
]

M = 16
Z = np.clip(np.random.default_rng(303).standard_normal((M, 8)), -3.5, 3.5).astype(np.float32)
rows = []
DACC = {"base48k": None, "adv48k": 0.863, "st_T010": 0.988, "st_T003": 0.990, "st_sn": 0.867}
for name, d, tag, col in ROWS:
    got = load(d, tag)
    if got is None:
        continue
    m, pal, K = got
    with torch.no_grad():
        i = m(torch.tensor(Z)).view(-1, 32, K).argmax(-1).numpy()
    faces = pal[i].reshape(len(Z), 96)
    a = np.clip(faces, 0, 15).reshape(len(Z), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    rows.append((name, img, col, DACC.get(tag)))

TILE, PAD = 48, 4
W = 360 + M * (TILE + PAD)
H = 110 + len(rows) * (TILE + 24) + 26
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "对抗的三种做法，与不做对抗的对照", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "同一批输入，逐列可比。右边数字是判别器的最终准确率。", font=f_s,
        fill=(150, 162, 178))
dr.text((12, 54), "D 越接近 1，生成器越是被推开去追高对比与怪颜色，而不是变好看。", font=f_s,
        fill=(224, 192, 90))
y = 82
for name, img, col, dacc in rows:
    dr.text((12, y + 6), name, font=f_l, fill=col)
    dr.text((12, y + 22), f"D 准确率 {dacc:.3f}" if dacc else "无判别器", font=f_s,
            fill=(150, 162, 178))
    for i in range(M):
        t = np.kron(np.clip(img[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (360 + i * (TILE + PAD), y))
    y += TILE + 24
p = os.path.join(OUT, "ADV_ALL.png")
cv.save(p)
print(f"wrote {p}  {cv.size}")
print()
print("  D 准确率对照")
for name, _, _, dacc in rows:
    print(f"    {name:36s} {'—' if dacc is None else f'{dacc:.3f}'}")
print("ADV_ALL_DONE")
