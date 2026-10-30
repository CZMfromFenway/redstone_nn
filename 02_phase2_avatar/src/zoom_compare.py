#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Zoomed visual comparison: does the fragmentation actually show?

The numbers say the 23k-multiply student breaks a face into 21.5 same-level regions where
real faces have 13.8 and the 512 teacher has 13.9. A count of regions does not tell you
whether a viewer would notice, so this renders the same faces at 3x zoom, real vs teacher
vs student, for direct comparison.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
OUT = os.path.join(ROOT, "results", "fragment2", "VISUAL")
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


class Teacher(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


ck = torch.load(os.path.join(ROOT, "results", "coherence", "final_lam5_20k", "final.pt"),
                weights_only=False)
T = Teacher(ck["args"]["hid"], 22); T.load_state_dict(ck["G"]); T.eval()

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U_, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
COND = torch.tensor(((X - mu) @ comps.T) / sd[0], dtype=torch.float32)
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]

N = 8
bv = va[:N]
real = full[bv].reshape(-1, 8, 8)
with torch.no_grad():
    t_arg = T.logits(COND[bv]).argmax(-1).numpy()
rows = [("真实", real)]
rows.append(("教师 512", layout.image_from_left_half(t_arg)))


def load_model(path):
    c = torch.load(path, weights_only=False)
    st = c.get("G") or c.get("S")
    norm = {k.replace("net.", "", 1) if k.startswith("net.") else k: v
            for k, v in st.items()}
    idx = sorted({int(k.split(".")[0]) for k in norm if k.endswith(".weight")})
    seq = []
    for j in idx:
        W = norm[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq)
    m.load_state_dict(norm)
    m.eval()
    return m


for path, label in (
        ("results/fragment2/tv_cont_0p1_(control)/final.pt", "学生 22→128→128→32 (23k 乘法)"),
        ("results/shrink5/ladder_32.pt", "学生 22→32→32 (1.7k 乘法)"),
):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        print(f"skip {label}")
        continue
    m = load_model(p)
    with torch.no_grad():
        sv = m(COND[bv]).clamp(0, 15).numpy()
    rows.append((label, layout.image_from_left_half(sv.round().astype(int))))

TILE = 96
PAD = 8
LAB = 22
fnt = font(13)
W = N * (TILE + PAD) + PAD
H = len(rows) * (TILE + PAD + LAB) + PAD + 26
cv = Image.new("RGB", (W, H), (22, 24, 28))
dr = ImageDraw.Draw(cv)
dr.text((PAD, 6), "放大 3 倍的直接对比：碎片化看得出来吗？", font=font(16), fill=(228, 232, 240))
for r, (label, imgs) in enumerate(rows):
    y = 26 + r * (TILE + PAD + LAB)
    for col in range(N):
        g = np.clip(imgs[col] / 15.0 * 255.0, 0, 255)
        t = np.kron(g, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (PAD + col * (TILE + PAD), y))
    dr.text((PAD, y + TILE + 3), label, font=fnt, fill=(170, 182, 198))
cv.save(os.path.join(OUT, "ZOOM_COMPARE.png"))
print(f"wrote {os.path.join(OUT, 'ZOOM_COMPARE.png')}  {cv.size}")
print("ZOOM_DONE")
