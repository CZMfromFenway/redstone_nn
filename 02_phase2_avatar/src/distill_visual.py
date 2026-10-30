#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Look at what the distilled 32-output student actually produces.

The numbers say the distilled student reaches neighbour agreement 73.0% with accuracy
0.7348, which would beat the 512 teacher (60.7% / 0.7005) on both counts. That is
suspicious for two reasons:
  * real faces sit at 60.5% agreement, so 73% is SMOOTHER than reality -- possibly blurry
  * agreement alone does not distinguish "smooth and correct" from "smooth and washed out"

So this renders the teacher, the student and real faces side by side, and also reports a
sharpeness measure (how much detail survives) so the claim can be judged.
"""
import json, os, sys
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
OUT = os.path.join(ROOT, "results", "distill", "VISUAL")
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


class Student(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 32))
    def forward(self, c):
        return self.net(c).clamp(0, 15)


ck = torch.load(os.path.join(ROOT, "results", "coherence", "final_lam5_20k", "final.pt"),
                weights_only=False)
T = Teacher(ck["args"]["hid"], 22); T.load_state_dict(ck["G"]); T.eval()

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.POS_MIRROR_COL]).mean(1) >= 0.95 if False else \
     (gyi[:, layout.POS_ROW, layout.POS_COL] ==
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
RTRUE = torch.from_numpy(full.astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]

N = 16
bv = va[:N]
with torch.no_grad():
    lg = T.logits(COND[bv])
    p = F.softmax(lg, -1)
    expv = (p * torch.arange(NLEV, dtype=torch.float32)).sum(-1)
    t_levels = lg.argmax(-1).numpy()
    t_exp = expv.numpy()
real = full[bv].reshape(-1, 8, 8)

rows = [("真实（16级）", real.astype(float))]
rows.append(("教师 512（argmax）", layout.image_from_left_half(t_levels).astype(float)))
rows.append(("教师 512（期望值四舍五入）", layout.image_from_left_half(t_exp.round().astype(int)).astype(float)))

STUDENTS = [("distill_plain", "学生·蒸馏（无TV）"),
            ("distill_tv1", "学生·蒸馏 + TV λ=1"),
            ("distill_tv5", "学生·蒸馏 + TV λ=5")]
for tag, label in STUDENTS:
    pth = os.path.join(ROOT, "results", "distill", tag, "final.pt")
    if not os.path.exists(pth):
        print(f"skip {tag}")
        continue
    ckS = torch.load(pth, weights_only=False)
    S = Student(ckS["args"]["hid"] if "hid" in ckS["args"] else 256, 22)
    S.load_state_dict(ckS["S"]); S.eval()
    with torch.no_grad():
        sv = S(COND[bv]).numpy()
    rows.append((label, layout.image_from_left_half(sv.round().astype(int)).astype(float)))
    # also the un-rounded continuous field, to show what the model really predicts
    rows.append((label + "（连续值）", np.clip(layout.image_from_left_half(sv), 0, 15)))

# ---- draw --------------------------------------------------------------------------
TILE, PAD, LAB = 40, 6, 20
fnt = font(13)
W = len(rows[0][1]) * 0 + 8 * (TILE + PAD) + PAD
W = N * (TILE + PAD) + PAD
H = len(rows) * (TILE + PAD + LAB) + PAD + 28
cv = Image.new("RGB", (W, H), (22, 24, 28))
dr = ImageDraw.Draw(cv)
dr.text((PAD, 7), "512 教师 与 32 蒸馏学生 的实际输出", font=font(16), fill=(228, 232, 240))
for r, (label, imgs) in enumerate(rows):
    y = 28 + r * (TILE + PAD + LAB)
    for col in range(N):
        g = imgs[col]
        # continuous rows are in level units 0..15 -> scale to 0..255
        # BUG FIX: these are grey LEVELS 0..15, not pixel values 0..255. Rendering them
        # directly made every face nearly black; they must be scaled by 255/15.
        t = np.kron(np.clip(g / 15.0 * 255.0, 0, 255),
                    np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (PAD + col * (TILE + PAD), y))
    dr.text((PAD, y + TILE + 3), label, font=fnt, fill=(170, 182, 198))
cv.save(os.path.join(OUT, "DISTILL_COMPARE.png"))
print(f"wrote {os.path.join(OUT, 'DISTILL_COMPARE.png')}  {cv.size}")

# ---- sharpness: how much high-frequency detail survives ----------------------------
print()
print("  detail measure: mean absolute difference between neighbouring pixels")
print("  (higher = more texture; real faces are the reference)")
print()
print(f"  {'variant':34s} {'detail':>8s} {'agree':>8s} {'acc':>7s}")
print("  " + "-" * 62)


def detail(imgs):
    d = []
    for g in np.asarray(imgs).reshape(-1, 8, 8):
        h = np.abs(np.diff(g, axis=1)).mean()
        v = np.abs(np.diff(g, axis=0)).mean()
        d.append((h + v) / 2)
    return float(np.mean(d))


def agree_of(imgs):
    out = []
    for g in np.asarray(imgs).reshape(-1, 8, 8):
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr_, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr_, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
        out.append(same / pairs)
    return float(np.mean(out))


Rt = RTRUE[bv].numpy()
variants = [
    ("真实（16级）", real.astype(float), None),
    ("教师 512（argmax）", t_levels.astype(float), t_levels),
    ("教师 512（期望四舍五入）", t_exp.round(), t_exp.round().astype(int)),
]
for tag, label in STUDENTS:
    pth = os.path.join(ROOT, "results", "distill", tag, "final.pt")
    if not os.path.exists(pth):
        continue
    ckS = torch.load(pth, weights_only=False)
    S = Student(256, 22); S.load_state_dict(ckS["S"]); S.eval()
    with torch.no_grad():
        sv = S(COND[bv]).numpy()
    variants.append((label, layout.image_from_left_half(sv.round().astype(int)).astype(float),
                     sv.round().astype(int)))

for label, imgs, lvls in variants:
    ag = agree_of(imgs)
    acc = "" if lvls is None else f"{float((lvls == Rt).mean()):.4f}"
    print(f"  {label:34s} {detail(imgs):8.2f} {ag*100:7.1f}% {acc:>7s}")
print("DISTILL_VIS_DONE")
