#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Show the effect of the w128 preference model as a picture.

Left  : 24 faces drawn the plain way, z ~ N(0,1)^22
Right : 24 faces drawn from the top 15% of a 600-candidate pool ranked by the reward model
        fitted on the user's 111 labels

Measured on held-out labels, keeping the top 15% raised the 优+良 rate from 40.7% to 87.5%.
This renders the two samplers so the improvement can be judged rather than just quoted.

It also reports the mean predicted score and the neighbour agreement of each side, so the
"better" claim is backed by numbers as well as by the eye.
"""
import json, os, sys
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
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)
OUT = os.path.join(ROOT, "results", "preference")
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


M = load_model(os.path.join(ROOT, "results", "shrink5", "ladder_128_128.pt"))
R = json.load(open(os.path.join(ROOT, "webapp", "preference_reward.json"), encoding="utf-8"))
print(f"reward model: fitted on {R.get('model')} with {R['n_labels']} labels, "
      f"cv AUC {R.get('cv_auc', float('nan')):.3f}")
w = np.array(R["w"], dtype=np.float64)
b = float(R["b"])
zm = np.array(R["zmean"]); zsd = np.array(R["zsd"])


def score(Z):
    return ((Z - zm) / zsd) @ w + b


def render(Z):
    with torch.no_grad():
        out = M(torch.tensor(Z, dtype=torch.float32)).clamp(0, 15)
    dec = out.round().numpy().astype(int)
    return layout.image_from_left_half(dec), dec


def agree_of(imgs):
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


N, POOL, KEEP = 28, 600, 0.15
rng = np.random.default_rng(0)

# plain sampler
Zrand = np.clip(rng.standard_normal((N, 22)), -3.5, 3.5)
# filtered sampler: for each keep, draw a pool and take the best
Zfilt = np.empty((N, 22))
for i in range(N):
    P = np.clip(rng.standard_normal((POOL, 22)), -3.5, 3.5)
    s = score(P)
    k = max(1, int(POOL * KEEP))
    Zfilt[i] = P[np.argsort(-s)[:k][rng.integers(0, k)]]

imgR, decR = render(Zrand)
imgF, decF = render(Zfilt)

print()
print(f"  {'sampler':22s} {'mean predicted score':>21s} {'neighbour agreement':>21s}")
print("  " + "-" * 66)
print(f"  {'plain N(0,1)':22s} {score(Zrand).mean():21.3f} {agree_of(imgR)*100:20.1f}%")
print(f"  {'top-15% filtered':22s} {score(Zfilt).mean():21.3f} {agree_of(imgF)*100:20.1f}%")
print(f"  gain: {score(Zfilt).mean()-score(Zrand).mean():+.3f} score")

# ---- draw -------------------------------------------------------------------------
COLS = 14
TILE = 56
PAD = 8
LAB = 24
fnt = font(15)
W = COLS * (TILE + PAD) + PAD
H = 2 * (TILE + PAD + LAB) + PAD + 30
cv = Image.new("RGB", (W, H), (22, 24, 28))
dr = ImageDraw.Draw(cv)
dr.text((PAD, 8), f"w128 模型的打标效果：随机 vs 奖励模型筛选（保留前 {KEEP*100:.0f}%）",
        font=font(17), fill=(230, 234, 242))
rows = [
    (f"随机采样 · 平均预测分 {score(Zrand).mean():.2f}", imgR),
    (f"筛选后 · 平均预测分 {score(Zfilt).mean():.2f}", imgF),
]
for r, (label, imgs) in enumerate(rows):
    y = 36 + r * (TILE + PAD + LAB)
    for col in range(min(COLS, len(imgs))):
        g = np.clip(imgs[col] / 15.0 * 255.0, 0, 255)
        t = np.kron(g, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (PAD + col * (TILE + PAD), y))
    dr.text((PAD, y + TILE + 4), label, font=fnt, fill=(175, 186, 200))
p = os.path.join(OUT, "W128_BEFORE_AFTER.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
print("BEFORE_AFTER_DONE")
