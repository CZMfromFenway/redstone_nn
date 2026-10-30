#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Render every colour approach at the same size, with the real data for reference.

The palette classifiers are the only ones that came close to the real data's edge sharpness
(12.6-13.5 against 15.56, where the continuous models sit at 9.5-10.0), so they are the ones
worth looking at. This puts them side by side with the continuous MSE model and the real art.
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
OUT = os.path.join(ROOT, "results", "colour")
PAL = os.path.join(OUT, "palette")
WEB = os.path.join(ROOT, "webapp")
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


def to_imgs(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3), dtype=np.float64)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def metrics(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = to_imgs(flat)
    x = img.reshape(len(img), 64, 3)
    mx = x.max(-1); mn = x.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    d = np.linalg.norm(a[:, :, None, :] - a[:, None, :, :], axis=-1)
    iu = np.triu_indices(32, 1)
    sep = d[:, iu[0], iu[1]].mean(1)
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    ed = np.concatenate([dh.reshape(len(img), -1), dv.reshape(len(img), -1)], 1)
    ncol = np.array([len({tuple(np.rint(c).astype(int)) for c in a[i]}) for i in range(len(a))])
    return {"sat": float(sat.mean()), "sep": float(sep.mean()),
            "edge_p90": float(np.percentile(ed, 90)), "n_colours": float(ncol.mean())}


z = np.load(os.path.join(OUT, "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
M = 4800
sel = np.random.default_rng(3).choice(len(rgb96), M, replace=False)
REAL = rgb96[sel]
CT = cpc[sel][:, :8].astype(np.float32)

GROUPS = [("真实数据（参照）", REAL)]

j = json.load(open(os.path.join(WEB, "colour_e6_model.json"), encoding="utf-8"))
Ls = j["gen"]["layers"]
seq = []
for i, L in enumerate(Ls):
    W = torch.tensor(L["W"], dtype=torch.float32); b = torch.tensor(L["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq.append(lin)
    if i != len(Ls) - 1:
        seq.append(nn.ReLU())
m6 = nn.Sequential(*seq).eval()
with torch.no_grad():
    GROUPS.append(("连续回归（原彩色模型）", m6(torch.tensor(CT)).numpy() * 15.0))

for tag, name in (("P8", "K=8 原始 k-means"), ("S8", "K=8 饱和优化"),
                  ("S16", "K=16 饱和优化"), ("S32", "K=32 饱和优化")):
    fp = os.path.join(PAL, f"{tag}.pt")
    if not os.path.exists(fp):
        continue
    ck = torch.load(fp, weights_only=False)
    K = ck["K"]; hid = ck["hid"]; C = ck["palette"]
    class PM(nn.Module):
        def __init__(self, hid, K):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(8, hid), nn.ReLU(),
                                      nn.Linear(hid, hid), nn.ReLU())
            self.head = nn.Linear(hid, 32 * K)
        def forward(self, c):
            return self.head(self.body(c)).view(-1, 32, self.K)
    PM.K = K
    net = PM(hid, K)
    net.load_state_dict(ck["state"]); net.eval()
    with torch.no_grad():
        pi = net(torch.tensor(CT)).view(-1, 32, K).argmax(-1).numpy()
    GROUPS.append((name, C[pi].reshape(M, 96)))

print("=" * 100)
print("COLOUR APPROACHES, SAME AXES, REAL DATA FOR REFERENCE")
print("=" * 100)
print(f"  {'':30s} {'饱和度':>8s} {'分离度':>8s} {'边缘p90':>9s} {'颜色数':>8s}")
print("  " + "-" * 68)
tab = []
for name, flat in GROUPS:
    m = metrics(flat)
    tab.append((name, m))
    print(f"  {name:30s} {m['sat']:8.3f} {m['sep']:8.2f} {m['edge_p90']:9.2f} "
          f"{m['n_colours']:8.1f}")
print()
print("  真实数据：饱和度 0.315  分离度 6.67  边缘 15.56  颜色数 7.2")

COLS, TILE, PAD = 16, 54, 5
W = 300 + COLS * (TILE + PAD) + PAD
H = 96 + len(GROUPS) * (TILE + 28) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "彩色方案总对比：调色板分类 vs 连续回归", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "真实脸只用 7 种颜色、边界极硬。连续输出做不到硬边界；"
        "分类到调色板索引天然就是硬边界。", font=f_s, fill=(150, 162, 178))
y = 74
for name, flat in GROUPS:
    m = metrics(flat)
    dr.text((12, y + 6), name[:26], font=f_l, fill=(110, 168, 254))
    dr.text((12, y + 22), f"边缘 {m['edge_p90']:.1f}  色 {m['n_colours']:.1f}", font=f_s,
            fill=(150, 162, 178))
    imgs = to_imgs(flat)
    for i in range(COLS):
        t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (300 + i * (TILE + PAD), y))
    y += TILE + 28
p = os.path.join(PAL, "PALETTE_SAT_CMP.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
json.dump({n: m for n, m in tab}, open(os.path.join(PAL, "cmp.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(PAL, 'cmp.json')}")
print("PALETTE_CMP_DONE")
