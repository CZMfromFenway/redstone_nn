#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Before / after snapping each palette onto real Minecraft block colours.

Top half: the palette as optimised. Bottom half: the same palette rebuilt from actual blocks, so
every colour in the picture is one you can physically place. The cost shows up as softer edges,
because the available block colours are packed more tightly than the optimised palette.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour", "mcfit")
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


z = np.load(os.path.join(ROOT, "results", "colour", "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb96)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr = sp["train"]


def train_K(snap, steps=12000, tag=""):
    K = len(snap)
    pos = rgb96.reshape(N, 32, 3)
    dd = ((pos[:, :, None, :] - snap[None, None, :, :]) ** 2).sum(-1)
    Y = torch.tensor(dd.argmin(-1).reshape(N, 32), dtype=torch.long)
    CT = torch.tensor(cpc[:, :8])
    hid = 128
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                        nn.Linear(hid, 32 * K))
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    tr_t = torch.tensor(tr, dtype=torch.long)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        bi = tr_t[torch.from_numpy(rng.integers(0, len(tr_t), 256))]
        lg = net(CT[bi]).view(-1, 32, K)
        loss = F.cross_entropy(lg.reshape(-1, K), Y[bi].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, CT, K


def sample(net, CT, snap, n, seed):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.standard_normal((n, 8)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        idx = net(torch.tensor(Z)).view(-1, 32, len(snap)).argmax(-1).numpy()
    return snap[idx].reshape(n, 96)


def to_img(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def edge_of(flat):
    img = to_img(flat)
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    return float(np.percentile(np.concatenate([dh.ravel(), dv.ravel()]), 90))


def load_palette(fn):
    j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
    return np.array(j["palette"], dtype=np.float64)


blocks = json.load(open(os.path.join(OUT, "blocks.json"), encoding="utf-8"))
rows = []
for tag, fn in (("K=8", "colour_pal8_model.json"), ("K=16", "colour_pal16_model.json"),
                ("K=32", "colour_pal32_model.json")):
    P = load_palette(fn)
    net, CT, K = train_K(P, tag=tag + "-orig")
    a = sample(net, CT, P, 16, 3)
    rows.append((f"{tag} 优化调色板", a, edge_of(a)))
    S = np.array(blocks[tag]["palette"], dtype=np.float64)
    net2, CT2, K2 = train_K(S, tag=tag + "-snap")
    b = sample(net2, CT2, S, 16, 3)
    rows.append((f"{tag} → 方块色", b, edge_of(b)))
    print(f"  {tag}: 优化边缘 {edge_of(a):.2f} → 方块化 {edge_of(b):.2f}")

TILE, PAD = 58, 5
W = 300 + 16 * (TILE + PAD) + PAD
H = 92 + len(rows) * (TILE + 26) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "调色板方块化：每个颜色都能真的用方块摆出来", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "地图 NORMAL 最亮只有 12.94 级（没有好白）；方块色最亮 14.94 级（雪块）。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "方块化的代价是边缘变软——可用方块色比优化出的调色板更挤。", font=f_s,
        fill=(130, 142, 158))
y = 76
for name, faces, ed in rows:
    dr.text((12, y + 6), name, font=f_l,
            fill=(110, 168, 254) if "优化" in name else (110, 200, 130))
    dr.text((12, y + 22), f"边缘 {ed:.1f}", font=f_s, fill=(150, 162, 178))
    imgs = to_img(faces)
    for i in range(16):
        t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (300 + i * (TILE + PAD), y))
    y += TILE + 26
dr.text((12, H - 20), "真实数据边缘 15.56。方块化后约 12.2，优化调色板约 14.3。", font=f_s,
        fill=(140, 152, 168))
p = os.path.join(OUT, "MCBLOCK_CMP.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
print("MCBLOCK_CMP_DONE")
