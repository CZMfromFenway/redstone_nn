#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
The three K=8 colour schemes side by side, with the metrics the user judges on.

k8free and k8snap share weights, so any difference is purely the palette.
k8mc has its own weights, so it can differ in both palette and shape.
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
OUT = os.path.join(ROOT, "results", "colour", "k8schemes")
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


def load(fn):
    j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
    pal = np.array(j["palette"], dtype=np.float64)
    K = len(pal)
    all = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True} for L in j["body"]]
    all.append({"W": np.array(j["head"]["W"]), "b": np.array(j["head"]["b"]), "relu": False})

    def gen(Z):
        v = Z
        for L in all:
            v = v @ L["W"].T + L["b"]
            if L["relu"]:
                v = np.maximum(0, v)
        return pal[v.reshape(len(Z), 32, K).argmax(-1)].reshape(len(Z), 96)
    return gen, pal, j["meta"]


def to_img(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def metrics(flat):
    img = to_img(flat)
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
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
            "edge": float(np.percentile(ed, 90)), "ncol": float(ncol.mean()),
            "lum": float((a @ LUM).mean())}


ARMS = [("k8free 自由拟合", "colour_pal8_model.json"),
        ("k8snap 事后吸附", "colour_k8snap_model.json"),
        ("k8mc 事前约束", "colour_k8mc_model.json")]

M = 4800
rng = np.random.default_rng(5)
Z = np.clip(rng.standard_normal((M, 8)), -3.5, 3.5).astype(np.float32)

print("=" * 100)
print("THREE K=8 SCHEMES")
print("=" * 100)
print(f"  {'arm':18s} {'K':>3s} {'乘法':>9s} {'饱和':>7s} {'分离度':>7s} {'边缘':>7s} {'颜色数':>7s}")
print("  " + "-" * 68)
rows = []
for name, fn in ARMS:
    gen, pal, meta = load(fn)
    faces = gen(Z)
    m = metrics(faces)
    print(f"  {name:18s} {len(pal):3d} {meta['mults']:9,} {m['sat']:7.3f} {m['sep']:7.2f} "
          f"{m['edge']:7.2f} {m['ncol']:7.1f}")
    rows.append((name, faces, m))
print(f"\n  真实数据参照：饱和 0.316，分离度 6.67，边缘 15.56，颜色数 7.2")

COLS, TILE = 20, 48
W = 300 + COLS * (TILE + 5) + 5
H = 100 + len(rows) * (TILE + 26) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "K=8 的三种配色方案", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "k8free 与 k8snap 是同一批权重，只换调色板；k8mc 是事前把调色板限定在 MC 颜色后重训的。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "同一批随机输入，逐列可比。", font=f_s, fill=(130, 142, 158))
y = 78
for name, faces, m in rows:
    dr.text((12, y + 6), name, font=f_l, fill=(110, 168, 254))
    dr.text((12, y + 22), f"边缘 {m['edge']:.1f} · 色 {m['ncol']:.0f}", font=f_s,
            fill=(150, 162, 178))
    imgs = to_img(faces[:COLS])
    for i in range(COLS):
        t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (300 + i * (TILE + 5), y))
    y += TILE + 26
p = os.path.join(OUT, "K8_SCHEMES.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
json.dump({n: m for n, _, m in rows}, open(os.path.join(OUT, "metrics.json"), "w",
                                          encoding="utf-8"), indent=2, ensure_ascii=False)
print("K8_FIG_DONE")
