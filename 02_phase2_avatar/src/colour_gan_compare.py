#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Compare every colour generator against the REAL data on the user's own criteria.

Numbers alone cannot settle which model is better, because the GAN trades reconstruction error
for other properties. So every model is placed on the same five axes, measured identically, and
the real data is one of the rows:

    saturation        lower is better for this user (their good faces 0.432, bad 0.544)
    separation        mean pairwise colour distance over the 32 positions (their best 13.5
                      on a different, cluster-centre definition; here all rows use one measure)
    edge p90          sharpness of colour boundaries (their best 14.27, bad 9.78)
    edge mean         average step between neighbouring pixels
    distinct colours  fewer is better (their best 20.4, bad 23.3)

Plus a rendered sheet, because these numbers have misled before.
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
GAN = os.path.join(OUT, "gan")
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
    """(n,96) 0..15 -> (n,8,8,3) mirrored."""
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3), dtype=np.float64)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def metrics(flat):
    img = to_imgs(flat)
    x = img.reshape(len(img), 64, 3)
    mx = x.max(-1); mn = x.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    d = np.linalg.norm(a[:, :, None, :] - a[:, None, :, :], axis=-1)
    iu = np.triu_indices(32, 1)
    sep = d[:, iu[0], iu[1]].mean(1)
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    ed = np.concatenate([dh.reshape(len(img), -1), dv.reshape(len(img), -1)], 1)
    ncol = np.array([len({tuple(np.rint(c).astype(int)) for c in a[i]}) for i in range(len(a))])
    lum = x @ np.array([0.299, 0.587, 0.114])
    llev = np.array([len(set(np.rint(l).astype(int).tolist())) for l in lum])
    return {"sat": float(sat.mean()), "sep": float(sep.mean()),
            "edge_p90": float(np.percentile(ed, 90)), "edge_mean": float(ed.mean()),
            "n_colours": float(ncol.mean()), "lum_levels": float(llev.mean())}


z = np.load(os.path.join(OUT, "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
M = 6000
sel = np.random.default_rng(3).choice(len(rgb96), M, replace=False)
REAL = rgb96[sel]
CT = cpc[sel][:, :8].astype(np.float32)


def colour_mse_model():
    j = json.load(open(os.path.join(WEB, "colour_e6_model.json"), encoding="utf-8"))
    Ls = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(Ls):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(Ls) - 1:
            seq.append(nn.ReLU())
    return nn.Sequential(*seq).eval()


def gan_model(tag):
    ck = torch.load(os.path.join(GAN, f"{tag}.pt"), weights_only=False)
    hid = ck.get("hid", 128)
    net = nn.Sequential(nn.Linear(8, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                        nn.Linear(hid, 96))
    st = {(k.replace("net.", "", 1) if k.startswith("net.") else k): v
          for k, v in ck["G"].items()}
    net.load_state_dict(st); net.eval()
    return net


GROUPS = [("真实数据", REAL)]
m6 = colour_mse_model()
with torch.no_grad():
    GROUPS.append(("MSE 纯回归（原彩色模型）", m6(torch.tensor(CT)).numpy() * 15.0))
for tag, name in (("C2", "GAN：对抗+MSE+TV5"), ("C3", "GAN：+饱和度/分离度匹配"),
                  ("C4", "GAN：TV12"), ("C5", "GAN：强饱和度匹配"),
                  ("A3", "GAN：对抗×3"), ("A5", "GAN：对抗×5"),
                  ("A3T2", "GAN：对抗×3 TV2")):
    fp = os.path.join(GAN, f"{tag}.pt")
    if not os.path.exists(fp):
        continue
    net = gan_model(tag)
    with torch.no_grad():
        GROUPS.append((name, net(torch.tensor(CT)).numpy()))

print("=" * 104)
print("ALL COLOUR GENERATORS ON THE SAME AXES  (real data is a row)")
print("=" * 104)
print(f"  {'':30s} {'饱和度':>8s} {'分离度':>8s} {'边缘p90':>9s} {'边缘均值':>9s} "
      f"{'颜色数':>7s} {'亮度级':>7s}")
print("  " + "-" * 88)
tab = []
for name, flat in GROUPS:
    m = metrics(flat)
    tab.append((name, m))
    print(f"  {name:30s} {m['sat']:8.3f} {m['sep']:8.2f} {m['edge_p90']:9.2f} "
          f"{m['edge_mean']:9.2f} {m['n_colours']:7.1f} {m['lum_levels']:7.1f}")
print()
print("  用户的标准：饱和度越低越好，分离度越高越好，边缘越锐越好，颜色数越少越好。")
print("  真实数据是参照列——模型应当在每一列上都靠近它，而不是单看某一列。")

# ---- sheet, same seeds for every model
COLS, TILE, PAD = 16, 52, 5
W = 300 + COLS * (TILE + PAD) + PAD
H = 96 + len(GROUPS) * (TILE + 26) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "彩色生成器逐行对比（同一批随机输入）", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "第一行是真实数据，只作参照。下面每行是同一批输入下各模型的输出。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "GAN 重建误差更大，但饱和度和边缘更接近真实——取舍要靠眼睛定。", font=f_s,
        fill=(130, 142, 158))
y = 76
for name, flat in GROUPS:
    m = metrics(flat)
    dr.text((12, y + 4), name[:26], font=f_l, fill=(110, 168, 254))
    dr.text((12, y + 20), f"饱和 {m['sat']:.2f}  边缘 {m['edge_p90']:.1f}", font=f_s,
            fill=(150, 162, 178))
    imgs = to_imgs(flat)
    for i in range(COLS):
        t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (300 + i * (TILE + PAD), y))
    y += TILE + 26
p = os.path.join(GAN, "COLOUR_GAN_CMP.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({n: m for n, m in tab}, open(os.path.join(GAN, "cmp.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(GAN, 'cmp.json')}")
print("GAN_CMP_DONE")
