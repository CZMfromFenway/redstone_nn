#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Evaluate the colour models and draw the comparison sheets.

TWO DIFFERENT PICTURES, BOTH NEEDED
  1. RECONSTRUCTION -- feed the true 4 PCs, compare the produced colour against the real face.
     This isolates stage 2's error from stage 1's.
  2. GENERATION -- feed RANDOM 4 PCs, i.e. the actual product: a colour face from noise.
     A model can reconstruct well and still generate badly, so both are shown.

The colour images are built by scattering the 32 predicted RGB triples to the canonical
positions and mirroring within each row, exactly as layout.py does for grayscale.
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
WEB = os.path.join(ROOT, "webapp")

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


def load_gray():
    j = json.load(open(os.path.join(WEB, "pca22_dim4x_model.json"), encoding="utf-8"))
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
    return nn.Sequential(*seq).eval()


def gray_path(net, pcs):
    with torch.no_grad():
        v = torch.clamp(net(torch.tensor(pcs, dtype=torch.float32)), 0, 15)
        lo = v.min(1, keepdim=True)[0]; hi = v.max(1, keepdim=True)[0]
        v = torch.clamp((v - lo) / (hi - lo).clamp_min(1e-6) * 15, 0, 15)
        v = torch.round(v)
    return v.numpy()


def rgb_image(rgb96):
    """(n,96) 0..15 -> (n,8,8,3) uint8, mirrored within each row."""
    a = np.clip(rgb96, 0, 15)
    a = a.reshape(-1, 32, 3)
    img = np.zeros((len(a), 8, 8, 3), dtype=np.float32)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return np.clip(img / 15.0 * 255.0, 0, 255).astype(np.uint8)


z = np.load(os.path.join(OUT, "cache.npz"))
gray32 = z["gray32"].astype(np.float32)
pc4 = z["pc4"]
rgb96 = z["rgb96"].astype(np.float32)
cpc = z["cpc"]
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
va = sp["val"]
results = json.load(open(os.path.join(OUT, "stage2_results.json"), encoding="utf-8"))
gnet = load_gray()
G_pred = gray_path(gnet, pc4)
G_IN = (G_pred / 15.0).astype(np.float32)

ORDER = [k for k in ("E1", "E2", "E3", "E4", "E5", "E6") if k in results]
print("variants:", ORDER)


def model_input(tag, idx, pcs_for_pc4):
    m = torch.load(os.path.join(OUT, f"stage2_{tag}.pt"), weights_only=False)
    sch, k = m["scheme"], m["k"]
    if sch == "A":
        x = G_IN[idx] if k == 0 else np.concatenate([G_IN[idx], cpc[idx][:, :k]], 1)
    elif sch == "B":
        x = np.concatenate([pcs_for_pc4, cpc[idx][:, :k]], 1)
    else:
        x = cpc[idx][:, :k]
    net = nn.Sequential(nn.Linear(m["d_in"], m["hid"]), nn.ReLU(),
                        nn.Linear(m["hid"], m["hid"]), nn.ReLU(), nn.Linear(m["hid"], 96))
    net.load_state_dict(m["state"]); net.eval()
    with torch.no_grad():
        return net(torch.tensor(x.astype(np.float32))).numpy() * 15.0, net, sch, k


# ================================================================ 1. reconstruction sheet
COLS = 10
TILE = 44
idx = va[:COLS]
true_img = rgb_image(rgb96[idx])
rows = [("真实", true_img)]
for tag in ORDER:
    p, _, _, _ = model_input(tag, idx, pc4[idx])
    rows.append((f"{tag} {results[tag]['note']}", rgb_image(p)))

W = 210 + COLS * (TILE + 4) + 4
H = 92 + len(rows) * (TILE + 30) + 26
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((10, 10), "彩色重建：给真实的 4 个 PCA，比较产出的颜色", font=f_t, fill=(234, 238, 244))
dr.text((10, 36), f"验证集 {COLS} 张。RGB 每通道 4 bit。4 bit 量化下限 = 0.205 级。",
        font=f_s, fill=(150, 162, 178))
dr.text((10, 52), "第一行是真实颜色；下面每行是同一批脸的模型预测。", font=f_s,
        fill=(130, 142, 158))
y = 76
for name, imgs in rows:
    dr.text((10, y + 4), name[:26], font=f_l, fill=(110, 168, 254))
    if name != "真实":
        tag = name.split()[0]
        if tag in results:
            dr.text((10, y + 20), f"MAE {results[tag]['rgb_mae']:.2f}", font=f_s,
                    fill=(150, 162, 178))
            dr.text((10, y + 32), f"色相 {results[tag]['hue_err']:.3f}", font=f_s,
                    fill=(150, 162, 178))
    for i, im in enumerate(imgs):
        tile = np.kron(im, np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(tile), (210 + i * (TILE + 4), y))
    y += TILE + 30
p1 = os.path.join(OUT, "CMP_STAGE2.png")
cv.save(p1)
print(f"wrote {p1}")

# ================================================================ 2. generation sheet
print()
print("=" * 100)
print("GENERATION: random 4 PCs -> colour face (the actual product)")
print("=" * 100)
rng = np.random.default_rng(4)
NZ = 10
Znew = np.clip(rng.standard_normal((NZ, 4)), -3.5, 3.5)
Gnew = gray_path(gnet, Znew)
Gnew_in = Gnew / 15.0

# colour components for a fresh seed are not available, so draw them from the training
# distribution: this is what the model will face in deployment, and it is the honest test
cidx = rng.choice(len(cpc), NZ, replace=False)
gen_rows = [("灰阶（阶段1）", None, Gnew)]
for tag in ORDER:
    m = torch.load(os.path.join(OUT, f"stage2_{tag}.pt"), weights_only=False)
    sch, k = m["scheme"], m["k"]
    if sch == "A":
        x = Gnew_in if k == 0 else np.concatenate([Gnew_in, cpc[cidx][:, :k]], 1)
    elif sch == "B":
        x = np.concatenate([Znew, cpc[cidx][:, :k]], 1)
    else:
        x = cpc[cidx][:, :k]
    net = nn.Sequential(nn.Linear(m["d_in"], m["hid"]), nn.ReLU(),
                        nn.Linear(m["hid"], m["hid"]), nn.ReLU(), nn.Linear(m["hid"], 96))
    net.load_state_dict(m["state"]); net.eval()
    with torch.no_grad():
        p = net(torch.tensor(x.astype(np.float32))).numpy() * 15.0
    gen_rows.append((f"{tag} {results[tag]['note']}", rgb_image(p), None))

W2 = 210 + NZ * (TILE + 4) + 4
H2 = 92 + len(gen_rows) * (TILE + 26) + 26
cv2 = Image.new("RGB", (W2, H2), (18, 20, 24))
dr2 = ImageDraw.Draw(cv2)
dr2.text((10, 10), "彩色生成：随机 4 个 PCA → 彩色头像", font=f_t, fill=(234, 238, 244))
dr2.text((10, 36), "第一行是阶段 1 的灰度结果（同一批随机种子）；下面每行是各方案上色后的样子。",
         font=f_s, fill=(150, 162, 178))
dr2.text((10, 52), "注意：颜色 PCA 是从训练集抽的，因为部署时也需要它作为条件输入。",
         font=f_s, fill=(130, 142, 158))
y = 76
for name, im, gonly in gen_rows:
    dr2.text((10, y + 4), name[:26], font=f_l, fill=(110, 168, 254))
    if gonly is not None:
        gi = layout.image_from_left_half(np.rint(gonly).astype(int))
        for i, g in enumerate(gi):
            tile = np.kron(np.clip(g / 15.0 * 255, 0, 255), np.ones((TILE // 8, TILE // 8)))
            rgb = np.stack([tile] * 3, -1).astype(np.uint8)
            cv2.paste(Image.fromarray(rgb), (210 + i * (TILE + 4), y))
    else:
        for i, g in enumerate(im):
            tile = np.kron(g, np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
            cv2.paste(Image.fromarray(tile), (210 + i * (TILE + 4), y))
    y += TILE + 26
p2 = os.path.join(OUT, "CMP_GENERATE.png")
cv2.save(p2)
print(f"wrote {p2}")

# ================================================================ 3. error maps
nvar = len(ORDER)
EM = 34
W3 = 200 + nvar * (EM * 8 + 20) + 20
H3 = 96 + EM * 8 + 40
cv3 = Image.new("RGB", (W3, H3), (18, 20, 24))
dr3 = ImageDraw.Draw(cv3)
dr3.text((10, 10), "逐位置误差：错误集中在哪（红=差）", font=f_t, fill=(234, 238, 244))
dr3.text((10, 34), "验证集上每个位置的平均 |预测−真实|，三通道取平均。越红错得越多。",
         font=f_s, fill=(150, 162, 178))
x = 200
for tag in ORDER:
    p, _, _, _ = model_input(tag, va, pc4[va])
    err = np.abs(np.clip(p, 0, 15) - rgb96[va]).reshape(-1, 32, 3).mean(-1).mean(0)
    full = np.zeros((8, 8))
    m = np.zeros((8, 8)); c = np.zeros((8, 8))
    for q in range(32):
        r, cl = layout.POS_ROW[q], layout.POS_COL[q]
        m[r, cl] += err[q]; c[r, cl] += 1
        m[r, layout.MIRROR_COL[q]] += err[q]; c[r, layout.MIRROR_COL[q]] += 1
    full = m / np.maximum(c, 1)
    mx = max(full.max(), 1e-6)
    rgbm = np.zeros((8, 8, 3), dtype=np.uint8)
    for r in range(8):
        for cl in range(8):
            t = full[r, cl] / mx
            rgbm[r, cl] = (int(255 * min(1, t * 2)), int(255 * max(0, 1 - abs(t - 0.5) * 2)),
                           int(255 * max(0, 1 - t * 2)))
    tile = np.kron(rgbm, np.ones((EM, EM, 1))).astype(np.uint8)
    cv3.paste(Image.fromarray(tile), (x, 56))
    dr3.text((x, 56 + EM * 8 + 4), tag, font=f_l, fill=(200, 208, 220))
    dr3.text((x, 56 + EM * 8 + 18), f"{full.mean():.2f}", font=f_s, fill=(150, 162, 178))
    x += EM * 8 + 20
p3 = os.path.join(OUT, "ERR_MAP.png")
cv3.save(p3)
print(f"wrote {p3}")
print("COLOUR_EVAL_DONE")
