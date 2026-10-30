#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
生成效果图。

    图 A  gallery_int.png     六个风格各四张，全整数管线，16 级
    图 B  gallery_compare.png 同样的脸：浮点 16 级 / 整数 16 级 / 整数 MC 11 档 / 差值

风格的采法：x = mu + chol @ z，z ~ N(0,I)，落在 ±3.5 之外则重采。
"""
import json
import os
import sys
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from intpipe import compile_layer, run, to_float_face  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(P2, "src"))
from layout import image_from_left_half  # noqa: E402

XMIN, XMAX, SPAN, NLEV = -3.5, 3.5, 7.0, 15

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
T1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
W2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
T2 = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

LUT = np.array(json.load(open(os.path.join(P2, "webapp", "merge_mc11.json"),
                              encoding="utf-8"))["_lut"], dtype=np.float64)
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]

B = 8
N = 1 << B
L1 = compile_layer(W1, T1, SPAN / N, B, relu=True, xbase=XMIN)
step1 = L1["c"] / L1["s"]
y1m = np.clip(np.floor_divide(L1["Fhi"] - L1["A"], L1["c"]), 0, N - 1)
L2 = compile_layer(W2, T2, step1, B, relu=False, xbase=0.0, ymax_in=y1m,
                   ztop=15.0, shared_scale=True)


def post_C(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    span = np.maximum(ymax - ymin, 1)
    return np.clip(np.floor_divide((Y - ymin) * NLEV + (span >> 1), span), 0, NLEV)


K = 4
rng = np.random.default_rng(7)
xs = []
for e in styles:
    mu, chol = np.array(e["mu"]), np.array(e["chol"])
    got = []
    while len(got) < K:
        z = rng.standard_normal(4)
        v = mu + chol @ z
        if np.all(np.abs(v) <= 3.5):
            got.append(v)
    xs.append(np.array(got))
x_all = np.concatenate(xs, 0)                      # (6K, 4)
ns = len(x_all)

Xt = np.clip(np.floor((x_all - XMIN) / SPAN * N), 0, N - 1).astype(np.int64)
xr = XMIN + (Xt + 0.5) * SPAN / N
z2f = np.maximum(0.0, xr @ W1.T + T1) @ W2.T + T2
lv_f = to_float_face(z2f)
_, _, Y1 = run(Xt, L1)
_, _, Y2 = run(Y1, L2)
lv_i = post_C(Y2)

print(f"生成 {ns} 张：{len(styles)} 个风格 × {K} 张")
d = lv_i - lv_f
print(f"  整数 vs 浮点：不同像素 {100*(d!=0).mean():.2f}%，平均差 {np.abs(d).mean():.3f} 级")
print(f"  逐张最大差分布：" + " ".join(f"{int(v)}" for v in np.abs(d).max(1)))

img_i = image_from_left_half(lv_i)
img_f = image_from_left_half(lv_f)
img_m = image_from_left_half(np.zeros_like(lv_i))     # 占位，下面用亮度重填

# ---- 图 A：整数管线的画廊
SC, PAD, GAP = 30, 6, 14
Wc = K * 8 * SC + (K - 1) * PAD + 2 * GAP
Hc = len(styles) * (8 * SC + GAP) + GAP
canvas = np.full((Hc, Wc, 3), 18, dtype=np.uint8)
for s in range(len(styles)):
    for k in range(K):
        g = img_i[s * K + k]
        px = (g / NLEV * 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC, 0), SC, 1)
        y0 = GAP + s * (8 * SC + GAP)
        x0 = GAP + k * (8 * SC + PAD)
        canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC] = np.stack([big, big, big], -1)
out = os.path.join(RESULTS, "gallery_int.png")
Image.fromarray(canvas).save(out)
print(f"  wrote {out}  ({os.path.getsize(out)/1e3:.0f} KB)")

# ---- 图 B：浮点 / 整数 / MC11 / 差值，取每风格前 2 张共 12 张
sel = [s * K + k for s in range(len(styles)) for k in (0, 1)]
NS2 = len(sel)
rows = [("float16", img_f[sel]), ("int16", img_i[sel])]
lux_i = LUT[lv_i[sel]]
img_mc = image_from_left_half(np.searchsorted(LUT, LUT[lv_i[sel]]) * 0)  # 占位
# MC11 用亮度值直接画
mc = np.zeros((NS2, 8, 8))
for t in range(NS2):
    mc[t] = image_from_left_half(LUT[lv_i[sel]][t])
SC2, PAD2 = 18, 4
Wc2 = NS2 * 8 * SC2 + (NS2 + 1) * PAD2
Hc2 = 4 * 8 * SC2 + 5 * PAD2
canvas2 = np.full((Hc2, Wc2, 3), 18, dtype=np.uint8)
tints = [(235, 235, 235), (90, 190, 255), (255, 200, 110), (255, 110, 110)]
for r in range(4):
    for t in range(NS2):
        if r == 0:
            g = img_f[sel][t] / NLEV * 15
        elif r == 1:
            g = img_i[sel][t] / NLEV * 15
        elif r == 2:
            g = mc[t]
        else:
            g = np.abs(img_i[sel][t] - img_f[sel][t]).astype(float) * 15
        px = np.clip(g / 15.0 * 255, 0, 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC2, 0), SC2, 1)
        y0 = PAD2 + r * (8 * SC2 + PAD2)
        x0 = PAD2 + t * (8 * SC2 + PAD2)
        canvas2[y0:y0 + 8 * SC2, x0:x0 + 8 * SC2] = np.stack([big, big, big], -1)
        canvas2[y0 - 2:y0, x0:x0 + 8 * SC2] = tints[r]
out2 = os.path.join(RESULTS, "gallery_compare.png")
Image.fromarray(canvas2).save(out2)
print(f"  wrote {out2}  ({os.path.getsize(out2)/1e3:.0f} KB)")
print("  行序：浮点16 / 整数16 / 整数MC11 / 差值")
print("GALLERY_DONE")
