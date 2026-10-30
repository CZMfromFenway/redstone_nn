#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
端到端对比，全整数管线。

相对上一版的修正
    1. 全部常数是整数：锚点 A、步长 c 都由整数位权重算出并取整
    2. 输出层 32 个神经元共用同一个缩放 s 与同一步长 c，
       只让锚点 A 逐神经元不同 —— 这样 32 个 Y 才可比
    3. 后处理用方案 C：(Y-Ymin)*15 + span/2 再整除以 span，
       两端都准；span 的倒数每张脸只算一次，32 个像素复用
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

XMIN, XMAX = -3.5, 3.5
SPAN = XMAX - XMIN
NLEV = 15

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
T1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
W2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
T2 = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)


def post_C(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    span = np.maximum(ymax - ymin, 1)
    return np.clip(np.floor_divide((Y - ymin) * NLEV + (span >> 1), span), 0, NLEV)


rng = np.random.default_rng(11)
TEST = 12
print("=" * 100)
print("END-TO-END 全整数管线   b = 8 / 11")
print("=" * 100)

for b in (8, 11):
    N = 1 << b
    L1 = compile_layer(W1, T1, SPAN / N, b, relu=True, xbase=XMIN)
    step1 = L1["c"] / L1["s"]
    y1max = np.clip(np.floor_divide(L1["Fhi"] - L1["A"], L1["c"]), 0, N - 1)
    L2 = compile_layer(W2, T2, step1, b, relu=False, xbase=0.0, ymax_in=y1max,
                       ztop=15.0, shared_scale=True)

    xt = np.clip(rng.standard_normal((TEST, 4)), XMIN, XMAX)
    Xt = np.clip(np.floor((xt - XMIN) / SPAN * N), 0, N - 1).astype(np.int64)
    xr = XMIN + (Xt + 0.5) * SPAN / N

    z2f = np.maximum(0.0, xr @ W1.T + T1) @ W2.T + T2
    _, _, Y1 = run(Xt, L1)
    _, _, Y2 = run(Y1, L2)
    lv_f = to_float_face(z2f)
    lv_i = post_C(Y2)
    d = lv_i - lv_f

    print()
    print("=" * 100)
    print(f"b = {b}")
    print("=" * 100)
    print(f"  层 1：缩放逐神经元 [{L1['s'].min():.1f}, {L1['s'].max():.1f}]，"
          f"c 中位 {int(np.median(L1['c']))}，A 范围 [{L1['A'].min()}, {L1['A'].max()}]")
    print(f"  层 2：缩放统一 {L2['s'][0]:.2f}，步长统一 {int(L2['c'][0])}，"
          f"A 范围 [{L2['A'].min()}, {L2['A'].max()}]")
    print(f"  16 级像素：不同 {int((d!=0).sum())}/{d.size}"
          f"（{100*(d!=0).mean():.1f}%）  平均差 {np.abs(d).mean():.3f}  "
          f"最大 {np.abs(d).max()} 级")
    print(f"    逐张最大差 " + " ".join(f"{int(v)}" for v in np.abs(d).max(1)))

    imgf, imgi = image_from_left_half(lv_f), image_from_left_half(lv_i)
    dimg = image_from_left_half(np.abs(d))
    SC, PAD = 16, 4
    Wc = TEST * 8 * SC + (TEST + 1) * PAD
    Hc = 3 * 8 * SC + 4 * PAD
    canvas = np.full((Hc, Wc, 3), 20, dtype=np.uint8)
    for row, (imgs, tint) in enumerate(((imgf, (235, 235, 235)), (imgi, (90, 190, 255)))):
        for k in range(TEST):
            px = (imgs[k] / NLEV * 255).astype(np.uint8)
            big = np.repeat(np.repeat(px, SC, 0), SC, 1)
            y0 = PAD + row * (8 * SC + PAD)
            x0 = PAD + k * (8 * SC + PAD)
            canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC] = np.stack([big, big, big], -1)
            canvas[y0 - 2:y0, x0:x0 + 8 * SC] = tint
    dmax = max(1, int(np.abs(d).max()))
    for k in range(TEST):
        px = (np.clip(dimg[k].astype(float) / dmax, 0, 1) * 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC, 0), SC, 1)
        y0 = PAD + 2 * (8 * SC + PAD)
        x0 = PAD + k * (8 * SC + PAD)
        canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC, 0] = big
        canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC, 1] = (big * 0.2).astype(np.uint8)
        canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC, 2] = (big * 0.2).astype(np.uint8)
        canvas[y0 - 2:y0, x0:x0 + 8 * SC] = (255, 110, 110)
    out = os.path.join(RESULTS, f"e2e_int_b{b}.png")
    Image.fromarray(canvas).save(out)
    print(f"    wrote {out}  前两行浮点/整数，第三行差值（最大 {dmax} 级）")

print()
print("E2E_INT_DONE")
