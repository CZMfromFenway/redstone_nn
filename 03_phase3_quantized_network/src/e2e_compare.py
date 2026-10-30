#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
端到端对比，含一个由实验发现的改进。

改进
    层 2 是输出层，它的结果随后要被 clip 到 [0, 15]。
    原来的量化区间取 z2 的全部解析范围（约 [-50, +70]），
    但 [0,15] 之外的级全部会被 clip 掉，等于扔掉。
    改成直接在 [0, 15] 上量化并在两端饱和，是无损的，级数不变而步长变细。

两条路输入相同
    浮点路吃的是 X 的格心重建值，输入侧偏差为零。
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
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(P2, "src"))
from layout import image_from_left_half  # noqa: E402

LIMIT, XMIN, XMAX, NLEV = 882, -3.5, 3.5, 16
SPAN = XMAX - XMIN
OUT_LO, OUT_HI = 0.0, 15.0

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
T1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
W2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
T2 = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)


def unpack(X, b):
    return ((X[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)


def make_layer(Wf, th, cunit, b, relu, xbase=0.0, ymax_in=None, out_range=None):
    out, nin = Wf.shape
    N = 1 << b
    cu = np.full(nin, float(cunit)) if np.isscalar(cunit) else np.asarray(cunit, float)
    xb = np.full(nin, float(xbase)) if np.isscalar(xbase) else np.asarray(xbase, float)
    top = np.full(nin, N - 1, float) if ymax_in is None else np.asarray(ymax_in, float)
    amax = np.abs(Wf * cu[None, :]).max(axis=1)
    scale = LIMIT / (amax * (1 << (b - 2)))
    Tb = np.rint(scale[:, None, None] * Wf[:, :, None] * cu[None, :, None]
                 * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Fp_min = np.where(Tb < 0, Tb, 0).sum(axis=(1, 2))
    C0 = th + (xb * Wf).sum(axis=1) + 0.5 * (Wf * cu[None, :]).sum(axis=1)
    hi = C0 + ((Wf.clip(min=0) * cu[None, :]) * top[None, :]).sum(axis=1)
    lo = C0 + ((Wf.clip(max=0) * cu[None, :]) * top[None, :]).sum(axis=1)
    if out_range is not None:
        base = np.full(out, float(out_range[0]))
        hi = np.full(out, float(out_range[1]))
    elif relu:
        base = np.zeros(out)
    else:
        base = lo.copy()
    Fa = scale * (base - C0) - Fp_min
    Fb = scale * (hi - C0) - Fp_min
    c = np.maximum(np.ceil((Fb - Fa) / N), 1).astype(np.int64)
    return dict(Tb=Tb, scale=scale, Fp_min=Fp_min, C0=C0, hi=hi, lo=lo,
                base=base, Fa=Fa, Fb=Fb, c=c, b=b, N=N)


def run_layer(X, L):
    Fp_ = np.einsum("nij,kij->nk", unpack(X, L["b"]), L["Tb"])
    F = Fp_ - L["Fp_min"][None, :]
    return np.clip(np.floor((F - L["Fa"][None, :]) / L["c"][None, :]),
                   0, L["N"] - 1).astype(np.int64)


def decode(Y, L):
    return L["base"][None, :] + Y * (L["c"] / L["scale"])[None, :]


def to_face(v):
    v = np.clip(v, 0, 15)
    lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
    r = np.clip((v - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return np.rint(r * (NLEV - 1)).astype(int)


rng = np.random.default_rng(11)
TEST = 12
print("=" * 100)
print("END-TO-END  float vs integer")
print("=" * 100)

for b in (8, 11):
    N = 1 << b
    L1 = make_layer(W1, T1, SPAN / N, b, relu=True, xbase=XMIN)
    step1 = L1["c"] / L1["scale"]
    y1max = np.floor((L1["Fb"] - L1["Fa"]) / L1["c"])

    xt = np.clip(rng.standard_normal((TEST, 4)), XMIN, XMAX)
    Xt = np.clip(np.floor((xt - XMIN) / SPAN * N), 0, N - 1).astype(np.int64)
    x_recon = XMIN + (Xt + 0.5) * SPAN / N

    z2f = np.maximum(0.0, x_recon @ W1.T + T1) @ W2.T + T2
    Y1 = run_layer(Xt, L1)
    ff = to_face(z2f)

    variants = {
        "全范围 [-50,70]": make_layer(W2, T2, step1, b, relu=False,
                                    xbase=0.0, ymax_in=y1max),
        "clip 到 [0,15]": make_layer(W2, T2, step1, b, relu=False, xbase=0.0,
                                    ymax_in=y1max, out_range=(OUT_LO, OUT_HI)),
    }

    print()
    print("=" * 100)
    print(f"b = {b}")
    print("=" * 100)
    for name, L2 in variants.items():
        span2 = float(L2["hi"].max() - L2["lo"].min())
        Y2 = run_layer(Y1, L2)
        z2i = decode(Y2, L2)
        dz = z2i - z2f
        dzc = np.clip(z2i, 0, 15) - np.clip(z2f, 0, 15)
        fi = to_face(z2i)
        diff = np.abs(ff - fi)
        step2 = L2["c"] / L2["scale"]
        print(f"  {name}")
        print(f"    层 2 每级物理量 中位 {np.median(step2):.5f}   c2 中位 {int(np.median(L2['c']))}"
              f"   量化区间跨度 {L2['hi'].max()-L2['lo'].min():.2f}")
        print(f"    原始偏差 均值 {dz.mean():+.5f}  最大 {np.abs(dz).max():.5f}"
              + ("   （饱和区被计入，不代表画面误差）" if "clip" in name else ""))
        print(f"    截断后偏差 均值 {dzc.mean():+.5f}  标准差 {dzc.std():.5f}  "
              f"最大 {np.abs(dzc).max():.5f}")
        print(f"    16 级层  不同像素 {int((diff>0).sum())}/{diff.size}"
              f"（{100*(diff>0).mean():.1f}%）  平均差 {diff.mean():.3f}  "
              f"最大 {diff.max()} 级")

    # 用改进版出图
    L2 = variants["clip 到 [0,15]"]
    Y2 = run_layer(Y1, L2)
    z2i = decode(Y2, L2)
    fi = to_face(z2i)
    diff = np.abs(ff - fi)
    c2 = np.clip(z2f, 0, 15)
    lo_f, hi_f = c2.min(1), c2.max(1)
    c2i = np.clip(z2i, 0, 15)
    lo_i, hi_i = c2i.min(1), c2i.max(1)
    print(f"    拉伸前极值变化   min {np.abs(lo_i-lo_f).max():.4f}   "
          f"max {np.abs(hi_i-hi_f).max():.4f}   占拉伸跨度 "
          f"{100*np.median(np.abs(hi_i-hi_f)+np.abs(lo_i-lo_f))/np.median(hi_f-lo_f):.3f}%")

    imgf, imgi = image_from_left_half(ff), image_from_left_half(fi)
    dimg = image_from_left_half(diff)
    SC, PAD = 16, 4
    Wc = TEST * 8 * SC + (TEST + 1) * PAD
    Hc = 3 * 8 * SC + 4 * PAD
    canvas = np.full((Hc, Wc, 3), 20, dtype=np.uint8)
    for row, (imgs, tint) in enumerate(((imgf, (235, 235, 235)), (imgi, (90, 190, 255)))):
        for k in range(TEST):
            px = (imgs[k] / (NLEV - 1) * 255).astype(np.uint8)
            big = np.repeat(np.repeat(px, SC, 0), SC, 1)
            y0 = PAD + row * (8 * SC + PAD)
            x0 = PAD + k * (8 * SC + PAD)
            canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC] = np.stack([big, big, big], -1)
            canvas[y0 - 2:y0, x0:x0 + 8 * SC] = tint
    dmax = max(1, int(diff.max()))
    for k in range(TEST):
        px = (np.clip(dimg[k].astype(float) / dmax, 0, 1) * 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC, 0), SC, 1)
        y0 = PAD + 2 * (8 * SC + PAD)
        x0 = PAD + k * (8 * SC + PAD)
        canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC, 0] = big
        canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC, 1] = (big * 0.2).astype(np.uint8)
        canvas[y0:y0 + 8 * SC, x0:x0 + 8 * SC, 2] = (big * 0.2).astype(np.uint8)
        canvas[y0 - 2:y0, x0:x0 + 8 * SC] = (255, 110, 110)
    out = os.path.join(RESULTS, f"e2e_clip_b{b}.png")
    Image.fromarray(canvas).save(out)
    print(f"    wrote {out}")

print()
print("E2E_DONE")
