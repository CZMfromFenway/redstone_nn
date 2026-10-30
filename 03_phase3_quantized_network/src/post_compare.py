#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
后处理的整数化方案对比。

前一步发现 span>>4 有系统性偏低：span/16 < span/15，梯子偏陡。
这里比较三种纯整数方案，全部以浮点管线为参照。

    A  span >> 4，自顶向下      顶=15，底可能 >0
    B  ceil(span/15)，自底向上   底=0，顶可能 <15
    C  (Y-Ymin)*15 + span/2 再整除以 span   两端都准，但代价是一次乘 15 与一次除变量
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from intpipe import compile_layer, run, to_float_face  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
XMIN, XMAX = -3.5, 3.5
SPAN = XMAX - XMIN
NLEV = 15

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
T1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
W2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
T2 = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)


def post_A(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    step = np.maximum((ymax - ymin) >> 4, 1)
    return np.clip(NLEV - np.floor_divide(ymax - Y, step), 0, NLEV)


def post_B(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    span = ymax - ymin
    step = np.maximum(-(-span // 15), 1)
    return np.clip(np.floor_divide(Y - ymin, step), 0, NLEV)


def post_C(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    span = np.maximum(ymax - ymin, 1)
    num = (Y - ymin) * NLEV + (span >> 1)
    return np.clip(np.floor_divide(num, span), 0, NLEV)


NF = 2000
rng = np.random.default_rng(21)
print("=" * 100)
print(f"后处理方案对比：{NF} 张脸，b=8，参照为浮点管线")
print("=" * 100)

for b in (8, 11):
    N = 1 << b
    L1 = compile_layer(W1, T1, SPAN / N, b, relu=True, xbase=XMIN)
    step1 = L1["c"] / L1["s"]
    y1max = np.clip(np.floor_divide(L1["Fhi"] - L1["A"], L1["c"]), 0, N - 1)
    L2 = compile_layer(W2, T2, step1, b, relu=False, xbase=0.0, ymax_in=y1max,
                       ztop=15.0, shared_scale=True)

    xt = np.clip(rng.standard_normal((NF, 4)), XMIN, XMAX)
    Xt = np.clip(np.floor((xt - XMIN) / SPAN * N), 0, N - 1).astype(np.int64)
    xr = XMIN + (Xt + 0.5) * SPAN / N
    z2f = np.maximum(0.0, xr @ W1.T + T1) @ W2.T + T2
    ref = to_float_face(z2f)

    _, _, Y1 = run(Xt, L1)
    _, _, Y2 = run(Y1, L2)

    print()
    print("=" * 100)
    print(f"b = {b}   参照 = 浮点网络 + 浮点后处理")
    print("=" * 100)
    print(f"  {'方案':30s} {'不同像素':>9s} {'平均差':>8s} {'平均偏差':>9s} "
          f"{'最大差':>7s} {'出现>=2级的脸':>13s}")
    print("  " + "-" * 86)
    for name, fn, cost in (("A  span>>4  自顶向下", post_A, "一次右移"),
                           ("B  ceil(span/15) 自底向上", post_B, "一次除常数"),
                           ("C  (Y-Ymin)*15/span 四舍五入", post_C, "乘15 + 除变量")):
        lv = fn(Y2)
        d = lv - ref
        per_face_max = np.abs(d).max(1)
        print(f"  {name:30s} {100*(d!=0).mean():8.1f}% {np.abs(d).mean():8.3f} "
              f"{d.mean():+9.3f} {np.abs(d).max():7d} "
              f"{100*(per_face_max>=2).mean():12.1f}%")
    print()
    for name, fn, cost in (("A", post_A, ""), ("B", post_B, ""), ("C", post_C, "")):
        lv = fn(Y2)
        d = lv - ref
        hist = {v: 100 * float((d == v).mean()) for v in range(-3, 4)}
        print(f"    {name} 偏差分布: " + "  ".join(f"{k:+d}:{v:5.2f}%"
                                                  for k, v in hist.items()))
    print()
    print(f"  成本：A {13*'=':<13s} B {'':<13s} C {'':<13s}")
    print(f"        A 一次右移；B 一次除以常数 15；C 一次乘 15 加一次除以变量 span")

print()
print("PP_DONE")
