#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
跳过 Y，直接用 F 做归一化。

现在：  F -> Y = clip((F-A)//c, 0, 255) -> level = round(15(Y-Ymin)/span_Y)
改后：  F_bar = F - A                -> level = round(15(F_bar-Fmin)/span_F)

为什么更准
    Y 的量化误差是半步（c/2），而 F_bar 自身的误差只有权重取整噪声（约 1 个单位）。
    跳过 Y 等于把 c/2 那一层误差整个去掉，精度提高约 c 倍。

代价
    F_bar 的数值比 Y 大 c 倍，所以 15*(F_bar-Fmin) 也大 c 倍。
    这个量仍在强模范围内，但要核一下最坏情况。
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
from intpipe import compile_layer, run, to_float_face, LIMIT  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
XMIN, XMAX, SPAN = -3.5, 3.5, 7.0
NLEV = 15
STRONG = (1 << 31) - 1

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
T1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
W2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
T2 = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

B = 8
N = 1 << B
L1 = compile_layer(W1, T1, SPAN / N, B, relu=True, xbase=XMIN)
step1 = L1["c"] / L1["s"]
y1m = np.clip(np.floor_divide(L1["Fhi"] - L1["A"], L1["c"]), 0, N - 1)
L2 = compile_layer(W2, T2, step1, B, relu=False, xbase=0.0, ymax_in=y1m,
                   ztop=15.0, shared_scale=True)

NF = 20000
rng = np.random.default_rng(41)
xt = np.clip(rng.standard_normal((NF, 4)), XMIN, XMAX)
X1 = np.clip(np.floor((xt - XMIN) / SPAN * N), 0, N - 1).astype(np.int64)
xr = XMIN + (X1 + 0.5) * SPAN / N
z2f = np.maximum(0.0, xr @ W1.T + T1) @ W2.T + T2

_, _, Y1 = run(X1, L1)
Fp2, F2, Y2 = run(Y1, L2)
Fbar = F2 - L2["A"][None, :]          # 去掉逐神经元常数的累加器值

print("=" * 104)
print(f"最后 32 个 F 的范围（{NF} 张随机脸）")
print("=" * 104)
print(f"  逐神经元 A（编译期常数）范围 [{L2['A'].min()}, {L2['A'].max()}]")
print(f"  逐神经元 F 上限（Fhi-Flo）范围 [{int((L2['Fhi']-L2['Flo']).min())}, "
      f"{int((L2['Fhi']-L2['Flo']).max())}]")
print()
print(f"  {'像素':>4s} {'F 最小':>9s} {'F 最大':>9s} | {'F_bar 最小':>10s} "
      f"{'F_bar p5':>9s} {'F_bar 中位':>10s} {'F_bar p95':>9s} {'F_bar 最大':>10s}")
print("  " + "-" * 88)
for p in range(32):
    f = F2[:, p]
    fb = Fbar[:, p]
    print(f"  {p:4d} {f.min():9d} {f.max():9d} | {fb.min():10d} "
          f"{int(np.percentile(fb,5)):9d} {int(np.median(fb)):10d} "
          f"{int(np.percentile(fb,95)):9d} {fb.max():10d}")
print()
print(f"  F_bar 全体：min {Fbar.min()}  max {Fbar.max()}")
span_f = Fbar.max(1) - Fbar.min(1)
print(f"  单张脸的 F_bar span：中位 {int(np.median(span_f))}  p5 {int(np.percentile(span_f,5))}  "
      f"p95 {int(np.percentile(span_f,95))}  最小 {span_f.min()}  最大 {span_f.max()}")
print(f"  需要 15*span 的最大值：{15*int(span_f.max()):,}   强模上限 {STRONG:,}  "
      f"占 {100*15*span_f.max()/STRONG:.3f}%")

# ---------------------------------------------------------------- 两种归一化
print()
print("=" * 104)
print("归一化：经过 Y  vs  直接用 F_bar")
print("=" * 104)


def lv_from_Y(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    sp = np.maximum(ymax - ymin, 1)
    return np.clip(np.floor_divide((Y - ymin) * NLEV + (sp >> 1), sp), 0, NLEV)


def lv_from_F(F):
    fmax, fmin = F.max(1, keepdims=True), F.min(1, keepdims=True)
    sp = np.maximum(fmax - fmin, 1)
    return np.clip(np.floor_divide((F - fmin) * NLEV + (sp >> 1), sp), 0, NLEV)


ref = to_float_face(z2f)
for name, lv in (("经过 Y（当前）", lv_from_Y(Y2)), ("直接用 F_bar（你的想法）", lv_from_F(Fbar))):
    d = lv - ref
    print(f"  {name:26s} 不同像素 {100*(d!=0).mean():5.2f}%  平均|差| {np.abs(d).mean():.4f}  "
          f"平均偏差 {d.mean():+.4f}  ≥2 级的脸 {100*(np.abs(d).max(1)>=2).mean():5.2f}%")

print()
print("  偏差分布：")
for name, lv in (("经过 Y", lv_from_Y(Y2)), ("直接用 F_bar", lv_from_F(Fbar))):
    d = lv - ref
    print(f"    {name:14s} " + "  ".join(
        f"{k:+d}:{100*float((d==k).mean()):5.2f}%" for k in range(-2, 3)))

# ---------------------------------------------------------------- 需要减几次
print()
print("=" * 104)
print("连减次数（强模信号不能移位相乘，只能连减）")
print("=" * 104)
print("  x15 部分：被乘数最大 %d，连减 15 次，结果最大 %d" % (int(span_f.max()), 15*int(span_f.max())))
print("  除以 span 部分：被除数最大 %d，连减次数 = 商，最大 15 次" % (15*int(span_f.max())))
print("  合计每像素最多 30 次连减，与 Y 版本相同（Y 版本被乘数更小，但次数一样）")

print()
print("CHECK_F_DONE")
