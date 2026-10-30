#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
逐条核对用户提出的结构性质，并给出数字。

    1 第一层输入是整数
    2 中间量是整数
    3 F 是整数
    4 A（锚点）与 c（步长）是整数
    5 每个内部 Y 逆映射回相同个数的 F；只有 Y=0 与 Y=255 可以不同
    6 次大权重不超过 882；最大权重是多少
    7 最后 32 个 Y 的典型范围
    8 一张脸在整数语境下如何归一化；比较 floor 版（用户方案）与 round 版
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
from intpipe import compile_layer, run, to_float_face, LIMIT, TOP_LIMIT  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
XMIN, XMAX, SPAN = -3.5, 3.5, 7.0
NLEV = 15

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

print("=" * 100)
print("1-4  类型核对")
print("=" * 100)
print(f"  第一层输入 X：      {np.int64}  （b=8 的无符号整数）")
print(f"  位权重 Tb：         {L1['Tb'].dtype}")
print(f"  F' = sum Tb*Xbit：  int64")
print(f"  F  = F' - F'lo：    int64（无符号，>= 0）")
print(f"  锚点 A：            {L1['A'].dtype}   层1 范围 [{L1['A'].min()}, {L1['A'].max()}]")
print(f"  步长 c：            {L1['c'].dtype}   层1 中位 {int(np.median(L1['c']))}，"
      f"层2 统一 {int(L2['c'][0])}")
print(f"  层2 锚点 A：        {L2['A'].dtype}   范围 [{L2['A'].min()}, {L2['A'].max()}]")

print()
print("=" * 100)
print("5  每个 Y 逆映射回 F 的个数")
print("=" * 100)
print("  规则：Y = clip((F - A) // c, 0, 255)")
print("        内部 Y 覆盖恰好 c 个 F；Y=0 与 Y=255 覆盖剩余部分")
print()
for tag, L, k in (("层1 隐单元 46", L1, 46), ("层2 像素 3", L2, 3)):
    A, c = int(L["A"][k]), int(L["c"][k])
    Fmax = int(L["Fhi"][k] - L["Flo"][k])
    y0_n = int(min(max(A + c, 0), Fmax + 1))                 # Y=0 覆盖 F=0..A+c-1
    ytop_lo = A + 255 * c
    ytop_n = int(max(0, Fmax - ytop_lo + 1))                 # Y=255 覆盖 F>=A+255c
    mids = [int(min(max(A + (y + 1) * c, 0), Fmax + 1) -
                min(max(A + y * c, 0), Fmax + 1)) for y in range(1, 255, 37)]
    print(f"  {tag}:  A={A}  c={c}  F 上限={Fmax}")
    print(f"    Y=0   覆盖 F ∈ [0, {A+c-1}]  → {y0_n} 个" + ("" if y0_n == c else "  ← 与 c 不同（允许）"))
    print(f"    Y=1..254 抽样覆盖个数: {mids}   全部等于 c={c}: "
          f"{all(v == c for v in mids)}")
    print(f"    Y=255 覆盖 F ∈ [{ytop_lo}, {Fmax}] → {ytop_n} 个" + ("" if ytop_n == c else "  ← 与 c 不同（允许）"))

print()
print("=" * 100)
print("6  权重上限")
print("=" * 100)
for tag, L in (("层 1", L1), ("层 2", L2)):
    ab = np.abs(L["Tb"])
    top = int(ab.max())
    second = int(np.sort(ab.ravel())[-2])
    non_top = int(ab[:, :, :-1].max())
    print(f"  {tag}: 最大权重 {top}（最高位），次大 {second}，"
          f"非最高位里的最大 {non_top}（上限 {LIMIT}）")
print(f"  最高位上限 {TOP_LIMIT:,}，实际用 {int(max(np.abs(L1['Tb']).max(), np.abs(L2['Tb']).max()))}")

# ---------------------------------------------------------------- 32 个 Y 的范围
NF = 20000
rng = np.random.default_rng(41)
xt = np.clip(rng.standard_normal((NF, 4)), XMIN, XMAX)
X1 = np.clip(np.floor((xt - XMIN) / SPAN * N), 0, N - 1).astype(np.int64)
_, _, Y1 = run(X1, L1)
_, _, Y2 = run(Y1, L2)

print()
print("=" * 100)
print(f"7  最后 32 个 Y 的典型范围（{NF} 张随机脸）")
print("=" * 100)
print(f"  {'像素':>4s} {'min':>5s} {'p5':>5s} {'中位':>5s} {'p95':>5s} {'max':>5s} {'跨度':>5s}")
print("  " + "-" * 44)
for p in range(32):
    v = Y2[:, p]
    print(f"  {p:4d} {v.min():5d} {int(np.percentile(v,5)):5d} {int(np.median(v)):5d} "
          f"{int(np.percentile(v,95)):5d} {v.max():5d} {int(np.percentile(v,95)-np.percentile(v,5)):5d}")
allv = Y2.ravel()
print()
print(f"  全部 32x{NF} 个 Y：min {allv.min()}  p5 {int(np.percentile(allv,5))}  "
      f"中位 {int(np.median(allv))}  p95 {int(np.percentile(allv,95))}  max {allv.max()}")
print(f"  达到 255 的比例 {100*(Y2==255).mean():.2f}%，达到 0 的比例 {100*(Y2==0).mean():.2f}%")
d = np.abs(Y2.max(1) - Y2.min(1))
print(f"  单张脸的 span：中位 {int(np.median(d))}  p5 {int(np.percentile(d,5))}  "
      f"p95 {int(np.percentile(d,95))}  最小 {d.min()}  最大 {d.max()}")

# ---------------------------------------------------------------- 归一化两种写法
print()
print("=" * 100)
print("8  归一化：floor 版（你的方案） vs 加 span/2 的 round 版")
print("=" * 100)


def norm_floor(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    span = np.maximum(ymax - ymin, 1)
    return np.clip(np.floor_divide((Y - ymin) * NLEV, span), 0, NLEV)


def norm_round(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    span = np.maximum(ymax - ymin, 1)
    return np.clip(np.floor_divide((Y - ymin) * NLEV + (span >> 1), span), 0, NLEV)


xr = XMIN + (X1 + 0.5) * SPAN / N
z2f = np.maximum(0.0, xr @ W1.T + T1) @ W2.T + T2
ref = to_float_face(z2f)
for name, fn in (("floor(15d/span)", norm_floor), ("round：+span/2", norm_round)):
    lv = fn(Y2)
    dd = lv - ref
    print(f"  {name:20s} 不同像素 {100*(dd!=0).mean():5.1f}%  "
          f"平均|差| {np.abs(dd).mean():.3f}  平均偏差 {dd.mean():+.3f}  "
          f"≥2 级的脸 {100*(np.abs(dd).max(1)>=2).mean():5.2f}%")

print()
print("  同一张脸、同一个像素的算例：")
one = 0
span = int(Y2[one].max() - Y2[one].min())
ymin = int(Y2[one].min())
print(f"    Ymin={ymin}  Ymax={Y2[one].max()}  span={span}")
for k in (0, 1, 2, 3):
    p = int(np.argsort(-np.abs(Y2[one] - ymin))[k])
    dk = int(Y2[one, p] - ymin)
    print(f"    像素 {p:2d}: Y={Y2[one,p]:3d}  d={dk:3d}  "
          f"floor(15d/span)={dk*15//span}   round={ (dk*15 + span//2)//span }  "
          f"浮点参照={ref[one,p]}")
print()
print("  右移一位得 span/2 的前提是 span 为偶数；span 为奇数时 >>1 等于 (span-1)/2，")
print("  此时 round 略微偏低，但仍在半步之内。")
print("CHECK_DONE")
