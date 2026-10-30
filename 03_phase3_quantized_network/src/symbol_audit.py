#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
符号清算：从输入 X 到第二层输出的全部表达式，逐个符号给出定义与实测值。

约定
    小写或希腊字母  = 实数
    大写            = 整数
    上标 (1) (2)    = 第几层
    下标 i          = 第 1 层的输入编号（0..3）
    下标 j          = 第 1 层的输出编号，同时是第 2 层的输入编号（0..63）
    下标 p          = 第 2 层的输出编号，即像素编号（0..31）
    下标 k          = 位号（0..b-1，k=b-1 为高位）

标 [模型] 的是训练得到的，标 [你的] 是你提出的记号，标 [我造的] 是我引入但之前没交代的。
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

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)   # W^(1)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)  # theta^(1)
W2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)   # W^(2)
TH2 = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)  # theta^(2)

B = 8
NB = 1 << B
XLO, XHI = -3.5, 3.5
DELTA = XHI - XLO                       # 输入的全跨度
LIMIT = 882
TOP = (1 << 31) - 1

print("=" * 104)
print("第 0 步  模型本身（浮点）")
print("=" * 104)
print("  [模型] W^(1)[j,i]  层1 权重  形状", W1.shape, " |W|最大 %.4f" % np.abs(W1).max())
print("  [模型] θ^(1)[j]    层1 偏置  形状", TH1.shape, " 范围 [%.4f, %.4f]" % (TH1.min(), TH1.max()))
print("  [模型] W^(2)[p,j]  层2 权重  形状", W2.shape, " |W|最大 %.4f" % np.abs(W2).max())
print("  [模型] θ^(2)[p]    层2 偏置  形状", TH2.shape, " 范围 [%.4f, %.4f]" % (TH2.min(), TH2.max()))
print()
print("  [模型] 第1层：  y1[j] = relu( Σ_i W^(1)[j,i]·x[i] + θ^(1)[j] )")
print("  [模型] 第2层：  z2[p] = Σ_j W^(2)[p,j]·y1[j] + θ^(2)[p]")
print("  [模型] 输出：    clip(z2[p], 0, 15) → 逐张 min-max → 16 级")

print()
print("=" * 104)
print("第 1 步  输入量化（x 是浮点采样值）")
print("=" * 104)
print("  [你的] b = %d，位宽" % B)
print("  [你的] X[i] = floor( (x[i] - x_min) / (x_max - x_min) · 2^b )   ∈ {0..%d}" % (NB - 1))
print("  [我造的] x_min = %.1f，x_max = %.1f" % (XLO, XHI))
print("  [我造的] Δ = x_max - x_min = %.1f   （输入全跨度）" % DELTA)
print("  [我造的] Δ_i = Δ / 2^b = %.8f   （每一级输入整数对应的物理量，逐输入相同）" % (DELTA / NB))
print("  [我造的] x̂[i] = x_min + (X[i] + 0.5)·Δ_i   （格心重建）")

cu1 = DELTA / NB
print()
print("  数值：Δ_i = %.8f" % cu1)

print()
print("=" * 104)
print("第 2 步  层 1 的整数位权重")
print("=" * 104)
print("  [你的] W^(1)_i^(k) 是该位位置上的整数权重，受 |W| ≤ 882（k < b-1）约束")
print("  [我造的] 层有 64 个输出，所以需要第三个下标 j：记为 U[j,i,k]")
print()
print("  [我造的] σ1[j] = 缩放，逐神经元，定义为使 max_i σ1[j]·|W^(1)[j,i]|·Δ_i·2^(b-2) = 882")
print("           层 1 的输出不需要互相比，所以每个神经元可以有自己的缩放。")
print("           层 2 则必须共用一个（见第 7 步）。")
amax1 = np.abs(W1 * cu1).max(axis=1)
sig1 = LIMIT / (amax1 * (1 << (B - 2)))
U1 = np.rint(sig1[:, None, None] * W1[:, :, None] * cu1
             * (1 << np.arange(B))[None, None, :]).astype(np.int64)
print(f"           σ1 范围 [{sig1.min():.1f}, {sig1.max():.1f}]")
print("  [我造的] U[j,i,k] = round( σ1[j] · W^(1)[j,i] · Δ_i · 2^k )")
print(f"           校验：k<b-1 的最大绝对值 = {np.abs(U1[:, :, :-1]).max()}（上限 882）")
print(f"                 k=b-1 的最大绝对值 = {np.abs(U1[:, :, -1]).max()}（上限 2^31-1）")

print()
print("=" * 104)
print("第 3 步  层 1 累加")
print("=" * 104)
print("  [你的] X_i^(k) 是 X[i] 的第 k 位")
print("  [你的] P'[j] = Σ_k Σ_i U[j,i,k]·X_i^(k)        ← 这就是你定义的 F'")
Plo1 = np.where(U1 < 0, U1, 0).sum(axis=(1, 2))
Phi1 = np.where(U1 > 0, U1, 0).sum(axis=(1, 2))
print("  [我造的] P_lo[j] = Σ min(0, U) = P' 的理论最小值")
print("  [你的] P[j] = P'[j] - P_lo[j]                  ← 这就是你定义的 F，>= 0")
print(f"         实测：P_lo ∈ [{Plo1.min()}, {Plo1.max()}]，P_hi ∈ [{Phi1.min()}, {Phi1.max()}]")
print(f"         P 的量程 ∈ [0, {int((Phi1 - Plo1).min())}] 到 [0, {int((Phi1 - Plo1).max())}]")

print()
print("=" * 104)
print("第 4 步  层 1 的实数量 ζ1 与常数 K1（这是我说 C0 的地方）")
print("=" * 104)
print("  [我造的] ζ1[j] = Σ_i W^(1)[j,i]·x̂[i] + θ^(1)[j]")
print("           即把格心重建的输入代进浮点网络得到的层 1 预激活值。")
print("           这就是我之前随口说的 z。")
print()
print("  [我造的] K1[j] = θ^(1)[j] + x_min·Σ_i W^(1)[j,i] + 0.5·Δ_i·Σ_i W^(1)[j,i]")
print("           含义：当所有 X[i]=0（即全部输入落在最低格中心）时的 ζ1 值。")
print("           这就是我之前叫 C0 的东西。")
K1 = TH1 + XLO * W1.sum(1) + 0.5 * cu1 * W1.sum(1)
print(f"           范围 [{K1.min():.4f}, {K1.max():.4f}]")
print()
print("  于是   Σ_i W^(1)[j,i]·Δ_i·X[i] = ζ1[j] - K1[j]")
print("  而     P'[j] ≈ σ1[j]·( ζ1[j] - K1[j] )")
print("  验证这条近似（关键恒等式）：")
rng = np.random.default_rng(5)
Xt = np.clip(np.floor(rng.standard_normal((4000, 4)) / DELTA * NB + NB / 2), 0, NB - 1)
Xt = np.clip(np.floor((np.clip(rng.standard_normal((4000, 4)), XLO, XHI) - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
bits = ((Xt[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
Pp = np.einsum("nik,jik->nj", bits, U1)
xh = XLO + (Xt + 0.5) * cu1
zeta1 = xh @ W1.T + TH1
pred = sig1[None, :] * (zeta1 - K1[None, :])
err = np.abs(Pp - pred)
print(f"         |P' - σ1(ζ1-K1)| 平均 {err.mean():.2f}，最大 {err.max():.0f}，"
      f"相对 P 量程 {100*err.max()/np.median(Phi1-Plo1):.3f}%")

print()
print("=" * 104)
print("第 5 步  层 1 输出 Y1")
print("=" * 104)
print("  [我造的] A1[j] = round( -σ1·K1[j] - P_lo[j] )")
print("           含义：ζ1 = 0 处的 P 值。层 1 有 ReLU，所以它同时也是 ReLU 阈值。")
A1 = np.rint(-sig1 * K1 - Plo1).astype(np.int64)
print(f"           范围 [{A1.min()}, {A1.max()}]")
zeta1max = TH1 + 3.5 * np.abs(W1).sum(1)
c1 = np.maximum(np.ceil(sig1 * zeta1max / NB), 1).astype(np.int64)
print("  [我造的] ζ1max[j] = θ^(1)[j] + 3.5·Σ_i |W^(1)[j,i]|")
print("           含义：ζ1 在输入立方体 [-3.5,3.5]^4 上的最大可能值")
print("  [我造的] c1[j] = ceil( σ1·ζ1max[j] / 2^b )")
print("           含义：每级 Y1 对应多少个整数的 P")
print(f"           c1 范围 [{c1.min()}, {c1.max()}]，中位 {int(np.median(c1))}")
print()
print("  [你的] Y1[j] = clip( floor( (P[j] - A1[j]) / c1[j] ), 0, 2^b-1 )")
print("         内部 Y 覆盖恰好 c1 个 P；Y=0 与 Y=255 覆盖剩余部分（允许不同）")

print()
print("=" * 104)
print("第 6 步  层 2 的输入步长 Δ2（这是我说 cunit 的地方）")
print("=" * 104)
print("  [我造的] Δ2[j] = c1[j] / σ1")
print("           含义：Y1[j] 每变一级，对应的物理量变化（即层 1 的量化步长）")
D2 = c1 / sig1
print(f"           范围 [{D2.min():.3e}, {D2.max():.3e}]，中位 {np.median(D2):.3e}")

print()
print("=" * 104)
print("第 7 步  层 2 的整数位权重")
print("=" * 104)
print("  [我造的] σ2 = 缩放，定义为使 max_{p,j} σ2·|W^(2)[p,j]|·Δ2[j]·2^(b-2) = 882")
amax2 = np.abs(W2 * D2[None, :]).max()
sig2 = LIMIT / (amax2 * (1 << (B - 2)))
V = np.rint(sig2 * W2[:, :, None] * D2[None, :, None]
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)
print(f"           σ2 = {sig2:.4f}")
print("  [我造的] V[p,j,k] = round( σ2 · W^(2)[p,j] · Δ2[j] · 2^k )")
print(f"           校验：k<b-1 的最大绝对值 = {np.abs(V[:, :, :-1]).max()}")
print(f"                 k=b-1 的最大绝对值 = {np.abs(V[:, :, -1]).max()}")

print()
print("=" * 104)
print("第 8 步  层 2 累加与常数 K2")
print("=" * 104)
print("  [你的] P'2[p] = Σ_k Σ_j V[p,j,k]·Y1_j^(k)")
print("  [你的] P2[p] = P'2[p] - P2_lo[p]，  P2_lo[p] = Σ min(0, V)")
Plo2 = np.where(V < 0, V, 0).sum(axis=(1, 2))
Phi2 = np.where(V > 0, V, 0).sum(axis=(1, 2))
print(f"         实测：P2_lo ∈ [{Plo2.min()}, {Plo2.max()}]，P2_hi ∈ [{Phi2.min()}, {Phi2.max()}]")
print()
print("  [我造的] K2[p] = θ^(2)[p] + 0.5·Σ_j W^(2)[p,j]·Δ2[j]")
print("           含义：当所有 Y1[j]=0 时的 ζ2 值（注意层 2 的输入下界是 0，不是 -3.5）")
K2 = TH2 + 0.5 * (W2 * D2[None, :]).sum(1)
print(f"           范围 [{K2.min():.4f}, {K2.max():.4f}]")

print()
print("=" * 104)
print("第 9 步  层 2 输出：锚点、截距修正、clip")
print("=" * 104)
print("  [我造的] A2[p] = round( -σ2·K2[p] - P2_lo[p] )")
print("           含义：ζ2 = 0 处的 P2 值。")
print("           层 2 无 ReLU，所以它只是截距修正，不是激活阈值；")
print("           在 S 处截断是因为画面只显示 [0,15]，那是显示下界。")
A2 = np.rint(-sig2 * K2 - Plo2).astype(np.int64)
print(f"           A2 范围 [{A2.min()}, {A2.max()}]，极差 {int(A2.max()-A2.min())}")
print("  [我造的] 上界 UPPER = round( 15·σ2 )")
UPPER = int(round(15 * sig2))
print(f"           UPPER = {UPPER}")
print()
print("  [你的] P̃[p] = clip( P2[p] - A2[p], 0, UPPER )")
print("         等于 clip( σ2·ζ2[p], 0, 15·σ2 )，即 σ2 乘上被截断到 [0,15] 的 ζ2。")

print()
print("=" * 104)
print("第 10 步  归一化与查表")
print("=" * 104)
print("  [你的] span = max_p P̃[p] - min_p P̃[p]")
print("  [你的] 起始信号 = 2^31 - (span >> 1)")
print("  [你的] 15d + span/2 = 起始信号 - ((起始信号 - P̃[p]) 连减 15 次)")
print("  [你的] 级数 = clip( 连减 span 的次数, 0, 15 )")
print("  [你的] 方块 = LUT[级数]")

print()
print("=" * 104)
print("关键近似的验证")
print("=" * 104)
Y1 = np.clip(np.floor_divide(Pp - Plo1[None, :] - A1[None, :], c1[None, :]), 0, NB - 1)
bitsY = ((Y1[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
Pp2 = np.einsum("njk,pjk->np", bitsY, V)
P2 = Pp2 - Plo2[None, :]
Ptil = np.clip(P2 - A2[None, :], 0, UPPER)
yh = Y1 * D2[None, :]
zeta2 = yh @ W2.T + TH2
pred2 = sig2 * np.clip(zeta2, 0, 15)
err2 = np.abs(Ptil - pred2)
print(f"  |P̃ - σ2·clip(ζ2,0,15)| 平均 {err2.mean():.2f}，最大 {err2.max():.0f}，"
      f"相对 UPPER {100*err2.max()/UPPER:.3f}%")
print("  这条近似成立，说明 P̃ 就是 σ2 倍的截断后 ζ2。")

print()
print("=" * 104)
print("全部常数的最终清单")
print("=" * 104)
print(f"  {'符号':10s} {'类型':8s} {'范围/值':40s}")
print("  " + "-" * 70)
rows = [
    ("Δ_i", "实数", f"{cu1:.8f}（逐输入相同）"),
    ("σ1", "实数", f"[{sig1.min():.1f}, {sig1.max():.1f}]（层1 逐神经元）"),
    ("U", "整数", f"|U|≤882（k<7），最大 {int(np.abs(U1).max())}（k=7）"),
    ("P_lo", "整数", f"[{int(Plo1.min())}, {int(Plo1.max())}]  逐神经元"),
    ("K1", "实数", f"[{K1.min():.3f}, {K1.max():.3f}]  逐神经元"),
    ("A1", "整数", f"[{int(A1.min())}, {int(A1.max())}]  逐神经元"),
    ("c1", "整数", f"[{int(c1.min())}, {int(c1.max())}]  逐神经元"),
    ("Δ2", "实数", f"[{D2.min():.2e}, {D2.max():.2e}]  逐输入"),
    ("σ2", "实数", f"{sig2:.4f}（层2 一个）"),
    ("V", "整数", f"|V|≤882（k<7），最大 {int(np.abs(V).max())}（k=7）"),
    ("P2_lo", "整数", f"[{int(Plo2.min())}, {int(Plo2.max())}]  逐神经元"),
    ("K2", "实数", f"[{K2.min():.3f}, {K2.max():.3f}]  逐神经元"),
    ("A2", "整数", f"[{int(A2.min())}, {int(A2.max())}]  逐神经元"),
    ("UPPER", "整数", f"{UPPER}（层2 一个）"),
]
for n, t, v in rows:
    print(f"  {n:10s} {t:8s} {v}")
print("SYMBOL_DONE")
