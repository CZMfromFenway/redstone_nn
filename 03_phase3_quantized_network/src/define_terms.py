#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
把记号钉死，并检验"全局平移是否足够"。

四个量，逐个定义
    F'   原始累加，sum_ij Tb[i][j] * Xbit[i][j]。数学上可正可负。
    F    F' - F'lo，其中 F'lo = sum min(0, Tb) 是本神经元的理论最小值。
         所以 F >= 0，且 F 的最小值 0 是本神经元自己的零点。
    A_p  本神经元的截距修正量，定义为 z=0 处的 F 值。
         输出层没有 ReLU，所以 A_p 不是激活阈值，而是"把 theta 造成的
         逐神经元截距抬平"的常数。
    F_t  clip(F - A_p, 0, 15s)，s 是全局共用缩放。
         下沿 0 由减法器的自然截断给出，上沿 15s 是公共常数。

要检验的命题
    只做一次全局平移（F = F' - F'lo_global）能否让 32 个值可比？
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

s2 = float(L2["s"][0])
TOP = int(round(15 * s2))

print("=" * 100)
print("四个量的定义与实测")
print("=" * 100)
print(f"  层 2 共用缩放 s = {s2:.4f}      上沿 15s = {TOP}")
print()
print(f"  {'量':6s} {'含义':40s} {'范围'}")
print("  " + "-" * 92)
print(f"  {'F´':6s} {'原始累加，可正可负':40s} "
      f"[{int(L2['Flo'].sum() and L2['Flo'].min())}, {int(L2['Fhi'].max())}]（各神经元不同）")
print(f"  {'F´lo':6s} {'F´ 的理论最小值 = 全部负权重之和':40s} "
      f"[{int(L2['Flo'].min())}, {int(L2['Flo'].max())}]")
print(f"  {'F':6s} {'F´ - F´lo >= 0，本神经元自己的零点':40s} "
      f"[0, {int((L2['Fhi']-L2['Flo']).max())}]")
print(f"  {'A_p':6s} {'z=0 处的 F，截距修正量':40s} "
      f"[{int(L2['A'].min())}, {int(L2['A'].max())}]")
print(f"  {'F_t':6s} {'clip(F - A_p, 0, 15s)':40s} [0, {TOP}]")

print()
print("=" * 100)
print("A_p 的离散程度：为什么需要逐神经元修正")
print("=" * 100)
spread = int(L2["A"].max() - L2["A"].min())
print(f"  A_p 范围 [{L2['A'].min()}, {L2['A'].max()}]，极差 {spread}")
print(f"  换成物理量：极差 / s = {spread/s2:.2f}，而显示区间只有 15")
print(f"  → 若不给逐神经元修正，32 个像素之间会差出 {spread/s2/15:.1f} 个满幅")
print()
print("  逐神经元 A_p：")
for r in range(4):
    print("    " + "  ".join(f"p{p:2d}:{int(L2['A'][p]):6d}" for p in range(r * 8, r * 8 + 8)))

# ---------------------------------------------------------------- 全局平移 vs 逐神经元
NF = 20000
rng = np.random.default_rng(41)
xt = np.clip(rng.standard_normal((NF, 4)), XMIN, XMAX)
X1 = np.clip(np.floor((xt - XMIN) / SPAN * N), 0, N - 1).astype(np.int64)
xr = XMIN + (X1 + 0.5) * SPAN / N
z2f = np.maximum(0.0, xr @ W1.T + T1) @ W2.T + T2
ref = to_float_face(z2f)

_, _, Y1 = run(X1, L1)
Fp2, F2, Y2 = run(Y1, L2)


def level_of(V):
    vmax, vmin = V.max(1, keepdims=True), V.min(1, keepdims=True)
    sp = np.maximum(vmax - vmin, 1)
    return np.clip(np.floor_divide((V - vmin) * NLEV + (sp >> 1), sp), 0, NLEV)


# 方案甲：逐神经元修正（正确）
V_ok = np.clip(F2 - L2["A"][None, :], 0, TOP)
# 方案乙：全局平移，不做逐神经元修正
G = int(L2["A"].max())
V_glob = np.clip(F2 - G, 0, TOP)
# 方案丙：全局平移后用原始 F（连上沿 clip 都不做）
V_raw = F2.astype(np.int64)

print()
print("=" * 100)
print("检验：全局平移够不够")
print("=" * 100)
print(f"  {'方案':34s} {'不同像素':>9s} {'平均|差|':>9s} {'≥2 级的脸':>10s}")
print("  " + "-" * 68)
for nm, V in (("逐神经元减 A_p，clip 到 [0,15s]", V_ok),
              ("只做一次全局平移 F - max(A_p)", V_glob),
              ("连平移都不做，直接用 F", V_raw)):
    d = level_of(V) - ref
    print(f"  {nm:34s} {100*(d!=0).mean():8.1f}% {np.abs(d).mean():9.3f} "
          f"{100*(np.abs(d).max(1)>=2).mean():9.1f}%")

print()
print("  全局平移失败的原因：F_p = s*z_p + s*C0_p + 常数，")
print("  每个神经元的截距 s*C0_p 不同，全局平移消不掉这一项，")
print("  于是 32 个值之间的差里混进了与 z 无关的逐神经元偏置。")
print()
print(f"  逐神经元的截距（s*C0_p，单位与 F 相同）范围：")
K = -(L2["A"] + L2["Flo"])
print(f"    [{int(K.min())}, {int(K.max())}]，极差 {int(K.max()-K.min())}")

print()
print("=" * 100)
print("最终结构（每个像素）")
print("=" * 100)
print("  1  求和节点：位贡献之和 + 常数(-F'lo_p)          → F >= 0，逐神经元常数")
print("  2  减法器（比较器的减法模式，天然截断在 0）")
print("     输出 = max(0, F - A_p)                        → 下沿 clip 免费，逐神经元常数")
print("  3  与公共常数 15s 比较，超过则取 15s             → 上沿 clip，全局常数")
print("  4  32 个值取 min/max，数电相减得 span，右移一位")
print("  5  起始信号 2^31 - span/2，减该值 15 次，再反减  → 15d + span/2")
print("  6  依次减 span 数次数，封顶 15                    → 级数")
print("  7  查 LUT 得方块")
print()
print("  逐神经元常数只有两个：-F'lo_p 与 A_p（可合并成一个），都写在容器里。")
print("  全局常数只有一个：15s。span 与 span/2 是每张脸现算的。")
print("DONE")
