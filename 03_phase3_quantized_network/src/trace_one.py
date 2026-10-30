#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
一个数字的旅程，全程整数。

与上一版的差别
    1. 所有编译期常数（锚点 A、步长 c）都是整数，运行期只有整数加减比较与整除
    2. 输出层 32 个神经元共用同一个缩放与步长，只让锚点 A 逐神经元不同，
       这样 32 个 Y 才是同一把尺子量出来的
    3. 后处理全程整数：span = Ymax - Ymin, step = span >> 4,
       level = 15 - clip((Ymax - Y) // step, 0, 15)
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
from intpipe import compile_layer, run, to_levels, to_float_face  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
XMIN, XMAX = -3.5, 3.5
SPAN = XMAX - XMIN

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
y1max = np.clip(np.floor_divide(L1["Fhi"] - L1["A"], L1["c"]), 0, N - 1)
L2 = compile_layer(W2, T2, step1, B, relu=False, xbase=0.0, ymax_in=y1max,
                   ztop=15.0, shared_scale=True)

rng = np.random.default_rng(11)
x_cont = np.clip(rng.standard_normal(4), XMIN, XMAX)
X = np.clip(np.floor((x_cont - XMIN) / SPAN * N), 0, N - 1).astype(np.int64)
x = XMIN + (X + 0.5) * SPAN / N
one = X[None, :]

print("=" * 104)
print("一张脸")
print("=" * 104)
print(f"  {'':10s} {'PC0':>11s} {'PC1':>11s} {'PC2':>11s} {'PC3':>11s}")
print(f"  {'采样':10s} " + " ".join(f"{v:11.5f}" for v in x_cont))
print(f"  {'X (b=8)':10s} " + " ".join(f"{v:11d}" for v in X))
print(f"  {'格心重建':10s} " + " ".join(f"{v:11.5f}" for v in x))
print(f"  {'二进制':10s} " + " ".join(f"{format(v,'08b'):>11s}" for v in X))

y1f = np.maximum(0.0, x @ W1.T + T1)
Fp1, F1, Y1 = run(one, L1)
active = np.where(y1f > 1e-9)[0]
j = int(active[np.argmax(np.abs(W1[active, 0]))])
p = int(np.argmax(np.abs(W2[:, j])))
row, col = p % 8, p // 8
print()
print(f"  激活 {len(active)}/64 个隐单元。通路：PC0 -> 隐单元 {j} -> 像素 {p} "
      f"(row {row}, col {col})")

print()
print("=" * 104)
print(f"层 1 · 隐单元 {j}")
print("=" * 104)
print(f"  {'i':>3s} {'':>5s} {'x_i':>10s} {'w':>10s} {'w*x_i':>10s} {'X_i':>6s}   位权重 k=0..7")
print("  " + "-" * 100)
for i in range(4):
    print(f"  {i:3d} {'PC%d' % i:>5s} {x[i]:10.5f} {W1[j,i]:+10.6f} {W1[j,i]*x[i]:+10.6f} "
          f"{X[i]:6d}   " + " ".join(f"{v:+d}" for v in L1["Tb"][j, i]))
print(f"  {'':>3s} {'th':>5s} {T1[j]:+10.6f}")

Fpj, F_j, Yj = int(Fp1[0, j]), int(F1[0, j]), int(Y1[0, j])
A1, c1 = int(L1["A"][j]), int(L1["c"][j])
b0 = (X[0] >> np.arange(B)) & 1
print()
print(f"  连续  z_{j} = {W1[j] @ x + T1[j]:+.6f}      y_{j} = relu(z) = {y1f[j]:.6f}")
print()
print("  整数（常数全为整数）")
print("    PC0 的位贡献：" + " ".join(f"{L1['Tb'][j,0,k]:+d}*{b0[k]}" for k in range(B))
      + f" = {int((L1['Tb'][j,0]*b0).sum()):+d}")
print(f"    F' = {Fpj}     F'lo = {int(L1['Flo'][j])}     F = {F_j}")
print(f"    A = {A1}（z=0 处的累加器读数，已取整）     c = {c1}")
print(f"    Y_{j} = floor(({F_j} - {A1}) / {c1}) = floor({F_j-A1} / {c1}) = {Yj}")
lo_bin, hi_bin = A1 + Yj * c1, A1 + Yj * c1 + c1 - 1
print(f"    即 Y={Yj} 覆盖 F ∈ [{lo_bin}, {hi_bin}]，共 {c1} 个值"
      + ("；F 在其中 ✓" if lo_bin <= F_j <= hi_bin else "；F 不在其中 ✗"))
print(f"    解码 y = Y*c/s = {Yj*c1/L1['s'][j]:.6f}（连续 {y1f[j]:.6f}，"
      f"差 {(Yj*c1/L1['s'][j]-y1f[j])/(c1/L1['s'][j]):+.3f} 级）")

print()
print("=" * 104)
print(f"层 2 · 像素 {p}   （32 个神经元共用缩放与步长）")
print("=" * 104)
print(f"  共用缩放 s = {L2['s'][0]:.3f}     共用步长 c = {int(L2['c'][0])}")
print(f"  锚点逐神经元不同：本像素 A = {int(L2['A'][p])}，"
      f"全体 [{int(L2['A'].min())}, {int(L2['A'].max())}]")
print("  于是 32 个 Y 都满足 Y ≈ (s/c) * z，是同一个换算，可以做 min-max。")

z2f_all = y1f @ W2.T + T2
Fp2, F2, Y2 = run(Y1, L2)
print()
print(f"  连续  z_{p} = {z2f_all[p]:+.6f}")
print(f"  整数  F' = {int(Fp2[0,p])}   F = {int(F2[0,p])}   A = {int(L2['A'][p])}   "
      f"Y_{p} = floor({int(F2[0,p]) - int(L2['A'][p])} / {int(L2['c'][0])}) = {int(Y2[0,p])}")

lv_i = to_levels(Y2)[0]
lv_f = to_float_face(z2f_all[None, :])[0]
Ymax, Ymin = int(Y2.max()), int(Y2.min())
span = Ymax - Ymin
step = max(1, span >> 4)
print()
print("=" * 104)
print("后处理：32 个 Y -> 16 级（全整数）")
print("=" * 104)
print(f"  Ymin = {Ymin}   Ymax = {Ymax}   span = {span}")
print(f"  step = span >> 4 = {span} >> 4 = {step}")
u = (Ymax - int(Y2[0, p])) // step
print(f"  像素 {p}：Y = {int(Y2[0,p])}，距 Ymax {Ymax - int(Y2[0,p])} 步，"
      f"除 step 得 {u} 级")
print(f"  level = 15 - {u} = {lv_i[p]}      浮点管线同一像素 = {lv_f[p]}")
print()
print(f"  {'像素':>4s} {'Y':>7s} {'距顶步数':>9s} {'整数级':>7s} {'浮点级':>7s} {'差':>4s}")
print("  " + "-" * 46)
for q in range(32):
    print(f"  {q:4d} {int(Y2[0,q]):7d} {(Ymax-int(Y2[0,q]))//step:9d} "
          f"{lv_i[q]:7d} {lv_f[q]:7d} {lv_i[q]-lv_f[q]:4d}")

print()
print("=" * 104)
print("旅程（整数列全部是整数）")
print("=" * 104)
print(f"  {'步骤':32s} {'连续':>15s} {'整数':>15s}")
print("  " + "-" * 66)
print(f"  {'1. PC0 采样值':32s} {x_cont[0]:15.5f} {'':>15s}")
print(f"  {'2. 量化成 X_0':32s} {'':>15s} {X[0]:15d}")
print(f"  {'3. 重建值 x_0':32s} {x[0]:15.5f} {format(int(X[0]),'08b'):>15s}")
print(f"  {'4. 给隐单元 %d 的位贡献' % j:32s} {W1[j,0]*x[0]:15.6f} "
      f"{int((L1['Tb'][j,0]*b0).sum()):15d}")
print(f"  {'5. 隐单元 %d 的 Y' % j:32s} {y1f[j]:15.6f} {Yj:15d}")
print(f"  {'6. 像素 %d 的 Y' % p:32s} {z2f_all[p]:15.6f} {int(Y2[0,p]):15d}")
print(f"  {'7. 像素 %d 的 16 级值' % p:32s} {lv_f[p]:15d} {lv_i[p]:15d}")
print()
print(f"  累加器量程  隐单元 {j}: F∈[0, {int(L1['Fhi'][j]-L1['Flo'][j])}]，"
      f"A={A1}，c={c1}")
print(f"              像素 {p}: F∈[0, {int(L2['Fhi'][p]-L2['Flo'][p])}]，"
      f"A={int(L2['A'][p])}，c={int(L2['c'][0])}")
print("TRACE_DONE")
