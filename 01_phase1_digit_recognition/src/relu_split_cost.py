#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReLU 拆分层成本：码值 +1 需要 Lambda 变化多少
=============================================
ReLU 编码: v_j = round(255*r_j/R_j),  r_j = max(0, u_j),  u_j = a_j*Lambda_j + c_j
故 dv=1  <=>  dLambda = R_j/(255*a_j)
其中 a_j = max_i|W1_ji| / (868*7.5)
"""
import numpy as np, json, os

base = r'D:\czm我的世界\文件\Comparator Network'
rec = os.path.join(base, 'DT_Redstone_Research_Archive_20261001', 'DT_baseline_recovery')
m = json.load(open(os.path.join(rec, 'teachers', 'N4_MLP_T1.json'), encoding='utf-8'))
W1 = np.array(m['W1']); b1 = np.array(m['b1'])
X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
Xi = X.astype(np.int64); N = len(X); h = 8; CAP = 868.0; x = np.arange(16)
Xn = (X.astype(float) - 7.5) / 7.5

rho = np.zeros((h, 4))
for j in range(h):
    rho[j] = CAP * np.abs(W1[j]) / np.abs(W1[j]).max()
up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)
S = np.zeros((N, h), dtype=np.int64)
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        S[:, j] += (up if W1[j, i] > 0 else dn)[j, i, Xi[:, i]]

mx = np.array([np.abs(W1[j]).max() for j in range(h)])
a = mx / (CAP * 7.5)
U = S * a[None, :] - np.abs(W1).sum(axis=1)[None, :] + b1[None, :]

# ReLU 老师的隐藏层输出（用官方 tanh 老师的 W1 造一个 ReLU 是无效的，
# 这里直接用"真实 u 经 ReLU"作为分析对象，因为编码规则只依赖 u 的范围）
R = np.maximum(U, 0)

print("=" * 100)
print("ReLU: 码值 +1 所需的 Lambda 变化")
print("=" * 100)
print(f"{'j':>4} | {'a_j':>12} | {'R_j':>9} | {'dLambda/码':>12} | {'Lambda范围':>16} | {'du/码':>10}")
print("-" * 100)
for j in range(8):
    dL = R[:, j].max() / (255 * a[j]) if a[j] > 0 else 0
    du = a[j] * dL
    print(f"{j:4d} | {a[j]:12.9f} | {R[:,j].max():9.3f} | {dL:12.1f} | "
          f"{S[:,j].min():7d}~{S[:,j].max():7d} | {du:10.5f}")
print("-" * 100)

dLs = np.array([R[:, j].max() / (255 * a[j]) for j in range(8)])
print(f"\n  dLambda/码: 最小 {dLs.min():.1f}  最大 {dLs.max():.1f}  比值 {dLs.max()/dLs.min():.2f}倍")
print(f"  对比 tanh:  最小 19     最大 4067    比值 214 倍")
print()
print("=" * 100)
print("为什么 ReLU 没有这个问题")
print("=" * 100)
print("  ReLU 在活跃区是 u_j 的**线性**函数: r_j = u_j = a_j*Lambda_j + c_j")
print("  故 dv = 1 对应 dLambda = R_j/(255*a_j) —— **处处相同**, 是常数")
print("  tanh 不是线性, 导数在两端趋于 0, 所以 dLambda/dv 在饱和区发散")
print()
print("  线性映射不需要查表:")
print("    v_j = round(255*(a_j*Lambda_j + c_j)/R_j) = round(k_j*Lambda_j + m_j)")
print("    Lambda -> v 是一次线性变换; 而 v 拆成两半后,")
print("    输出层对 v 的权重只差 16 倍, 所以 a_j/R_j 这个**逐通道因子**")
print("    可以完全由输出层的整数权重承担, 不需要额外硬件")
print()
print(f"  死区 (u_j<=0) 只需在 Lambda 的一个固定阈值处钳位 -> 1 次比较")
print()
print("=" * 100)
print("结论")
print("=" * 100)
print("  ReLU 拆分层的物理成本 = 1 次钳位(死区) + 输出层的整数权重")
print("  不需要查表, 不需要 255 个断点, 不需要 2116 格输入取样")
print("  => 这就是 ReLU 只要 48 单元而 tanh 要 176(或更多)单元的根本原因")
