#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 折叠阶梯下 Q 与 u_true 是否相关（若无关则阶梯构造错）"""
import numpy as np, json

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1 = np.array(d['W1']); b1 = np.array(d['b1'])
h = W1.shape[0]; CAP = 868.0; x = np.arange(16)
rho = np.zeros((h, 4))
for j in range(h):
    mx = np.abs(W1[j]).max()
    rho[j] = CAP * np.abs(W1[j]) / mx if mx > 1e-12 else 0.0
up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)

Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
Xi = Xall.astype(np.int64); M = len(Xall)
Utrue = Xall.astype(np.float64) @ W1.T + b1[None, :]

print("=" * 92)
print("逐神经元诊断")
print("=" * 92)
for j in range(h):
    print(f"\nj={j}  W1 = {np.round(W1[j],4).tolist()}  b1 = {b1[j]:.4f}")
    print(f"  符号: {''.join('+' if W1[j,i]>0 else '-' for i in range(4))}")
    Qt = np.zeros(M, dtype=np.int64)       # 直接相加(不折叠)
    Qf = np.zeros(M, dtype=np.int64)       # 折叠(减 lambda(0))
    offsum = 0
    for i in range(4):
        lam = (up if W1[j, i] > 0 else dn)[j, i, Xi[:, i]]
        o = 0 if W1[j, i] > 0 else int(round(15 * rho[j, i]))
        offsum += o
        Qt += lam; Qf += lam - o
    mxj = np.abs(W1[j]).max(); a = mxj / (CAP * 7.5)
    r_t = np.corrcoef(Qt, Utrue[:, j])[0, 1]
    r_f = np.corrcoef(Qf, Utrue[:, j])[0, 1]
    print(f"  Q(直加) 范围 [{Qt.min()},{Qt.max()}]  与u相关 {r_t:.6f}")
    print(f"  Q(折叠) 范围 [{Qf.min()},{Qf.max()}]  与u相关 {r_f:.6f}   (offsetsum={offsum})")
    # 理论斜率
    print(f"  理论 u = a*Q + c, a={a:.9f}")
    for tag, Qv in (("直加", Qt), ("折叠", Qf)):
        A = np.hstack([Qv[:, None].astype(float), np.ones((M, 1))])
        c, *_ = np.linalg.lstsq(A, Utrue[:, j], rcond=None)
        print(f"    [{tag}] 拟合 a={c[0]:.9f} c={c[1]:.4f} 残差max={np.max(np.abs(A@c-Utrue[:,j])):.4f}")
