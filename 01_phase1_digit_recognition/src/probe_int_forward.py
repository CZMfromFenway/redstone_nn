#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""确定整数化前向的精确形式：两个独立实现互校"""
import numpy as np, json, os

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
teach = os.path.join(base, "DT_Redstone_Research_Archive_20261001",
                     "DT_baseline_recovery", "teachers", "N4_MLP_T1.json")
m = json.load(open(teach, encoding="utf-8"))
W1 = np.array(m["W1"]); b1 = np.array(m["b1"]); W2 = np.array(m["W2"]); b2 = float(m["b2"])
X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
Xn = (X.astype(np.float64) - 7.5) / 7.5

U_real = Xn @ W1.T + b1                      # (N,h) 实数预激活 (teacher 口径)

print("=" * 90)
print("寻找整数化前向的精确形式")
print("=" * 90)
print("teacher:  u_j(x) = W_j . ((x-7.5)/7.5) + b_j")
print("展开:     = (1/7.5) W_j . x  -  W_j . 1  +  b_j     (因为 7.5/7.5=1)\n")

# 候选 A: U = s*( W.x/7.5 - sum(W) + b )   —— 直接按定义缩放
for s in (16, 64, 256, 1024):
    A = np.round(W1 * s / 7.5).astype(np.int64)
    c = np.round(s * (b1 - W1.sum(axis=1))).astype(np.int64)
    U = X.astype(np.int64) @ A.T + c
    # 与实数前向的 1bit 一致率
    ag = float(((U >= 0) == (U_real >= 0)).mean())
    # 排序一致率(更强的检验)
    ordok = np.mean([np.array_equal(np.argsort(U[:, j], kind="stable"),
                                    np.argsort(U_real[:, j], kind="stable"))
                     for j in range(W1.shape[0])])
    print(f"  s={s:5d}: A范围=[{A.min():4d},{A.max():4d}]  c范围=[{c.min():6d},{c.max():6d}]  "
          f"1bit一致率={ag*100:8.4f}%  排序一致神经元比例={ordok*100:.1f}%")

# 改用 python 原生 round(银行家舍入) 与 np.round 的差异检查
print("\n[舍入] np.round 是银行家舍入; 用 half-away-from-zero 重试")
def rnd(x):
    return np.floor(np.asarray(x) + 0.5).astype(np.int64)
for s in (16, 64, 256, 1024, 4096):
    A = rnd(W1 * s / 7.5)
    c = rnd(s * (b1 - W1.sum(axis=1)))
    U = X.astype(np.int64) @ A.T + c
    ag = float(((U >= 0) == (U_real >= 0)).mean())
    ordok = np.mean([np.array_equal(np.argsort(U[:, j], kind="stable"),
                                    np.argsort(U_real[:, j], kind="stable"))
                     for j in range(W1.shape[0])])
    print(f"  s={s:5d}: 1bit一致率={ag*100:8.4f}%  排序一致比例={ordok*100:.1f}%  "
          f"容器最大={15+int((A-A.min()).max())}")
