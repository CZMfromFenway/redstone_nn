#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键观察: 隐藏层输出是 0/1, 所以**输出层可以精确整数化**。
logit = sum_j W2_j * bit_j + b2,  bit_j ∈ {0,1}
=> 用整型 W2int = round(W2 * q) 与 b2int = round(b2 * q) 复刻,
   只要 q 足够大, logit 的符号与原实数 logit 完全一致 (有限个和).
"""
import numpy as np, json, os

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
teach = os.path.join(base, "DT_Redstone_Research_Archive_20261001",
                     "DT_baseline_recovery", "teachers", "N4_MLP_T1.json")
m = json.load(open(teach, encoding="utf-8"))
W1 = np.array(m["W1"]); b1 = np.array(m["b1"])
W2 = np.array(m["W2"]); b2 = float(m["b2"])
lab = np.load(os.path.join(os.path.dirname(teach), "N4_MLP_T1.npy")).astype(np.int8)
X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
Xn = (X.astype(np.float64) - 7.5) / 7.5
U_real = Xn @ W1.T + b1
bits_real = (U_real >= 0)
logit_real = bits_real.astype(float) @ W2 + b2

print("=" * 96)
print("输出层的精确整数化")
print("=" * 96)
print(f"实数 logit 的范围: [{logit_real.min():.6f}, {logit_real.max():.6f}]")
print(f"最小非零间隙: {np.min(np.abs(logit_real[np.abs(logit_real)>0])):.6f}")
print(f"|logit| 的最小值: {np.min(np.abs(logit_real)):.6f}\n")

# 用真实 bit 找最小可行的 q
gaps = np.abs(logit_real)
gmin = gaps[gaps > 0].min() if (gaps > 0).any() else 1.0
print(f"要精确复刻符号, 只需 q > 1/(2*gap_min) = {1/(2*gmin):.1f}")
for q in (16, 64, 256, 1024, 4096):
    W2i = np.round(W2 * q).astype(np.int64)
    b2i = np.round(b2 * q)
    logit_i = bits_real.astype(np.int64) @ W2i + b2i
    ag = float(((logit_i >= 0) == (logit_real >= 0)).mean())
    print(f"  q={q:5d}: 与实数 logit 的符号一致率 = {ag*100:.4f}%  "
          f"(W2int={W2i.tolist()}, b2int={b2i})")

print("\n" + "=" * 96)
print("端到端: 隐藏层整数化 (尺度 s) + 输出层精确整数化")
print("=" * 96)
print(f"{'s':>6} | {'容器max':>8} | {'隐藏bit一致率':>13} | {'端到端一致率':>12} | 输出层 q")
print("-" * 96)
for s in (32, 64, 128, 256, 512, 1024, 2048):
    A = np.round(W1 * s / 7.5).astype(np.int64)
    cmax = int(15 + (A - A.min(axis=1, keepdims=True)).max())
    if cmax > 897:
        print(f"{s:6d} | {cmax:8d} | 越界"); continue
    C = np.round(s * (b1 - W1.sum(axis=1))).astype(np.int64)
    U = X.astype(np.int64) @ A.T + C
    bits = (U >= 0).astype(np.int64)
    bit_ag = float(((bits == 1) == bits_real).mean())
    # 输出层: 用足够大的 q 精确复刻
    q = max(4096, int(2 / gmin) + 2)
    W2i = np.round(W2 * q).astype(np.int64); b2i = np.round(b2 * q)
    logit_i = bits @ W2i + b2i
    pred = (logit_i >= 0).astype(np.int8)
    acc = float((pred == lab).mean())
    print(f"{s:6d} | {cmax:8d} | {bit_ag*100:12.4f}% | {acc*100:11.4f}% | q={q}")
