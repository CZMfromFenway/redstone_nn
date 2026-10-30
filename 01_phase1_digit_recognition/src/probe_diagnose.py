#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断端到端退化的原因"""
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
bits_real = (U_real >= 0).astype(np.int64)
logit_real = bits_real @ W2 + b2

print("=" * 80)
print("诊断")
print("=" * 80)
print(f"标签正类比例 = {lab.mean()*100:.4f}%   负类比例 = {(1-lab).mean()*100:.4f}%")
print(f"(输出层偏置 b2 = {b2:.6f} — 必须计入 logit)")

# 1. 用实数 bit + 原 W2 是否能复现标签?
pred_real = (logit_real >= 0).astype(np.int8)
print(f"\n[1] bits_real + 原 W2 + b2:  与标签一致率 = {(pred_real==lab).mean()*100:.4f}%")
print(f"    bits_real + 原 W2 的预测正类比例 = {pred_real.mean()*100:.4f}%")

# 2. 用实数 bit + 整数化 W2
for q in (64, 4096):
    W2i = np.round(W2 * q).astype(np.int64); b2i = np.round(b2 * q)
    li = bits_real @ W2i + b2i
    p = (li >= 0).astype(np.int8)
    print(f"[2] bits_real + 整数化W2(q={q}): 一致率 = {(p==lab).mean()*100:.4f}%  "
          f"预测正类比例 = {p.mean()*100:.4f}%")

# 3. 用整数化隐藏 bit + 整数化 W2
for s in (256, 2048):
    A = np.round(W1 * s / 7.5).astype(np.int64)
    C = np.round(s * (b1 - W1.sum(axis=1))).astype(np.int64)
    U = X.astype(np.int64) @ A.T + C
    bits = (U >= 0).astype(np.int64)
    q = 4096
    W2i = np.round(W2 * q).astype(np.int64); b2i = np.round(b2 * q)
    li = bits @ W2i + b2i
    p = (li >= 0).astype(np.int8)
    print(f"[3] s={s:5d} 整数 bit + 整数 W2: 一致率 = {(p==lab).mean()*100:.4f}%  "
          f"预测正类比例 = {p.mean()*100:.4f}%")
    print(f"    整数 bit 与实数 bit 的逐神经元差异总数 = {int((bits != bits_real).sum())}")
    print(f"    实数 logit 与整数 logit 的符号差异数 = {int(((li>=0)!=(logit_real>=0)).sum())}")

# 4. 直接看: 用真 bit 与标签的关系
print(f"\n[4] logit_real 与标签的关系")
pos = logit_real[lab == 1]; neg = logit_real[lab == 0]
print(f"    标签=1 时 logit 范围 [{pos.min():.6f}, {pos.max():.6f}]")
print(f"    标签=0 时 logit 范围 [{neg.min():.6f}, {neg.max():.6f}]")
mis = ((logit_real >= 0).astype(np.int8) != lab)
print(f"    logit_real>=0 判断错误的样本数 = {int(mis.sum())}")
if mis.sum():
    print(f"    这些样本的 logit 绝对值: max={np.abs(logit_real[mis]).max():.6f}")
