#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对齐 iris_compiled_v3 与 iris_tune4bit_v2 的前向，逐项打印"""
import numpy as np, json

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1f = np.array(d['W1']); b1f = np.array(d['b1'])
W2f = np.array(d['W2']); b2f = np.array(d['b2'])
Xq = np.array(d['Xq'], dtype=np.int64); y3 = np.array(d['y3'])
CAP = 868.0; x = np.arange(16)
Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
Uf = Xall.astype(np.float64) @ W1f.T + b1f[None, :]
alive = [j for j in range(W1f.shape[0]) if np.maximum(Uf[:, j], 0).max() > 1e-9]
h = len(alive)
W1 = W1f[alive]; b1 = b1f[alive]; W2 = W2f[alive]
idx150 = Xq[:, 0] * 4096 + Xq[:, 1] * 256 + Xq[:, 2] * 16 + Xq[:, 3]

rho = np.zeros((h, 4))
for j in range(h):
    mx = np.abs(W1[j]).max(); rho[j] = CAP * np.abs(W1[j]) / mx
up = np.round(x[None, None, :] * rho[:, :, None])
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None])
off = np.where(W1 > 0, 0.0, np.round(15 * rho))
Q = np.zeros((len(Xall), h))
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        lam = up[j, i, Xall[:, i]] if W1[j, i] > 0 else dn[j, i, Xall[:, i]]
        Q[:, j] += lam - off[j, i]
a = np.array([np.abs(W1[j]).max() for j in range(h)]) / CAP
U = Q * a[None, :] + b1[None, :]
rr = np.maximum(U, 0)
R = rr.max(axis=0)
v = np.clip(np.round(15 * rr / R[None, :]), 0, 15)

print("=" * 92)
print("对齐检查（用 iris_compiled_v3 的 raw 前向口径）")
print("=" * 92)
print(f"alive = {alive}, h = {h}")
print(f"R_j = {np.round(R,6).tolist()}")
print(f"R_j/15 = {np.round(R/15,6).tolist()}")
print()

# ---- 口径 A: raw（未取整权重）----
logitA = (v * (R / 15)[None, :]) @ W2.T + b2f[None, :]
predA = (np.argmax(logitA[idx150], 1) == y3).mean()
print(f"[口径A] logit = (v * R/15) @ W2.T + b2     -> 150点 3类 {predA:.4f}")

# ---- 口径 B: iris_compiled_v3 的做法（A 取整, th = round(S*b2)）----
for S in (25, 50, 100):
    Wt = (R / 15)[None, :] * W2.T
    A = np.round(S * Wt)
    th = np.round(S * b2f)
    logitB = v @ A.T + th[None, :]
    predB = (np.argmax(logitB[idx150], 1) == y3).mean()
    err = int(np.sum((logitB >= 0).astype(int) != (np.maximum(U, 0) @ W2 + b2f >= 0).astype(int)))
    print(f"[口径B] S={S:4d}: logit = v@A.T + th        -> 150点 3类 {predB:.4f}  全空间错误 {err}")

# ---- 口径 C: 我 v2 里的写法 ----
logitC = (v * (R / 15)[None, :]) @ W2.T + b2f[None, :]
print(f"[口径C] 同 A = {np.max(logitC[idx150]-logitA[idx150]):.3e}")
print()
print("=> 若 A 与 B 都给出高准确率, 说明问题在我的 v2 脚本的参数传递")
print()
print("logitA 前三行:")
print(np.round(logitA[idx150][:3], 5))
print("logitB(S=50) 前三行:")
Wt = (R / 15)[None, :] * W2.T; A = np.round(50 * Wt); th = np.round(50 * b2f)
logitB = v @ A.T + th[None, :]
print(np.round(logitB[idx150][:3], 5))
print(f"b2f = {b2f}")
print(f"th(S=50) = {th}")
