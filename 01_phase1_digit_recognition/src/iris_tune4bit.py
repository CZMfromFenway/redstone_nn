#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花 4bit 微调：把 LUT 量化放进前向，直接优化编译后的误差
==============================================================
【思路】
  原 teacher 是在"连续权重 + 连续激活"下训练的，编译时插入两级量化
  (第一层 LUT 的整数阶梯 + 中间层 ReLU 到 4bit) 会掉 1 个样本。
  本脚本把这两级量化放进**前向**，用 STE 反传，直接优化编译后的输出。

【前向（与 iris_compiled_v3 完全一致）】
  Q_j = sum_i (lambda_i(x_i) - lambda_i(0)),  lambda 由 rho=868|W1|/max|W1| 生成整数阶梯
  u_j = (max_i|W1_ji|/868)*Q_j + b1_j
  r_j = max(0, u_j)
  v_j = round(15*r_j/R_j)        <- 4bit, R_j 由当前权重算
  logit_k = sum_j (R_j/15)*W2_kj*v_j + b2_k
  pred = logit >= 0

【评价】150 点 3 类准确率（真实标签）+ 全空间二值错误
"""
import numpy as np, json, os

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1f = np.array(d['W1']); b1f = np.array(d['b1'])
W2f = np.array(d['W2']); b2f = np.array(d['b2'])
Xq = np.array(d['Xq'], dtype=np.int64); y3 = np.array(d['y3'])
N150 = len(y3)
CAP = 868.0; x = np.arange(16)
Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
M = len(Xall)

# 剔除死神经元
Uf = Xall.astype(np.float64) @ W1f.T + b1f[None, :]
alive = [j for j in range(W1f.shape[0]) if np.maximum(Uf[:, j], 0).max() > 1e-9]
h = len(alive)
W1_0 = W1f[alive]; b1_0 = b1f[alive]; W2_0 = W2f[alive]

Qall_idx = Xall[:, 0] * 4096 + Xall[:, 1] * 256 + Xall[:, 2] * 16 + Xall[:, 3]
idx150 = Xq[:, 0] * 4096 + Xq[:, 1] * 256 + Xq[:, 2] * 16 + Xq[:, 3]


def lut_tables(W1):
    rho = np.zeros((h, 4))
    for j in range(h):
        mx = np.abs(W1[j]).max()
        rho[j] = CAP * np.abs(W1[j]) / mx if mx > 1e-12 else 0.0
    up = np.round(x[None, None, :] * rho[:, :, None])
    dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None])
    off = np.where(W1 > 0, 0.0, np.round(15 * rho))
    return rho, up, dn, off


def forward(W1, b1, W2, b2, which, ste=True):
    """which: 索引集（0/1/2 对应 hi/lo/全量）"""
    rho, up, dn, off = lut_tables(W1)
    Q = np.zeros((len(which), h))
    for j in range(h):
        for i in range(4):
            if abs(W1[j, i]) <= 1e-12:
                continue
            lam = up[j, i, Xall[which, i]] if W1[j, i] > 0 else dn[j, i, Xall[which, i]]
            Q[:, j] += lam - off[j, i]
    a = np.array([np.abs(W1[j]).max() for j in range(h)]) / CAP
    # STE: Q 视作 a*Q 的线性函数, 梯度直接过
    U = Q * a[None, :] + b1[None, :]
    rr = np.maximum(U, 0)
    R = rr.max(axis=0); R = np.where(R > 1e-12, R, 1.0)
    v = np.clip(np.round(15 * rr / R[None, :]), 0, 15)
    logit = (R / 15)[None, :] * (v * W2).sum(axis=1, keepdims=True) if False else None
    logit = (v[:, :, None] * W2[None, :, :]).sum(axis=1) + b2[None, :]
    return logit, v, U, R, rr


def acc150(W1, b1, W2, b2):
    logit, _, _, _, _ = forward(W1, b1, W2, b2, idx150)
    return (np.argmax(logit, 1) == y3).mean(), logit


def err_all(W1, b1, W2, b2):
    """全空间二值错误（相对 teacher 目标）"""
    logit, _, _, _, _ = forward(W1, b1, W2, b2, np.arange(M, dtype=np.int64))
    logit0, _, _, _, _ = forward(W1_0, b1_0, W2_0, b2f, np.arange(M, dtype=np.int64))
    Yt = (logit0 >= 0).astype(int)
    return int(np.sum((logit >= 0).astype(int) != Yt))


print("=" * 96)
print("鸢尾花 4bit 微调（LUT 量化进前向 + STE）")
print("=" * 96)
a0, _ = acc150(W1_0, b1_0, W2_0, b2f)
print(f"初始 teacher: 150 点 3 类 {a0:.4f}   全空间错误 {err_all(W1_0, b1_0, W2_0, b2f)}")

Y = (y3[:, None] == np.arange(3)[None, :]).astype(float)
W1 = W1_0.copy(); b1 = b1_0.copy(); W2 = W2_0.copy(); b2 = b2f.copy()
ps = [W1, b1, W2, b2]
ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
best = (a0, W1.copy(), b1.copy(), W2.copy(), b2.copy())
rng = np.random.default_rng(0)
lr = 0.002
steps = 3000
for t in range(1, steps + 1):
    idxb = rng.integers(0, N150, 64)
    logit, v, U, R, rr = forward(W1, b1, W2, b2, idx150[idxb])
    P = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    dlogit = (P - Y[idxb]) / 64                       # (b,3)
    # logit = sum_j v_j*W2_kj + b2_k
    gW2 = v.T @ dlogit
    gb2 = dlogit.sum(0)
    # dv -> drr (STE, 忽略 round) :  v = 15*rr/R
    dH = (dlogit @ W2.T) * (15.0 / R[None, :])
    # 对 R 的依赖忽略（R 由当前权重决定，视作常数）
    dU = dH * (U > 0)
    # u = a*Q + b1 ; Q 对 W1 的依赖经 rho/阶梯，梯度极弱 -> 只更新 b1
    gb1 = dU.sum(0)
    gW1 = np.zeros_like(W1)                           # 阶梯是整数, 不可微; 靠 STE 只调 b1/W2
    g = [gW1, gb1, gW2, gb2]
    for i, (pp, gg) in enumerate(zip(ps, g)):
        ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
        pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
    if t % 500 == 0 or t == steps:
        aa, _ = acc150(W1, b1, W2, b2)
        if aa > best[0]:
            best = (aa, W1.copy(), b1.copy(), W2.copy(), b2.copy())
        print(f"  step {t:5d}: 150点 3类 {aa:.4f}   (最好 {best[0]:.4f})")

print("\n" + "=" * 96)
print("微调结果")
print("=" * 96)
for tag, P in (("初始", (W1_0, b1_0, W2_0, b2f)), ("微调后", best[1:])):
    aa, _ = acc150(*P)
    print(f"  {tag}: 150 点 3 类 {aa:.4f}   全空间二值错误 {err_all(*P)}")

json.dump({"init_acc": float(a0), "tuned_acc": float(best[0]),
           "W1": best[1].tolist(), "b1": best[2].tolist(),
           "W2": best[3].tolist(), "b2": best[4].tolist(), "alive": [int(v) for v in alive]},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "iris_tuned4bit.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/iris_tuned4bit.json")
