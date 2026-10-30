#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAN 梯度检验：定位普通 GAN 为什么不学
========================================
用数值梯度 (有限差分) 对照解析梯度，找出 bug。
检验对象:
  D 对所有参数的梯度
  G 对所有参数的梯度
"""
import numpy as np

D_Z, H_G, H_D = 8, 16, 8
BS = 6
rng = np.random.default_rng(0)


def init(dims, seed=0, scale=None):
    r = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        s = scale if scale else np.sqrt(2.0 / dims[i])
        P.append(r.normal(0, s, (dims[i], dims[i + 1])))
        P.append(np.zeros(dims[i + 1]))
    return P


def g_forward(P, Z):
    acts = [Z]; h = Z; L = len(P) // 2
    for i in range(L):
        h = h @ P[2 * i] + P[2 * i + 1]
        h = np.maximum(h, 0) if i < L - 1 else np.tanh(h)
        acts.append(h)
    return acts


def d_forward(P, X):
    acts = [X]; h = X; L = len(P) // 2
    for i in range(L):
        h = h @ P[2 * i] + P[2 * i + 1]
        if i < L - 1:
            h = np.where(h > 0, h, 0.2 * h)
        acts.append(h)
    return acts


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def bce_with_logits(logit, target):
    return np.mean(np.maximum(logit, 0) - logit * target + np.log1p(np.exp(-np.abs(logit))))


Z = rng.normal(0, 1, (BS, D_Z))
Xr = rng.normal(0, 0.6, (BS, 64))
G = init([D_Z, H_G, 64], seed=1)
D = init([64, H_D, 1], seed=2, scale=0.1)

# ---------------- D 的梯度 ----------------
Xf = g_forward(G, Z)[-1]
ar = d_forward(D, Xr); af = d_forward(D, Xf)
L_D = bce_with_logits(ar[-1], 0.9) + bce_with_logits(af[-1], 0.0)

d_ar = (sig(ar[-1]) - 0.9) / BS
d_af = (sig(af[-1]) - 0.0) / BS
g = {}
g['D.W2'] = ar[-2].T @ d_ar + af[-2].T @ d_af
g['D.b2'] = d_ar.sum(0) + d_af.sum(0)
dHr = (d_ar @ D[2].T) * np.where(ar[-2] > 0, 1.0, 0.2)
dHf = (d_af @ D[2].T) * np.where(af[-2] > 0, 1.0, 0.2)
g['D.W1'] = Xr.T @ dHr + Xf.T @ dHf
g['D.b1'] = dHr.sum(0) + dHf.sum(0)


def D_loss_from(Dp):
    a = d_forward(Dp, Xr); b = d_forward(Dp, Xf)
    return bce_with_logits(a[-1], 0.9) + bce_with_logits(b[-1], 0.0)


print("=" * 78)
print("判别器 D 梯度检验")
print("=" * 78)
eps = 1e-5
names = ['D.W1', 'D.b1', 'D.W2', 'D.b2']
for i, nm in enumerate(names):
    idx = names.index(nm)
    Dp = [p.copy() for p in D]
    num = np.zeros_like(Dp[idx])
    flat = Dp[idx].reshape(-1)
    for k in range(min(flat.size, 12)):
        old = flat[k]
        flat[k] = old + eps
        lp = D_loss_from(Dp)
        flat[k] = old - eps
        lm = D_loss_from(Dp)
        flat[k] = old
        num.reshape(-1)[k] = (lp - lm) / (2 * eps)
    ana = g[nm].reshape(-1)[:12]
    num = num.reshape(-1)[:12]
    err = np.max(np.abs(ana - num) / (np.abs(ana) + np.abs(num) + 1e-8))
    print(f"  {nm:>6}: 最大相对误差 {err:.3e}   {'OK' if err < 1e-4 else '*** 错误 ***'}")
    if err >= 1e-4:
        print(f"          解析: {np.round(ana,6)}")
        print(f"          数值: {np.round(num,6)}")

# ---------------- G 的梯度 ----------------
Xf = g_forward(G, Z)[-1]
ad = d_forward(D, Xf)
L_G = bce_with_logits(ad[-1], 0.9)
d_ad = (sig(ad[-1]) - 0.9) / BS
dHd = (d_ad @ D[2].T) * np.where(ad[-2] > 0, 1.0, 0.2)
dPre = (dHd @ D[0].T) * (1 - Xf ** 2)
gg = {}
ag = g_forward(G, Z)
gg['G.W2'] = ag[-2].T @ dPre
gg['G.b2'] = dPre.sum(0)
dH = (dPre @ G[2].T) * (ag[-2] > 0)
gg['G.W1'] = Z.T @ dH
gg['G.b1'] = dH.sum(0)


def G_loss_from(Gp):
    X = g_forward(Gp, Z)[-1]
    a = d_forward(D, X)
    return bce_with_logits(a[-1], 0.9)


print()
print("=" * 78)
print("生成器 G 梯度检验")
print("=" * 78)
namesG = ['G.W1', 'G.b1', 'G.W2', 'G.b2']
for idx, nm in enumerate(namesG):
    Gp = [p.copy() for p in G]
    num = np.zeros_like(Gp[idx])
    flat = Gp[idx].reshape(-1)
    for k in range(min(flat.size, 12)):
        old = flat[k]
        flat[k] = old + eps
        lp = G_loss_from(Gp)
        flat[k] = old - eps
        lm = G_loss_from(Gp)
        flat[k] = old
        num.reshape(-1)[k] = (lp - lm) / (2 * eps)
    ana = gg[nm].reshape(-1)[:12]
    num = num.reshape(-1)[:12]
    err = np.max(np.abs(ana - num) / (np.abs(ana) + np.abs(num) + 1e-8))
    print(f"  {nm:>6}: 最大相对误差 {err:.3e}   {'OK' if err < 1e-4 else '*** 错误 ***'}")
    if err >= 1e-4:
        print(f"          解析: {np.round(ana,6)}")
        print(f"          数值: {np.round(num,6)}")

# ---------------- 判别能力测试 ----------------
print()
print("=" * 78)
print("判别器收敛测试（只训 D，看能否把真假分开）")
print("=" * 78)
D2 = init([64, H_D, 1], seed=2, scale=0.1)
m = [np.zeros_like(p) for p in D2]; v = [np.zeros_like(p) for p in D2]
for t in range(1, 2001):
    a = d_forward(D2, Xr); b = d_forward(D2, Xf)
    da = (sig(a[-1]) - 0.9) / BS; db = (sig(b[-1]) - 0.0) / BS
    gw2 = a[-2].T @ da + b[-2].T @ db; gb2 = da.sum(0) + db.sum(0)
    dha = (da @ D2[2].T) * np.where(a[-2] > 0, 1.0, 0.2)
    dhb = (db @ D2[2].T) * np.where(b[-2] > 0, 1.0, 0.2)
    gw1 = Xr.T @ dha + Xf.T @ dhb; gb1 = dha.sum(0) + dhb.sum(0)
    for i, (p_, g_) in enumerate(zip(D2, [gw1, gb1, gw2, gb2])):
        m[i] = 0.9 * m[i] + 0.1 * g_; v[i] = 0.999 * v[i] + 0.001 * g_ ** 2
        p_ -= 1e-3 * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    if t % 500 == 0:
        pr = sig(d_forward(D2, Xr)[-1]).mean()
        pf = sig(d_forward(D2, Xf)[-1]).mean()
        print(f"  step {t:5d}: D(真)={pr:.4f}  D(假)={pf:.4f}")
print("  -> 若 D 能分开 (真->1, 假->0)，说明 D 实现正确，问题在 GAN 动力学")
