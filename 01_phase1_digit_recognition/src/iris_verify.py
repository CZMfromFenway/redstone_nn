#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核实: (1) CV 是否真的用了被选中的权重 (2) 死神经元问题的普遍性"""
import numpy as np, json, os, importlib.util

p = r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\src\iris_teacher.py'
src = open(p, encoding='utf-8').read().split("N_SEEDS = 40")[0]
ns = {}
exec(compile(src, p, 'exec'), ns)
Xr = ns['Xr']; y3 = ns['y3']; N = ns['N']; Xq = ns['Xq']; Y = ns['Y']
train_relu = ns['train_relu']

print("=" * 92)
print("核实 1: 被选中 teacher 的权重, 直接在 5 折上评（不做折内重训）")
print("=" * 92)
d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1 = np.array(d['W1']); b1 = np.array(d['b1']); W2 = np.array(d['W2']); b2 = np.array(d['b2'])
Z = np.maximum(Xq @ W1.T + b1, 0) @ W2 + b2
acc_full = (np.argmax(Z, 1) == y3).mean()
print(f"  该 teacher 在全量 150 (全量量化) 上: 3类 {acc_full:.4f}")
print(f"  错分样本: {np.where(np.argmax(Z,1) != y3)[0].tolist()}")
print(f"  真实标签 {y3[np.argmax(Z,1) != y3].tolist()}  预测 {np.argmax(Z,1)[np.argmax(Z,1)!=y3].tolist()}")
print()

print("=" * 92)
print("核实 2: 死神经元的普遍性（40 个种子）")
print("=" * 92)
dead_cnt = 0; alive_dist = []
for s in range(40):
    r = train_relu(Xq, Y, 4, seed=1000 + s)
    W1s, b1s = r[2], r[3]
    H = np.maximum(Xq @ W1s.T + b1s, 0)
    alive = int((H.max(0) > 1e-9).sum())
    alive_dist.append(alive)
    if alive < 4:
        dead_cnt += 1
import collections
print(f"  存活神经元数分布: {dict(sorted(collections.Counter(alive_dist).items()))}")
print(f"  含死神经元的种子: {dead_cnt}/40")
print()

print("=" * 92)
print("核实 3: 若强制 4 个神经元全存活, 能否达到更高准确率")
print("=" * 92)
# 用更强正则 + 更多步 + 不同初始化尺度
best = None
for s in range(60):
    rng = np.random.default_rng(7000 + s)
    W1i = rng.normal(0, 0.5, (4, 4)); b1i = np.abs(rng.normal(0, 0.5, 4))  # 正偏置, 减少死区
    W2i = rng.normal(0, 0.5, (4, 3)); b2i = rng.normal(0, 0.2, 3)
    # 内联训练（带正偏置初始化）
    X, Yt, h = Xq, Y, 4
    W1, b1c, W2c, b2c = W1i.copy(), b1i.copy(), W2i.copy(), b2i.copy()
    ps = [W1, b1c, W2c, b2c]; ms = [np.zeros_like(q) for q in ps]; vs = [np.zeros_like(q) for q in ps]
    rng2 = np.random.default_rng(7000 + s)
    bb = None
    for t in range(1, 9001):
        idx = rng2.integers(0, N, 32)
        xb = X[idx]; yb = Yt[idx]
        U = xb @ W1.T + b1c; Hf = np.maximum(U, 0); Zt = Hf @ W2c + b2c
        P = 1/(1+np.exp(-np.clip(Zt,-30,30))); dZ = (P-yb)/32
        dU = (dZ @ W2c.T) * (U > 0)
        g = [dU.T@xb, dU.sum(0), Hf.T@dZ, dZ.sum(0)]
        for i,(pp,gg) in enumerate(zip(ps,g)):
            ms[i]=0.9*ms[i]+0.1*gg; vs[i]=0.999*vs[i]+0.001*gg**2
            pp -= 0.02*(ms[i]/(1-0.9**t))/(np.sqrt(vs[i]/(1-0.999**t))+1e-8)
        if t % 3000 == 0 or t == 9000:
            Zf = np.maximum(X@W1.T+b1c,0)@W2c+b2c
            a3 = (np.argmax(Zf,1)==np.argmax(Yt,1)).mean()
            Hx = np.maximum(X@W1.T+b1c,0); al = int((Hx.max(0)>1e-9).sum())
            if bb is None or a3 > bb[0]:
                bb = (a3, al, W1.copy(), b1c.copy(), W2c.copy(), b2c.copy())
    if best is None or bb[0] > best[0]:
        best = bb
print(f"  最好: 3类 {best[0]:.4f}  存活 {best[1]}/4")
