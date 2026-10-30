#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 为什么 4bit 输入下 h=8/16/32 结果完全相同"""
import numpy as np, importlib.util, os

p = r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\src\iris_mlp.py'
src = open(p, encoding='utf-8').read().split("# ---------------------------------------------------------------- 主流程")[0]
ns = {}
exec(compile(src, p, 'exec'), ns)
to_int = ns['to_int']; train_relu = ns['train_relu']; evaluate = ns['evaluate']
Xr = ns['Xr']; y3 = ns['y3']; N = ns['N']

Xi = to_int(Xr, 16).astype(float)
Y = np.zeros((N, 3))
for k in range(3):
    Y[:, k] = (y3 == k).astype(float)

print("4bit 输入: 不同取值组合数 =", len(np.unique(Xi, axis=0)), "/ 150 个样本")
Xf = ns['to_float'](Xr)
print("浮点输入: 不同取值组合数 =", len(np.unique(Xf, axis=0)), "/ 150")
X8 = ns['to_split16'](Xr)
print("8bit输入: 不同取值组合数 =", len(np.unique(X8, axis=0)), "/ 150")
print()

for tag, X in (("4bit", Xi), ("8bit", X8)):
    print(f"=== {tag} ===")
    for h in (8, 16, 32):
        b = None
        for seed in (0, 1, 2):
            r = train_relu(X, Y, h, seed=seed, steps=8000)
            if b is None or r[0] > b[0]:
                b = r
        ba, a3, per = evaluate(b, X, Y, y3)
        Hf = np.maximum(X @ b[1].T + b[2], 0)
        alive = int((Hf.max(axis=0) > 1e-9).sum())
        print(f"  h={h:2d}: 全量二值 {ba:.4f}  3类 {a3:.4f}  存活神经元 {alive}/{h}")
    print()
