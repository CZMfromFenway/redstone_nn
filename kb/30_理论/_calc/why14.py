# -*- coding: utf-8 -*-
"""查 b=14 时 eps_y 为何很小但非零：逐通道看 Gamma、N-P 的范围与负半边比例。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_baseline2 as E

b, n, m = 14, 16, 16
rng = np.random.default_rng(1000 * b + 17 * n + 0)
w = rng.normal(0.0, 1.0 / np.sqrt(n), size=(m, n))
th = np.zeros(m)
x = rng.uniform(E.X_MIN, E.X_MAX, size=(E.NSAMP, n))
delta = E.DELTA / 2 ** b
r = E.layer_int(w, th, x, E.WREF, b, delta)

N, P, G, sig = r['N'], r['P'], r['Gamma'], r['sigma']
eta_int = (N - P) / sig
y_hat = np.maximum(r['Y'] * G / sig, 0)
y_ref = np.maximum(eta_int, 0)
span = y_ref.max(0) - y_ref.min(0)
err = np.sqrt(np.mean((y_hat - y_ref) ** 2, axis=0)) / np.maximum(span, 1e-12)

D = N - P
print('2^b-1 =', 2 ** b - 1)
print('每通道 (N-P) 最大值 / Gamma / 负值比例 / eps_y')
for j in range(m):
    neg = float(np.mean(D[:, j] < 0))
    print('  ch%2d  max(N-P)=%7.0f  Gamma=%3.0f  neg=%4.2f  eps_y=%.2e' % (
        j, D[:, j].max(), G[j], neg, err[j]))
print()
print('全局 max(N-P) =', D.max(), ' 相对 2^b-1 =', D.max() / (2 ** b - 1))
print('Gamma 取值集合:', np.unique(G))
print('eps_y 均值 =', err.mean())
# 误差来自哪里：只在 Y != N-P 的地方
diff = (r['Y'] - D)
print('Y != N-P 的样本比例 =', float(np.mean(np.abs(diff) > 0.5)))
