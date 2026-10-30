#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花 4bit 微调 v2（修正 R_j/lv 因子 + 完整反传）
====================================================
【修正】前向的 logit 必须含 R_j/lv 因子:
    v_j = round(15*r_j/R_j)
    logit_k = sum_j (R_j/15)*W2_kj*v_j + b2_k      <- 少了 R_j/15 就完全不对
【反传】R_j = max r_j 依赖权重, 也计入梯度:
    dv/drr = (15/R) + (-15*rr/R^2)*dR/drr
    其中 dR/drr 只在取到 max 的那个样本上为 1

自检: 初始点应精确复现 iris_compiled_v3 的 150 点 0.9867
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

Uf = Xall.astype(np.float64) @ W1f.T + b1f[None, :]
alive = [j for j in range(W1f.shape[0]) if np.maximum(Uf[:, j], 0).max() > 1e-9]
h = len(alive)
W1_0 = W1f[alive]; b1_0 = b1f[alive]; W2_0 = W2f[alive]

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


def forward(W1, b1, W2, b2, rows, lv=15):
    rho, up, dn, off = lut_tables(W1)
    Q = np.zeros((len(rows), h))
    for j in range(h):
        for i in range(4):
            if abs(W1[j, i]) <= 1e-12:
                continue
            lam = up[j, i, Xall[rows, i]] if W1[j, i] > 0 else dn[j, i, Xall[rows, i]]
            Q[:, j] += lam - off[j, i]
    a = np.array([np.abs(W1[j]).max() for j in range(h)]) / CAP
    U = Q * a[None, :] + b1[None, :]
    rr = np.maximum(U, 0)
    R = rr.max(axis=0); R = np.where(R > 1e-12, R, 1.0)
    v = np.clip(np.round(lv * rr / R[None, :]), 0, lv)
    logit = (v * (R / lv)[None, :]) @ W2.T + b2[None, :]
    return logit, v, U, R, rr


def acc150(W1, b1, W2, b2, lv=15):
    logit, _, _, _, _ = forward(W1, b1, W2, b2, idx150, lv)
    return (np.argmax(logit, 1) == y3).mean()


def err_all(W1, b1, W2, b2, lv=15):
    logit, _, _, _, _ = forward(W1, b1, W2, b2, np.arange(M, dtype=np.int64), lv)
    logit0, _, _, _, _ = forward(W1_0, b1_0, W2_0, b2f, np.arange(M, dtype=np.int64), lv)
    Yt = (logit0 >= 0).astype(int)
    return int(np.sum((logit >= 0).astype(int) != Yt))


print("=" * 96)
print("鸢尾花 4bit 微调 v2（修正 R/lv 因子）")
print("=" * 96)
a0 = acc150(W1_0, b1_0, W2_0, b2f)
print(f"[自检] 初始点 150 点 3 类 = {a0:.4f}   (iris_compiled_v3 报的是 0.9867)")
print(f"       全空间二值错误 = {err_all(W1_0, b1_0, W2_0, b2f)}  (v3 报的是 3023)")
if abs(a0 - 0.9867) < 1e-6:
    print("       自检通过 ✓")
else:
    print("       !! 与 v3 不一致, 需检查")

Y = (y3[:, None] == np.arange(3)[None, :]).astype(float)
lv = 15.0
W1 = W1_0.copy(); b1 = b1_0.copy(); W2 = W2_0.copy(); b2 = b2f.copy()
ps = [W1, b1, W2, b2]
ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
best = (a0, W1.copy(), b1.copy(), W2.copy(), b2.copy())
rng = np.random.default_rng(0)
lr = 0.005
for t in range(1, 4001):
    rows = rng.integers(0, N150, 64)
    logit, v, U, R, rr = forward(W1, b1, W2, b2, idx150[rows], lv)
    B = len(rows)
    P = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    dlogit = (P - Y[rows]) / B
    # logit_k = sum_j (R_j/lv)*v_j*W2_kj + b2_k
    gW2 = ((R / lv)[None, :] * v).T @ dlogit
    gb2 = dlogit.sum(0)
    # d/dv_j
    dHv = (dlogit @ W2.T) * (R / lv)[None, :]
    # v = lv*rr/R  -> dv/drr = lv/R - lv*rr/R^2 * dR/drr,  dR/drr = 1 at argmax
    argmax = np.argmax(rr, axis=0)                    # (h,)
    dH = dHv * (lv / R[None, :])
    corr = np.zeros_like(dH)
    for j in range(h):
        corr[argmax[j], j] = -dHv[argmax[j], j] * lv * rr[argmax[j], j] / (R[j] ** 2)
    dH = dH + corr
    dU = dH * (U > 0)
    gb1 = dU.sum(0)
    gW1 = np.zeros_like(W1)      # 阶梯为整数, 不可微
    g = [gW1, gb1, gW2, gb2]
    for i, (pp, gg) in enumerate(zip(ps, g)):
        ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
        pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
    if t % 500 == 0 or t == 4000:
        aa = acc150(W1, b1, W2, b2, lv)
        if aa > best[0]:
            best = (aa, W1.copy(), b1.copy(), W2.copy(), b2.copy())
        print(f"  step {t:5d}: 150点 3类 {aa:.4f}  (最好 {best[0]:.4f})")

print("\n" + "=" * 96)
print("结果")
print("=" * 96)
for tag, P in (("初始 teacher", (W1_0, b1_0, W2_0, b2f)), ("微调后", best[1:])):
    aa = acc150(*P, lv); ee = err_all(*P, lv)
    print(f"  {tag:14s}: 150 点 3 类 {aa:.4f}   全空间二值错误 {ee}")

json.dump({"init_acc": float(a0), "tuned_acc": float(best[0]),
           "init_err_all": int(err_all(W1_0, b1_0, W2_0, b2f, lv)),
           "tuned_err_all": int(err_all(*best[1:], lv)),
           "W1": best[1].tolist(), "b1": best[2].tolist(),
           "W2": best[3].tolist(), "b2": best[4].tolist(),
           "alive": [int(v) for v in alive]},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "iris_tuned4bit_v2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/iris_tuned4bit_v2.json")
