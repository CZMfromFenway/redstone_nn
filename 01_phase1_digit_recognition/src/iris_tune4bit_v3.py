#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花 4bit 微调 v3（正确口径：直接优化取整后的整数权重）
============================================================
【口径修正】
  错误做法: logit = v · (R/lv) · W2 + b2          <- R/lv 打乱了三路输出的相对尺度
  正确做法: A = round(S * (R/lv) * W2),  th = round(S * b2)
            logit = v · A + th
            （两者在同一 S 下取整, 取整误差量级一致, 比例才正确）

【可微参数化（STE）】
  令 W2t = (R/lv) * W2   （逐通道因子先融合进去）
      A  = round(S * W2t)
      th = round(S * b2)
  反传时把 round 视作恒等, 于是 d/dW2t = S * (dv 侧梯度), d/db2 = S * (...)

【自检】初始点必须复现 iris_compiled_v3 的 0.9867 / 全空间 3023
"""
import numpy as np, json, os

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1f = np.array(d['W1']); b1f = np.array(d['b1'])
W2f = np.array(d['W2']); b2f = np.array(d['b2'])
Xq = np.array(d['Xq'], dtype=np.int64); y3 = np.array(d['y3'])
N150 = len(y3)
CAP = 868.0; x = np.arange(16); LV = 15.0
Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
M = len(Xall)

Uf = Xall.astype(np.float64) @ W1f.T + b1f[None, :]
alive = [j for j in range(W1f.shape[0]) if np.maximum(Uf[:, j], 0).max() > 1e-9]
h = len(alive)
W1 = W1f[alive]; b1 = b1f[alive]; W2_0 = W2f[alive]
idx150 = Xq[:, 0] * 4096 + Xq[:, 1] * 256 + Xq[:, 2] * 16 + Xq[:, 3]


def lut_and_Q(rows):
    rho = np.zeros((h, 4))
    for j in range(h):
        mx = np.abs(W1[j]).max(); rho[j] = CAP * np.abs(W1[j]) / mx
    up = np.round(x[None, None, :] * rho[:, :, None])
    dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None])
    off = np.where(W1 > 0, 0.0, np.round(15 * rho))
    Q = np.zeros((len(rows), h))
    for j in range(h):
        for i in range(4):
            if abs(W1[j, i]) <= 1e-12:
                continue
            lam = up[j, i, Xall[rows, i]] if W1[j, i] > 0 else dn[j, i, Xall[rows, i]]
            Q[:, j] += lam - off[j, i]
    a = np.array([np.abs(W1[j]).max() for j in range(h)]) / CAP
    U = Q * a[None, :] + b1[None, :]
    return U


U_cache = {}


def get_U(rows):
    key = rows.tobytes()
    if key not in U_cache:
        U_cache[key] = lut_and_Q(rows)
    return U_cache[key]


def forward(rows, W2, b2, S):
    U = get_U(rows)
    rr = np.maximum(U, 0)
    R = np.maximum(U, 0).max(axis=0)               # 固定用满空间的 R（与编译一致）
    R = np.where(R > 1e-12, R, 1.0)
    v = np.clip(np.round(LV * rr / R[None, :]), 0, LV)
    W2t = (R / LV)[None, :] * W2
    A = np.round(S * W2t)
    th = np.round(S * b2)
    logit = v @ A.T + th[None, :]
    return logit, v, A, th, W2t, R


# 固定 R（与 iris_compiled_v3 完全一致，用全空间 U）
U_all = lut_and_Q(np.arange(M, dtype=np.int64))
R_fix = np.maximum(U_all, 0).max(axis=0)


def forward2(rows, W2, b2, S):
    U = get_U(rows)
    rr = np.maximum(U, 0)
    v = np.clip(np.round(LV * rr / R_fix[None, :]), 0, LV)
    W2t = (R_fix / LV)[None, :] * W2
    A = np.round(S * W2t)
    th = np.round(S * b2)
    return v @ A.T + th[None, :], v, A, th, W2t


def acc150(W2, b2, S):
    logit, _, _, _, _ = forward2(idx150, W2, b2, S)
    return (np.argmax(logit, 1) == y3).mean()


def err_all(W2, b2, S):
    logit, _, _, _, _ = forward2(np.arange(M, dtype=np.int64), W2, b2, S)
    logit0, _, _, _, _ = forward2(np.arange(M, dtype=np.int64), W2_0, b2f, S)
    Yt = (logit0 >= 0).astype(int)
    return int(np.sum((logit >= 0).astype(int) != Yt))


print("=" * 96)
print("鸢尾花 4bit 微调 v3（正确口径）")
print("=" * 96)
print(f"alive={alive}  R_fix={np.round(R_fix,6).tolist()}")
S = 50
a0, e0 = acc150(W2_0, b2f, S), err_all(W2_0, b2f, S)
print(f"[自检] S=50 初始点: 150点 3类 {a0:.4f}  全空间错误 {e0}"
      f"   (iris_compiled_v3: 0.9867 / 3023)")
print("  自检 " + ("通过 ✓" if abs(a0 - 0.9867) < 1e-9 else "失败"))

Y = (y3[:, None] == np.arange(3)[None, :]).astype(float)
for S in (25, 50, 100):
    W2 = W2_0.copy(); b2 = b2f.copy()
    ms = [np.zeros_like(W2), np.zeros_like(b2)]; vs = [np.zeros_like(W2), np.zeros_like(b2)]
    ps = [W2, b2]
    best = (acc150(W2, b2, S), W2.copy(), b2.copy())
    rng = np.random.default_rng(0); lr = 0.05
    for t in range(1, 3001):
        rows = rng.integers(0, N150, 64)
        logit, v, A, th, W2t = forward2(idx150[rows], W2, b2, S)
        P = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        dlogit = (P - Y[rows]) / len(rows)
        gA = v.T @ dlogit                      # (h,3)
        gth = dlogit.sum(0)
        gW2t = S * gA                          # STE: A = round(S*W2t)
        gW2 = gW2t * (R_fix / LV)[None, :]
        gb2 = S * gth
        g = [gW2, gb2]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == 3000:
            aa = acc150(W2, b2, S)
            if aa > best[0]:
                best = (aa, W2.copy(), b2.copy())
            print(f"  S={S:4d} step {t:5d}: 150点 {aa:.4f}  (最好 {best[0]:.4f})")
    print(f"  => S={S}: 最优 {best[0]:.4f}   全空间错误 {err_all(best[1], best[2], S)}")
    json.dump({"S": S, "init_acc": float(a0), "tuned_acc": float(best[0]),
               "W2": best[1].tolist(), "b2": best[2].tolist()},
              open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "results", f"iris_tuned4bit_S{S}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
print("\n完成。")
