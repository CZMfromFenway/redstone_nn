#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花 4bit 微调 v4：整数坐标下降
====================================
【口径】内联复制 iris_compiled_v3 的已验证路径:
    A  = round(S * (R/lv) * W2)          (3 x h)
    th = round(S * b2)                   (3,)
    logit = v @ A.T + th                 其中 v = round(lv*relu(U)/R)
    判据 sign(logit)
  **自检**: S=50 时必须复现 150点 0.9867 / 全空间 3023

【微调】A 与 th 是整数, 只有 9+3=12 个参数 -> 直接整数坐标下降
"""
import numpy as np, json, os

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1f = np.array(d['W1']); b1f = np.array(d['b1'])
W2f = np.array(d['W2']); b2f = np.array(d['b2'])
Xq = np.array(d['Xq'], dtype=np.int64); y3 = np.array(d['y3'])
N150 = len(y3)
CAP = 868.0; x = np.arange(16); LV = 15
Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
M = len(Xall)

Uf = Xall.astype(np.float64) @ W1f.T + b1f[None, :]
alive = [j for j in range(W1f.shape[0]) if np.maximum(Uf[:, j], 0).max() > 1e-9]
h = len(alive)
W1 = W1f[alive]; b1 = b1f[alive]; W2 = W2f[alive]
idx150 = Xq[:, 0] * 4096 + Xq[:, 1] * 256 + Xq[:, 2] * 16 + Xq[:, 3]

# ---- 第一层（与 iris_compiled_v3 逐行一致）----
rho = np.zeros((h, 4))
for j in range(h):
    mx = np.abs(W1[j]).max()
    rho[j] = CAP * np.abs(W1[j]) / mx
up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)
off = np.where(W1 > 0, 0, np.round(15 * rho)).astype(np.int64)
Xi = Xall.astype(np.int64)
Q = np.zeros((M, h), dtype=np.int64)
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        Q[:, j] += (up if W1[j, i] > 0 else dn)[j, i, Xi[:, i]] - off[j, i]
a = np.array([np.abs(W1[j]).max() for j in range(h)]) / CAP
U = Q * a[None, :] + b1[None, :]

rr = np.maximum(U, 0)
R = rr.max(axis=0)
v_all = np.clip(np.round(LV * rr / R[None, :]), 0, LV).astype(np.int64)   # (M,h)

print("=" * 96)
print("鸢尾花 4bit 微调 v4：整数坐标下降")
print("=" * 96)
print(f"alive={alive}   R_j={np.round(R,6).tolist()}   R/lv={np.round(R/LV,6).tolist()}")


def make_Ath(S):
    A = np.round(S * ((R / LV)[None, :] * W2.T)).astype(np.int64)   # (3,h)
    th = np.round(S * b2f).astype(np.int64)
    return A, th


def logit_of(rows, A, th):
    return v_all[rows] @ A.T + th[None, :]


# ---- 自检 ----
Yt_all = None
print("\n[自检：必须与 iris_compiled_v3 一致]")
for S in (25, 50, 100):
    A, th = make_Ath(S)
    lg = logit_of(idx150, A, th)
    acc = (np.argmax(lg, 1) == y3).mean()
    print(f"  S={S:4d}: 150点 3类 {acc:.4f}   权重max {int(np.abs(A).max())}")
S0 = 50
A0, th0 = make_Ath(S0)
lg0 = logit_of(idx150, A0, th0)
acc0 = (np.argmax(lg0, 1) == y3).mean()
lg0_all = logit_of(np.arange(M, dtype=np.int64), A0, th0)
Yt = (lg0_all >= 0).astype(int)
err0 = int(np.sum((lg0_all >= 0).astype(int) != Yt))
print(f"  => 取 S={S0}: 150点 {acc0:.4f}  全空间(自比) {err0}  "
      f"{'自检通过 ✓' if abs(acc0-0.9867) < 1e-9 else '!! 与 0.9867 不符'}")

# ---- 用 teacher 的浮点 logit 作为全空间真值标签 ----
Utrue_all = Xall.astype(np.float64) @ W1f.T + b1f[None, :]
Yt_all3 = (np.maximum(Utrue_all[:, alive], 0) @ W2 + b2f >= 0).astype(int)
print(f"\n全空间二值真值(teacher 浮点) 正类数 {[int(Yt_all3[:,k].sum()) for k in range(3)]}")

def err_all_bin(A, th):
    lg = logit_of(np.arange(M, dtype=np.int64), A, th)
    return int(np.sum((lg >= 0).astype(int) != Yt_all3))

def err150_3(A, th):
    lg = logit_of(idx150, A, th)
    return int((np.argmax(lg, 1) != y3).sum())

print(f"初始: 150点错误 {err150_3(A0,th0)}   全空间二值错误 {err_all_bin(A0,th0)}")

# ---- 整数坐标下降 ----
A, th = A0.copy(), th0.copy()
cur150, curall = err150_3(A, th), err_all_bin(A, th)
print("\n[坐标下降] 目标: 先最小化 150 点错误, 再最小化全空间错误")
rng = np.random.default_rng(0)
for it in range(1, 21):
    improved = False
    for k in range(3):
        for j in range(h):
            for delta in (+1, -1, +2, -2, +5, -5):
                A2 = A.copy(); A2[k, j] += delta
                e1 = err150_3(A2, th)
                if e1 < cur150 or (e1 == cur150 and err_all_bin(A2, th) < curall):
                    A, cur150, curall = A2, e1, err_all_bin(A2, th)
                    improved = True
        for delta in (+1, -1, +2, -2, +5, -5, +10, -10):
            th2 = th.copy(); th2[k] += delta
            e1 = err150_3(A, th2)
            if e1 < cur150 or (e1 == cur150 and err_all_bin(A, th2) < curall):
                th, cur150, curall = th2, e1, err_all_bin(A, th2)
                improved = True
    print(f"  iter {it:2d}: 150点错误 {cur150}  全空间二值错误 {curall}  "
          f"权重max {int(np.abs(A).max())}  {'(有改进)' if improved else '(无改进)'}")
    if not improved:
        break

print("\n" + "=" * 96)
print("结果")
print("=" * 96)
print(f"  初始: 150 点错误 {err150_3(A0,th0)} ({1-err150_3(A0,th0)/N150:.4f})  "
      f"全空间二值错误 {err_all_bin(A0,th0)}")
print(f"  微调: 150 点错误 {cur150} ({1-cur150/N150:.4f})  全空间二值错误 {curall}")
print(f"  权重max: 初始 {int(np.abs(A0).max())} -> 微调 {int(np.abs(A).max())}  "
      f"{'OK' if np.abs(A).max()<=897 else '越界'}")
print(f"\n  A0 = {A0.tolist()}")
print(f"  A  = {A.tolist()}")
print(f"  th0= {th0.tolist()}")
print(f"  th = {th.tolist()}")

json.dump({"S": int(S0), "alive": [int(v) for v in alive], "R": R.tolist(),
           "A0": A0.tolist(), "th0": th0.tolist(), "A": A.tolist(), "th": th.tolist(),
           "err150_init": int(err150_3(A0, th0)), "err150_tuned": int(cur150),
           "errall_init": int(err_all_bin(A0, th0)), "errall_tuned": int(curall)},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "iris_tuned4bit_v4.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/iris_tuned4bit_v4.json")
