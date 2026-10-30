#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花编译（干净版）：4 -> 3 -> 3
===================================
已查明: teacher 的 j=0 是死神经元, 真实结构 4->3->3, 150 点准确率 0.9933。

【流程】
  1. 载入 teacher, 剔除死神经元 -> 有效 W1(3x4), W2(3x3)
  2. 第一层 4->3: 每神经元每坐标一张 LUT
       rho = 868*|W1_ji|/max_i|W1_ji|
       正权重上阶梯(反式), 负权重下阶梯(顺式), 均非负
       折叠: Q_j = sum_i (lambda_i - lambda_i(0)),  u_j = (max_i|W1_ji|/868)*Q_j + b1_j
  3. 中间层: ReLU + 线性转弱模  v_j = round(lv*max(0,u_j)/R_j)
  4. 输出层: 整数权重, 逐通道 R_j/lv 因子吸收
  5. 在全部 65536 状态上评估, 目标 = teacher 的二值输出
"""
import numpy as np, json, os

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1f = np.array(d['W1']); b1f = np.array(d['b1'])
W2f = np.array(d['W2']); b2f = np.array(d['b2'])
CAP = 868.0; x = np.arange(16)

# ---- 剔除死神经元（在全空间上判断）----
Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
Uf = Xall.astype(np.float64) @ W1f.T + b1f[None, :]
alive = [j for j in range(W1f.shape[0]) if np.maximum(Uf[:, j], 0).max() > 1e-9]
W1 = W1f[alive]; b1 = b1f[alive]; W2 = W2f[alive]
h = len(alive); N = len(Xall)
print("=" * 96)
print(f"鸢尾花编译: 4 -> {h} -> 3 （剔除死神经元 {[j for j in range(W1f.shape[0]) if j not in alive]}）")
print("=" * 96)
print(f"W1 = {np.round(W1,4).tolist()}")
print(f"b1 = {np.round(b1,4).tolist()}")
print(f"W2 = {np.round(W2,4).tolist()}")
print(f"b2 = {np.round(b2f,4).tolist()}")

# ---- 第一层 LUT ----
rho = np.zeros((h, 4))
for j in range(h):
    mx = np.abs(W1[j]).max()
    rho[j] = CAP * np.abs(W1[j]) / mx
up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)
off = np.where(W1 > 0, 0, np.round(15 * rho)).astype(np.int64)

print(f"\n[可建造性] lam_up [{up.min()},{up.max()}] neg={int((up<0).sum())}; "
      f"lam_dn [{dn.min()},{dn.max()}] neg={int((dn<0).sum())}")
print(f"           单档 rho_max {rho.max():.1f} <=897;  满档 {15*rho.max():.0f} <=13455")

Xi = Xall.astype(np.int64)
Q = np.zeros((N, h), dtype=np.int64)
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        Q[:, j] += (up if W1[j, i] > 0 else dn)[j, i, Xi[:, i]] - off[j, i]
a = np.array([np.abs(W1[j]).max() for j in range(h)]) / CAP
U = Q * a[None, :] + b1[None, :]
Utrue = Xall.astype(np.float64) @ W1.T + b1[None, :]
print(f"\n[第一层精度] max|U_lut - U_true| = {np.max(np.abs(U-Utrue)):.4f}  "
      f"(u 各列量程 {np.round(Utrue.max(0)-Utrue.min(0),2).tolist()})")

# ---- 中间层 & 输出层 ----
def build(bits, S):
    lv = (1 << bits) - 1
    rr = np.maximum(U, 0)
    R = rr.max(axis=0); R = np.where(R > 1e-12, R, 1.0)
    v = np.clip(np.round(lv * rr / R[None, :]), 0, lv).astype(np.int64)
    Rj = R
    if bits == 4:
        Z = v.astype(np.float64)
    else:
        Z = np.concatenate([(v >> 4), (v & 0xF)], axis=1).astype(np.float64)
        Rj = np.concatenate([R, R])
    Wt = (R / lv)[None, :] * W2.T                       # (3, h)
    if bits == 4:
        A = np.round(S * Wt)
    else:
        A = np.concatenate([np.round(S * 16 * Wt), np.round(S * Wt)], axis=1)
    th = np.round(S * b2f)          # logit = Z@A - th
    return Z, A, th, R, lv

# teacher 的二值目标（在全空间上，用编译后的 U —— 保持自洽）
Yt = (np.maximum(U, 0) @ W2 + b2f >= 0).astype(int)

print("\n" + "=" * 96)
print("编译结果（全 65536 状态；目标 = teacher 在量化输入空间上的二值输出）")
print("=" * 96)
res = {}
for bits in (4, 8):
    print(f"\n--- 中间层 {bits} bit {'(1 条线/神经元)' if bits==4 else '(2 条线/神经元, 拆分)'} ---")
    best = None
    print(f"  {'S':>6} | {'权重max':>8} | {'容器':>9} | {'错误/65536':>10} | {'错误率':>9}")
    print("  " + "-" * 62)
    for S in (25, 50, 100, 200, 400, 800, 1600, 3200):
        Z, A, th, R, lv = build(bits, S)
        pred = (Z @ A.T - th[None, :] >= 0).astype(int)
        err = int(np.sum(pred != Yt))
        wm = int(np.abs(A).max())
        ok = "OK" if wm <= 897 else "越界"
        print(f"  {S:6d} | {wm:8d} | {ok:>9} | {err:10d} | {err/N*100:8.4f}%")
        if wm <= 897 and (best is None or err < best[1]):
            best = (S, err, wm, A, th, R)
    if best:
        S, err, wm, A, th, R = best
        print(f"  => 容器内最优: S={S}, 错误 {err}/{N} ({err/N*100:.4f}%), 权重max={wm}")
        print(f"     R_j = {np.round(R,6).tolist()}")
        print(f"     R_j/lv = {np.round(R/((1<<bits)-1),6).tolist()}")
        # 150 点 3 类
        Xq = np.array(d['Xq'], dtype=np.int64); y3 = np.array(d['y3'])
        idx150 = Xq[:,0]*4096 + Xq[:,1]*256 + Xq[:,2]*16 + Xq[:,3]
        Z150, A150, th150, _, _ = build(bits, S)
        pred150 = (Z150[idx150] @ A150.T - th150[None, :] >= 0).astype(int)
        # 3类: 三个 one-vs-rest 中取... teacher 用 argmax(logit)
        logit = Z150[idx150] @ A150.T - th150[None, :]
        acc3 = (np.argmax(logit, 1) == y3).mean()
        print(f"     150 点 3 类准确率 = {acc3:.4f}")
        res[f"mid{bits}bit"] = {"S": int(S), "err": int(err), "wmax": int(wm),
                                "R": R.tolist(), "acc3_150": float(acc3),
                                "A": A.tolist(), "th": th.tolist()}

json.dump({"alive": [int(v) for v in alive], "results": res,
           "W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2f.tolist()},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "iris_compiled.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/iris_compiled.json")
