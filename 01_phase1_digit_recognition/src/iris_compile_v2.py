#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花 Teacher 编译 v2：第一层按"折叠阶梯"消除巨大常数偏移
==============================================================
【关键修正】
  直接相加:   Q_j = sum_i lambda_i   ->  u_j = a_j*Q_j - sum|W1| + b1
              常数 sum|W1| 使 Q 的绝对值很大, 量化步长被迫按它定 -> 误差 58%
  折叠相加:   Q'_j = sum_i (lambda_i - lambda_i(0))
              u_j = a_j*Q'_j + [a_j*sum_i lambda_i(0) - sum|W1| + b1]
              Q' 的范围 = 真实变化范围, 量化误差按变化范围定 -> 误差大降
  lambda_i(0): 正权重(上阶梯)=0 ; 负权重(下阶梯)=15*rho
  两者都是非负常数, 物理上只是把该坐标的阈值平移 -> **完全可建造**

【中间层】ReLU 线性转弱模: v_j = round(lv * max(0,u_j)/R_j)
【输出层】整数权重, 逐通道 R_j/lv 因子吸收进权重
"""
import numpy as np, json, os

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1 = np.array(d['W1']); b1 = np.array(d['b1'])
W2 = np.array(d['W2']); b2 = np.array(d['b2'])
Xq = np.array(d['Xq'], dtype=np.int64); y3 = np.array(d['y3'])
N = len(y3); h = W1.shape[0]
CAP = 868.0; x = np.arange(16)
Y3oh = (y3[:, None] == np.arange(3)[None, :]).astype(int)

print("=" * 100)
print("鸢尾花 Teacher 编译 v2（折叠阶梯）")
print("=" * 100)

rho = np.zeros((h, 4))
for j in range(h):
    mx = np.abs(W1[j]).max()
    rho[j] = CAP * np.abs(W1[j]) / mx if mx > 1e-12 else 0.0
up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)
off = np.zeros((h, 4), dtype=np.int64)
for j in range(h):
    for i in range(4):
        off[j, i] = 0 if W1[j, i] > 0 else int(round(15 * rho[j, i]))

print(f"可建造性: lam_up [{up.min()},{up.max()}] 负值 {int((up<0).sum())}")
print(f"          lam_dn [{dn.min()},{dn.max()}] 负值 {int((dn<0).sum())}")
print(f"          单档 rho_max {rho.max():.1f} <= 897;  满档 {15*rho.max():.0f} <= 13455")

Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
Xi = Xall.astype(np.int64); M = len(Xall)
Q = np.zeros((M, h), dtype=np.int64)
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        lam = (up if W1[j, i] > 0 else dn)[j, i, Xi[:, i]]
        Q[:, j] += lam - off[j, i]                 # 折叠
mx = np.array([np.abs(W1[j]).max() for j in range(h)])
# 尺度: u_j = a_j * Q_j + b1_j,  a_j = max_i|W1_ji| / 868
a = mx / CAP
U = Q * a[None, :] + b1[None, :]        # 折叠后常数项恰为 b1

# 与真值对比（全空间）
Utrue = (Xall.astype(np.float64) @ W1.T) + b1[None, :]
print(f"\n[折叠后的量化误差] 全空间 max|U_lut - U_true| = {np.max(np.abs(U-Utrue)):.4f}"
      f"   (u 各列量程 {np.round(Utrue.max(0)-Utrue.min(0),2).tolist()})")

# 150 点上的 teacher 准确率（作为参照）
Uq = np.array([U[Xq[n,0]*4096 + Xq[n,1]*256 + Xq[n,2]*16 + Xq[n,3]] for n in range(N)])
Zq = np.maximum(Uq, 0) @ W2 + b2
print(f"  编译后 150 点 3类准确率（vs 真实标签）= {(np.argmax(Zq,1)==y3).mean():.4f}")

# ---------------- 中间层 + 输出层 ----------------
def evaluate(bits, S_list=(50, 100, 200, 500, 1000, 2000, 4000)):
    lv = (1 << bits) - 1
    R = np.maximum(U, 0).max(axis=0); R = np.where(R > 1e-12, R, 1.0)
    rr = np.maximum(U, 0)
    v4 = np.clip(np.round(lv * rr / R[None, :]), 0, lv).astype(np.int64)
    if bits == 4:
        Z = v4.astype(np.float64); Rj = R
    else:
        Z = np.concatenate([(v4 >> 4), (v4 & 0xF)], axis=1).astype(np.float64)
        Rj = np.concatenate([R, R])
    # teacher 目标（在编译后的 u 上，用浮点第二层）
    Yt = (np.maximum(U, 0) @ W2 + b2 >= 0).astype(int)
    out = {}
    Wt = (R / lv)[None, :] * W2.T                 # (3,h)
    for S_ in S_list:
        A = np.concatenate([np.round(S_ * 16 * Wt), np.round(S_ * Wt)], axis=1) if bits == 8 \
            else np.round(S_ * Wt)
        th = -np.round(S_ * b2)
        pred = (Z @ A.T + th[None, :] >= 0).astype(int)
        err = int(np.sum(pred != Yt)); wm = int(np.abs(A).max())
        out[S_] = (err, wm, A, th)
    return out, Z, R, lv, Yt

print("\n" + "=" * 100)
print("编译结果（全 65536 状态，目标 = teacher 在量化输入空间上的输出）")
print("=" * 100)
res_all = {}
for bits in (4, 8):
    out, Z, R, lv, Yt = evaluate(bits)
    print(f"\n--- 中间层 {bits} bit {'(1 条线/神经元)' if bits==4 else '(2 条线/神经元, 拆分)'} ---")
    print(f"  R_j = {np.round(R,4).tolist()}   lv={lv}")
    print(f"  逐通道因子 R_j/lv = {np.round(R/lv,6).tolist()}")
    print(f"  {'S':>6} | {'权重max':>8} | {'容器':>8} | {'全空间错误':>10} | {'错误率':>9}")
    print("  " + "-" * 58)
    for S_, (err, wm, A, th) in out.items():
        ok = "OK" if wm <= 897 else "越界%d" % wm
        print(f"  {S_:6d} | {wm:8d} | {ok:>8} | {err:10d} | {err/M*100:8.4f}%")
        res_all[f"mid{bits}bit_S{S_}"] = {"err": int(err), "wmax": int(wm),
                                          "err_rate": float(err/M)}
    # 保存最优（容器内）
    ok_S = [S_ for S_, (e, wm, A, t) in out.items() if wm <= 897]
    if ok_S:
        bestS = min(ok_S, key=lambda s: out[s][0])
        res_all[f"mid{bits}bit_best"] = {"S": int(bestS), "err": int(out[bestS][0]),
                                         "wmax": int(out[bestS][1])}

json.dump({"teacher": d, "results": res_all},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "iris_compile_v2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/iris_compile_v2.json")
