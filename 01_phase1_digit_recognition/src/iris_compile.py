#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花 Teacher 编译成强模 LUT 结构
====================================
【结构】 4 -> 4 -> 3 (one-vs-rest, 三个二值输出)
【输入】 4 条 0~15 弱模线（4bit 量化）
【中间层】 ReLU 的线性转弱模：死区钳位 + 线性缩放 -> 0~15（或 0~255 拆分）

【第一层 4->4】每神经元每坐标一张 LUT
    rho_{j,i} = 868*|W1_ji| / max_i|W1_ji|
    W>0: 上阶梯 lambda_up(x)=round(x*rho)         (反式, x=0 处 0)
    W<0: 下阶梯 lambda_dn(x)=round(15*rho-x*rho)  (顺式, x=15 处 0), 非负
    u_j = a_j * sum_i lambda - sum_i|W1_ji| + b1_j ,  a_j = max_i|W1_ji|/(868*7.5)

【中间层】 r_j = max(0, u_j)
    v_j = round(lv * r_j / R_j) ,  R_j = max r_j   (lv = 15 或 255)
    这是逐通道线性映射 + 死区钳位，不需要查表

【输出层 4->3】整数权重
    A_j = round(S * W2_kj * R_j / lv)   （把 R_j/lv 这个逐通道因子吸收进权重）
    判据 sign(sum_j A_j*v_j + b2_k + const)
"""
import numpy as np, json, os

# ---------------- 载入 teacher ----------------
d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1 = np.array(d['W1']); b1 = np.array(d['b1'])
W2 = np.array(d['W2']); b2 = np.array(d['b2'])
Xq = np.array(d['Xq'], dtype=np.int64)
y3 = np.array(d['y3'])
N = len(y3); h = W1.shape[0]
CAP = 868.0; x = np.arange(16)

print("=" * 100)
print("鸢尾花 Teacher 编译：4 -> 4 -> 3 (one-vs-rest)")
print("=" * 100)
print(f"Teacher: 全量 150 样本 3 类准确率 {d['full_acc3']:.4f}")
print(f"W1 = {np.round(W1,4).tolist()}")
print(f"b1 = {np.round(b1,4).tolist()}")
print(f"W2 = {np.round(W2,4).tolist()}")
print(f"b2 = {np.round(b2,4).tolist()}")

# ---------------- 第一层 LUT ----------------
rho = np.zeros((h, 4))
for j in range(h):
    mx = np.abs(W1[j]).max()
    rho[j] = CAP * np.abs(W1[j]) / mx if mx > 1e-12 else 0.0
up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)

print(f"\n[第一层可建造性]")
print(f"  lam_up [{up.min()}, {up.max()}]  负值 {int((up<0).sum())}")
print(f"  lam_dn [{dn.min()}, {dn.max()}]  负值 {int((dn<0).sum())}")
print(f"  单档 rho_max = {rho.max():.1f} <= 897 ? {rho.max()<=897}")
print(f"  满档 15*rho_max = {15*rho.max():.0f} <= 13455 ? {15*rho.max()<=13455}")

# 全空间推算 u（用 LUT）
X6536 = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
Xi = X6536.astype(np.int64)
S = np.zeros((len(X6536), h), dtype=np.int64)
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        S[:, j] += (up if W1[j, i] > 0 else dn)[j, i, Xi[:, i]]
mx = np.array([np.abs(W1[j]).max() for j in range(h)])
a = mx / (CAP * 7.5)
U = S * a[None, :] - np.abs(W1).sum(axis=1)[None, :] + b1[None, :]

# 在 150 个真实输入点上与 teacher 对比
Uq = S[Xq[:, 0] * 4096 + Xq[:, 1] * 256 + Xq[:, 2] * 16 + Xq[:, 3]] * 0
for n_ in range(N):
    idx = Xq[n_, 0] * 4096 + Xq[n_, 1] * 256 + Xq[n_, 2] * 16 + Xq[n_, 3]
    Uq[n_] = U[idx]
Utrue_q = Xq @ W1.T + b1
print(f"\n[第一层量化误差] 150 点上的 max|U_lut - U_true| = "
      f"{np.max(np.abs(Uq - Utrue_q)):.3e}   (u 量程 {Utrue_q.max()-Utrue_q.min():.2f})")

# ---------------- 中间层 + 输出层 ----------------
def run(mid_bits):
    lv = (1 << mid_bits) - 1
    R = np.maximum(U, 0).max(axis=0)
    R = np.where(R > 1e-12, R, 1.0)
    rr = np.maximum(U, 0)
    v = np.clip(np.round(lv * rr / R[None, :]), 0, lv).astype(np.float64)
    if mid_bits > 8:
        raise ValueError
    if mid_bits == 4:
        Z = v
        nch = 1
    else:
        vi = v.astype(np.int64)
        Z = np.concatenate([(vi >> 4), (vi & 0xF)], axis=1).astype(np.float64)
        nch = 2
    # 输出层：逐通道把 R_j/lv 吸收进权重
    # 真实 logit = sum_j W2_kj * rr_j + b2_k = sum_j W2_kj*(R_j/lv)*v_j + b2_k
    # 对拆分：v_j = 16*hi+lo -> 权重 16*(R_j/lv)*W2_kj 与 1*(R_j/lv)*W2_kj
    res = {}
    for S_ in (100, 250, 500, 1000, 2000, 4000):
        Wt = (R / lv)[None, :] * W2.T          # (3, h) 每个输出对每通道的实数权重
        if nch == 2:
            A = np.concatenate([np.round(S_ * 16 * Wt), np.round(S_ * Wt)], axis=1)
        else:
            A = np.round(S_ * Wt)
        th = -np.round(S_ * (b2 - 0))
        logit = Z @ A.T + th[None, :]
        err = int(np.sum((logit >= 0).astype(int) != (y3[:, None] == np.arange(3)[None, :]).astype(int)))
        wm = int(np.abs(A).max())
        res[S_] = (err, wm, A, th)
    return res, Z, nch, R, lv

print("\n" + "=" * 100)
print("编译结果（在全部 65536 状态上评估，目标 = teacher 的输出）")
print("=" * 100)
# 全空间 teacher 输出
Utrue_all = (np.indices((16,) * 4, dtype=np.float64).reshape(4, -1).T - np.array([4.3, 2.0, 1.0, 0.1])) \
            / np.array([3.6, 2.4, 5.9, 2.4])
# 用 teacher 在全空间的 4bit 量化输入上定义真值
Yt_all = np.zeros((len(X6536), 3))
Zt_all = np.maximum(U, 0) @ W2 + b2
Yt_all = (Zt_all >= 0).astype(int)
print(f"  teacher 在 65536 状态上的三路输出统计: "
      f"{[int(Yt_all[:, k].sum()) for k in range(3)]}")

out = {"teacher": d, "results": {}}
for bits in (4, 8):
    res, Z, nch, R, lv = run(bits)
    print(f"\n--- 中间层 {bits} bit {'(1 条线)' if bits==4 else '(2 条线, 拆分)'} ---")
    print(f"  R_j = {np.round(R,4).tolist()}   lv = {lv}")
    print(f"  {'S':>6} | {'权重max':>8} | {'容器':>6} | {'150点错误':>9} | {'全空间错误':>10} | {'错误率':>9}")
    print("  " + "-" * 68)
    for S_, (err150, wm, A, th) in res.items():
        if nch == 2:
            Rj = np.concatenate([R, R])
        else:
            Rj = R
        rr = np.maximum(U, 0)
        v = np.clip(np.round(lv * rr / Rj[None, :]), 0, lv)
        if nch == 2:
            vi = v.astype(np.int64)
            ZZ = np.concatenate([(vi >> 4), (vi & 0xF)], axis=1).astype(np.float64)
        else:
            ZZ = v.astype(np.float64)
        logit_all = ZZ @ A.T + th[None, :]
        pred_all = (logit_all >= 0).astype(int)
        err_all = int(np.sum(pred_all != Yt_all))
        ok = "OK" if wm <= 897 else "越界%d" % wm
        print(f"  {S_:6d} | {wm:8d} | {ok:>6} | {err150:9d} | {err_all:10d} | "
              f"{err_all/len(X6536)*100:8.4f}%")
        out["results"][f"mid{bits}bit_S{S_}"] = {
            "err_150pts_binary": int(err150), "err_65536_binary": int(err_all),
            "wmax": int(wm), "R": R.tolist()}

json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "results", "iris_compile.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/iris_compile.json")
