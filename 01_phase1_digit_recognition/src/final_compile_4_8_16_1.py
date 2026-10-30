#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
定稿实验：从 Teacher 精确编译到 4->8->16->1 的离散强模 LUT
==========================================================
【架构】  4 -> 8(强模LUT) -> 16(拆分弱模) -> 1(高精度整数阈值)
【输入空间】(x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
【目标标签】N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
【评价】   全部 65536 状态上的最终 0/1 错误数

【第一层 4->8】 每神经元每坐标一张 LUT
    lambda_{j,i}(x) = round(x * rho_{j,i}),   rho_{j,i} = 868*|W1_ji| / max_i|W1_ji|
    (相邻差为整数且浮动; 负权重用下降阶梯保证 lambda>=0)

【第二层 8->16->1】 编码 + 拆分 + 整数权重
    v_j = round((tanh_j+1)/2*255),  v_j = 16*hi_j + lo_j
    A_hi_j = s2 * 32*W2_j/255,  A_lo_j = s2 * 2*W2_j/255 = A_hi_j/16
    物理约束: 15*|A_hi| <= 13455 且 15*|A_lo| <= 13455  (即 |A| <= 897)
"""
import numpy as np, json, os

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

# ---------------- 第一层: 逐神经元逐坐标的阶梯 ----------------
def build_layer1(W1):
    """对每个神经元 j, 每个坐标 i 求 rho_{j,i} 与 16 点阶梯"""
    rho = np.zeros_like(W1, dtype=np.float64)
    for j in range(W1.shape[0]):
        mx = np.abs(W1[j]).max()
        rho[j] = 868.0 * np.abs(W1[j]) / mx
    x = np.arange(16)
    # lam[j,i,x] = round(x*rho[j,i])  (幅值上升阶梯, 非负)
    lam = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
    sign = np.where(W1 >= 0, 1, -1)
    return rho, lam, sign

# ---------------- 第二层: 整数权重 ----------------
def build_layer2(W2, s2):
    A_hi = np.round(s2 * 32 * W2 / 255).astype(np.int64)
    A_lo = np.round(s2 * 2 * W2 / 255).astype(np.int64)
    return A_hi, A_lo

def forward(X, W1, b1, W2, b2, s2, lam, sign):
    """
    整数前向, 完全对应硬件:
      第1层: t_j = sum_i sign_ji * lam[j,i,x_i]     (整数, 非负阶梯)
             注意: 负权重用下降阶梯等价于 15*rho - lam 再取负号
      为保持 lambda>=0, 负坐标用下降阶梯: lam_desc = 15*rho - lam_up
      两者对 sum 的贡献差一个常数, 由 f (tanh 的查表) 吸收
      这里为了数值验证, 用"上升阶梯 + 输出层承担负号"的等价形式:
             u_j = ( sum_i sign_ji*lam[j,i,x_i] - C_j ) / s1_j + b1_j
      其中 s1_j 由 rho 的尺度决定。
    """
    N = len(X); Xi = X.astype(np.int64)
    h = W1.shape[0]
    # 第一层的等效: lam[j,i,x] = round(x*rho[j,i]) ≈ rho[j,i]*x
    # 且 rho[j,i] = 868*|W1_ji|/mx_j.  而真实贡献需要 W1_ji/7.5 * x
    # 故 lam 与真实 tanh 预激活的关系: W1_ji/7.5*x = lam * (mx_j/(868*7.5))
    scale1 = np.array([np.abs(W1[j]).max() / (868 * 7.5) for j in range(h)])   # (h,)
    U = np.zeros((N, h))
    for j in range(h):
        s = np.zeros(N, dtype=np.int64)
        for i in range(4):
            s += sign[j, i] * lam[j, i, Xi[:, i]]
        U[:, j] = s * scale1[j]                      # 近似 W1_j.x/7.5
    U = U - W1.sum(axis=1)[None, :] + b1[None, :]     # 补上 -sum W1 + b1
    Hc = np.tanh(U)
    return U, Hc

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1 = np.array(m["W1"]); b1 = np.array(m["b1"])
    W2 = np.array(m["W2"]); b2 = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    Hc_true = np.tanh(Xn @ W1.T + b1)
    errA = int(np.sum(((Hc_true @ W2 + b2) >= 0).astype(np.int8) != lab))

    print("=" * 100, flush=True)
    print("定稿：Teacher -> 4->8->16->1 离散强模 LUT 的精确编译", flush=True)
    print("=" * 100, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)
    print(f"参照 A 原 Teacher = {errA}/{N}", flush=True)

    rho, lam, sign = build_layer1(W1)
    print(f"\n[第一层] 逐神经元逐坐标的步长 rho (每神经元用自己 max|W1| 定尺度):", flush=True)
    for j in range(W1.shape[0]):
        print(f"   j={j}: max|W1|={np.abs(W1[j]).max():.4f}  rho={np.round(rho[j],2).tolist()}  "
              f"满档={np.round(15*rho[j],0).tolist()}", flush=True)
    print(f"   (满分档应 ≤13455; 最大者恰为 13020 = 15*868)", flush=True)

    print(f"\n[第二层] 权重随 s2 的变化，以及最终错误率:", flush=True)
    print(f"   {'s2':>9} | {'A_hi_max':>8} | {'A_lo_max':>8} | {'hi满档':>7} | {'lo满档':>7} | "
          f"{'越界':>4} | {'错误':>7} | {'错误率':>9}", flush=True)
    rows = []
    U, Hc = forward(X, W1, b1, W2, b2, 1.0, lam, sign)
    print(f"   第一层近似后 tanh 与真值最大偏差 = {np.max(np.abs(Hc - Hc_true)):.3e}", flush=True)
    for s2 in (1000, 10000, 50000, 100000, 140000, 150000):
        A_hi, A_lo = build_layer2(W2, s2)
        ahm, alm = np.abs(A_hi).max(), np.abs(A_lo).max()
        over = "是" if (15 * ahm > 13455 or 15 * alm > 13455) else "否"
        v = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)
        Z = np.concatenate([((v >> 4) & 0xF), (v & 0xF)], axis=1).astype(np.int64)
        logit = Z @ np.concatenate([A_hi, A_lo]) - s2 * (b2 - W2.sum())
        e = int(np.sum(((logit >= 0).astype(np.int8) != lab)))
        print(f"   {s2:9d} | {ahm:8.0f} | {alm:8.0f} | {15*ahm:7.0f} | {15*alm:7.0f} | "
              f"{over:>4} | {e:7d} | {e/N*100:8.4f}%", flush=True)
        rows.append({"s2": s2, "A_hi_max": float(ahm), "A_lo_max": float(alm),
                     "over": over == "是", "err": e})

    print(f"\n[诊断] 逐环节的误差来源", flush=True)
    # 1) 第一层 LUT 的量化误差
    dU = np.max(np.abs(U - (Xn @ W1.T + b1)))
    print(f"   第一层 LUT 造成的预激活最大偏差 = {dU:.3e}", flush=True)
    e1 = int(np.sum(((np.tanh(U) @ W2 + b2) >= 0).astype(np.int8) != lab))
    print(f"   仅第一层量化(第二层用实数)时的错误 = {e1}/{N}", flush=True)
    # 2) 8 位编码的误差
    v = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)
    Hq = 2 * v / 255 - 1
    e2 = int(np.sum(((Hq @ W2 + b2) >= 0).astype(np.int8) != lab))
    print(f"   再加 8 位编码(第二层仍实数)时的错误 = {e2}/{N}", flush=True)
    # 3) 整数权重
    A_hi, A_lo = build_layer2(W2, 140000)
    Z = np.concatenate([((v >> 4) & 0xF), (v & 0xF)], axis=1).astype(np.float64)
    logit = Z @ np.concatenate([A_hi, A_lo]) - 140000 * (b2 - W2.sum())
    e3 = int(np.sum(((logit >= 0).astype(np.int8) != lab)))
    print(f"   再加整数权重时的错误 = {e3}/{N}  <== 最终", flush=True)

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump({"A": errA, "N": N, "rows": rows, "layer1_maxdev": float(dU),
               "err_layer1_only": int(e1), "err_layer1_8bit": int(e2), "err_final": int(e3)},
              open(os.path.join(resdir, "final_compile_4_8_16_1.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/final_compile_4_8_16_1.json", flush=True)

if __name__ == "__main__":
    main()
