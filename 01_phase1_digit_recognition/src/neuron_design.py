#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：强模 LUT 网络能多接近同结构的 MLP
========================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 16^4 = 65536 个状态穷举，无训练/测试划分
  目标标签   N4_MLP_T1 的官方标签（4->8->1 tanh MLP，seed 71000，正类 18436），
             取自 DT_baseline_recovery/teachers/N4_MLP_T1.npy，SHA-256 已校验

  参照组 A   原 Teacher：4->8->1 实数权重 tanh，官方前向
  参照组 B   同结构实数 MLP：4->8->1 tanh，在同一批 65536 条标签上从零训练
             （numpy 手写反传 + Adam，全批梯度，lr=0.05，4000 步，3 个种子取最好）
  实验组 C   强模 LUT：同形状 4->8->1
             * 每坐标一张 LUT；阶梯取值非负、相邻差为整数、|差| ≤ 868
             * 正权重：上升阶梯  s[x] = round(x * s_j * w)
             * 负权重：下降阶梯  s[x] = round((15-x) * s_j * |w|) + span，span = round(15*s_j*|w|)
               （整体抬高使容器非负，差仍是 −round(s_j*|w|)）
             * s_j 逐神经元取：由该神经元自己的 max_i|w_ji| 决定
             * 容器 b_t = 15 + (s[t]-s[t-1])，b_0 = 0；容器上限 897
             * 激活 = 精确 tanh 表（存模拟量）；输出层 8->1 用整数权重 + 整数偏置

  评价       全部 65536 状态上的最终 0/1 错误数（hard MAE 口径）
"""
import numpy as np, json, os

CAP_STEP, CONT_HI = 868, 897

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def train_mlp(X, y, h=8, seed=0, steps=4000, lr=0.05):
    rng = np.random.default_rng(seed)
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    W1 = rng.normal(0, 0.5, (h, 4)); b1 = np.zeros(h)
    W2 = rng.normal(0, 0.5, h); b2 = 0.0
    params = [W1, b1, W2, np.array(b2)]
    ms = [np.zeros_like(p) for p in params]; vs = [np.zeros_like(p) for p in params]
    N = len(y)
    for t in range(1, steps + 1):
        H = np.tanh(Xn @ W1.T + b1); z = H @ W2 + b2
        p = 1 / (1 + np.exp(-z)); dz = (p - y) / N
        gW2 = H.T @ dz; gb2 = dz.sum()
        dH = np.outer(dz, W2) * (1 - H ** 2)
        gW1 = dH.T @ Xn; gb1 = dH.sum(axis=0)
        for i, (pp, g) in enumerate(zip(params, [gW1, gb1, gW2, np.array(gb2)])):
            ms[i] = 0.9 * ms[i] + 0.1 * g; vs[i] = 0.999 * vs[i] + 0.001 * g ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
    err = int(np.sum(((np.tanh(Xn @ W1.T + b1) @ W2 + b2) >= 0).astype(np.int8) != y))
    return {"err": err, "seed": seed}

# ---------------------------------------------------------------- 实验组 C
def build_lut(w, s):
    """
    为一条坐标造 LUT。直接实现阶梯 lambda_i(x)，只需要它的相邻差为整数且足够小。

    实测规律 (b_1..b_15 = B 常数时, 每步 p):
        反式: p = B - 15            顺式: p = 15 - B
    本设计取 上升阶梯  lambda[x] = round(x * s * |w|), 步长 d = round(s*|w|) >= 0:
        正权重: lambda = +up  -> 反式, B = 15 + d,      容器 = 15 + d      (常数)
        负权重: lambda = -up  -> 顺式, B = 15 - (-d) = 15 + d, 容器 = 15 + d (常数)
    两种情形容器都是同一个常数 15 + d, 且 d = round(s*|w|) <= 868, 容器 <= 883 ✓
    b_0 = 0; 顺式取负号后 lambda 变号, 于是"上升的 up"变成"实际下降的 lambda"。
    """
    w = float(w); aw = abs(w)
    x = np.arange(16)
    up = np.round(x * s * aw).astype(np.int64)      # 上升阶梯
    d = int(up[1])                                   # 常数步长
    b = np.full(16, 15 + d, dtype=np.int64); b[0] = 0
    if d > CAP_STEP or b.max() > CONT_HI or b.min() < 0:
        return None
    sign = 1 if w >= 0 else -1
    return {"lam": sign * up, "up": up, "d": d, "container": b, "s": s,
            "sign": sign, "form": "反式" if sign > 0 else "顺式",
            "max_step": d, "cont_max": int(b.max())}

def max_s_for(W_ji, cap=CAP_STEP):
    """斜率 = W_ji/7.5；最大步长 = round(s*|W_ji|/7.5) <= cap  =>  s <= cap*7.5/|W_ji|"""
    aw = abs(float(W_ji))
    return 10 ** 9 if aw == 0 else int(cap * 7.5 / aw)

def build_design(W1):
    h, n = W1.shape
    tables, scales = [], []
    for j in range(h):
        s = min(max_s_for(W1[j, i]) for i in range(n))
        row = [build_lut(W1[j, i] / 7.5, s) for i in range(n)]
        if any(t is None for t in row):
            return None
        tables.append(row); scales.append(s)
    return {"tables": tables, "scales": scales}

def forward_strong(design, X, W1, b1, W2, b2, out_q=None):
    """
    强模 LUT 前向。
      每条坐标直接查表: lambda_i(x) = lam_i[x]  (正权重为上升阶梯, 负权重为其相反数)
      t_j = sum_i lambda_i(x_i)  =  s_j * sum_i (W_ji/7.5) x_i      (舍入误差 <= 0.5/s 每项)
      u_j = t_j / s_j  -  sum_i W_ji  +  b_j
      h_j = tanh(u_j)                              (激活表, 存模拟量, 不二值化)
      y   = 1[ sum_j W2_j h_j + b2 >= 0 ]
    """
    Xi = X.astype(np.int64); N = len(Xi)
    h = len(design["tables"])
    H = np.zeros((N, h))
    for j in range(h):
        tab = design["tables"][j]
        s = design["scales"][j]
        t = np.zeros(N, dtype=np.int64)
        for i, tb in enumerate(tab):
            t += tb["lam"][Xi[:, i]]            # lambda_i(x_i) 直接查表
        u = t / s - W1[j].sum() + b1[j]
        H[:, j] = np.tanh(u)
    if out_q is None:
        logit = H @ W2 + b2
    else:
        logit = H @ (np.round(W2 * out_q) / out_q) + np.round(b2 * out_q) / out_q
    return (logit >= 0).astype(np.int8), H

# ---------------------------------------------------------------- 主流程
def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1 = np.array(m["W1"]); b1 = np.array(m["b1"])
    W2 = np.array(m["W2"]); b2 = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)

    print("=" * 104)
    print("实验：强模 LUT 网络能多接近同结构的 MLP")
    print("=" * 104)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 个状态穷举，无训练/测试划分")
    print(f"目标标签 : N4_MLP_T1（4->8->1 tanh MLP，seed 71000，正类 {int(y.sum())}）")
    print(f"评价口径 : 全部 {N} 状态上的最终 0/1 错误数（hard MAE）")

    predA = ((np.tanh(Xn @ W1.T + b1) @ W2 + b2) >= 0).astype(np.int8)
    errA = int(np.sum(predA != lab))
    print(f"\n[参照组 A] 原 Teacher：4->8->1 实数权重 tanh，官方前向（|W1|max={np.abs(W1).max():.4f}）")
    print(f"   最终错误 = {errA}/{N}")

    print(f"\n[参照组 B] 同结构实数 MLP 4->8->1 tanh，在同一批标签上从零训练")
    print(f"   numpy 手写反传 + Adam，全批梯度，lr=0.05，4000 步，3 个种子取最好")
    bestB = None
    for seed in (0, 1, 2):
        r = train_mlp(X, y, seed=seed)
        print(f"   seed={seed}: 最终错误 = {r['err']}/{N}")
        if bestB is None or r["err"] < bestB["err"]:
            bestB = r
    print(f"   参照组 B 最好 = {bestB['err']}/{N}")

    print(f"\n[实验组 C] 强模 LUT，同形状 4->8->1")
    print(f"   每坐标一张 LUT；逐神经元自己的 s；阶梯非负、方向随权重符号；b_0=0")
    print(f"   容器上限 {CONT_HI}、步长上限 {CAP_STEP}；激活 = 精确 tanh 表（存模拟量）")
    design = build_design(W1)
    if design is None:
        print("   构造失败"); return
    print(f"\n   {'j':>2} | {'max|W|':>8} | {'s_j':>6} | {'步长 d':>8} | {'坐标形式':>18} | {'容器(常数)':>10}")
    for j in range(W1.shape[0]):
        tb = design["tables"][j]
        print(f"   {j:2d} | {np.abs(W1[j]).max():8.4f} | {design['scales'][j]:6d} | "
              f"{max(t['max_step'] for t in tb):8d} | "
              f"{' '.join(t['form'] for t in tb):>18} | {max(t['cont_max'] for t in tb):10d}")

    predC, Hc = forward_strong(design, X, W1, b1, W2, b2)
    errC = int(np.sum(predC != lab))
    dh = float(np.max(np.abs(Hc - np.tanh(Xn @ W1.T + b1))))
    print(f"\n   隐藏层（模拟量）与实数 tanh 的最大偏差 = {dh:.3e}")
    print(f"   最终错误 = {errC}/{N}")
    for q in (1024, 65536):
        pq, _ = forward_strong(design, X, W1, b1, W2, b2, out_q=q)
        print(f"   输出层整数化 (q={q})：最终错误 = {int(np.sum(pq != lab))}/{N}")

    print("\n" + "=" * 104)
    print("汇总（同一 65536 状态、同一标签、同一评价口径）")
    print("=" * 104)
    print(f"   {'方案':>30} | {'形状':>8} | {'最终错误':>9} | {'错误率':>10}")
    print(f"   {'A 原 Teacher（实数权重）':>30} | {'4->8->1':>8} | {errA:9d} | {errA/N*100:9.4f}%")
    print(f"   {'B 同结构实数 MLP（重训）':>30} | {'4->8->1':>8} | {bestB['err']:9d} | {bestB['err']/N*100:9.4f}%")
    print(f"   {'C 强模 LUT':>30} | {'4->8->1':>8} | {errC:9d} | {errC/N*100:9.4f}%")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump({"A_err": errA, "B_err": bestB["err"], "C_err": errC,
               "scales": design["scales"], "hidden_tanh_maxdev": dh},
              open(os.path.join(res, "strong_vs_mlp.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/strong_vs_mlp.json")

if __name__ == "__main__":
    main()
