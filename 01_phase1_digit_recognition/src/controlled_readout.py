#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
受控对照：同一第一层 + 三种第二层实现的整数读出
================================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  第一层     固定为"逐神经元逐坐标步长 rho=868*|W1|/max|W1|"的强模 LUT
  第二层     三个受控变体，读出全部为**整数权重**（可建造）
     A  连续 tanh → 整数读出            (不可建造, 只作上界)
     B  4 bit 直接量化 → 整数读出        (可建造, 1 条纵向线/神经元)
     C  8 bit 编码 + 拆分 → 整数读出     (可建造, 2 条纵向线/神经元)
     D  8 bit 编码不拆分 → 整数读出      (不可建造, 仅作诊断: 隔离"拆分"的贡献)
  读出权重   由最小二乘反解后取整; 阈值 = 常数项
"""
import numpy as np, json, os

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def layer1_lut(X, W1):
    """第一层: lam[j,i,x] = round(x*rho[j,i]),  rho=868|W1|/max|W1|"""
    rho = np.zeros_like(W1, dtype=float)
    for j in range(W1.shape[0]):
        rho[j] = 868.0 * np.abs(W1[j]) / np.abs(W1[j]).max()
    x = np.arange(16)
    lam = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
    sgn = np.where(W1 >= 0, 1, -1)
    Xi = X.astype(np.int64)
    h = W1.shape[0]; N = len(X)
    s = np.zeros((N, h), dtype=np.int64)
    for j in range(h):
        for i in range(4):
            s[:, j] += sgn[j, i] * lam[j, i, Xi[:, i]]
    scale1 = np.array([np.abs(W1[j]).max() / (868 * 7.5) for j in range(h)])
    U = s * scale1[None, :]
    return U                                   # ≈ W1.x/7.5

def err_of(Z, w, th, y):
    return int(np.sum(((Z @ w - th) >= 0).astype(np.int64) != y))

def fit_int(Z, target, S):
    """最小二乘拟合 target, 再按尺度 S 取整; 返回 (w_int, th_int, 残差)"""
    A = np.hstack([Z, np.ones((len(Z), 1))]).astype(float)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    w = np.round(coef[:-1] * S).astype(np.int64)
    th = -round(coef[-1] * S)
    return w, th

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1 = np.array(m["W1"]); b1 = np.array(m["b1"])
    W2 = np.array(m["W2"]); b2 = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)

    U1 = layer1_lut(X, W1)                       # ≈ W1.x/7.5
    pre1 = U1 - W1.sum(axis=1)[None, :] + b1[None, :]
    Hc = np.tanh(pre1)
    z_true = Hc @ W2 + b2
    errA = int(np.sum((z_true >= 0).astype(np.int8) != lab))

    print("=" * 104, flush=True)
    print("受控对照：同一第一层，三种第二层的整数读出", flush=True)
    print("=" * 104, flush=True)
    print(f"输入空间 : 全部 {N} 状态；目标标签 N4_MLP_T1（正类 {int(y.sum())}）", flush=True)
    print(f"参照     : A 原 Teacher（实数第一层）= {errA}/{N}", flush=True)
    print(f"第一层   : 强模 LUT, rho=868*|W1|/max|W1|；tanh 与真值最大偏差 "
          f"{np.max(np.abs(Hc - np.tanh(Xn @ W1.T + b1))):.3e}", flush=True)
    print(f"\n   注: 下表的 z 目标统一取 z_true（第一层 LUT 后的真值代理），"
          f"以便隔离第二层的影响", flush=True)

    print(f"\n{'方案':>34} | {'输入维度':>8} | {'S':>8} | {'整数权重max':>11} | "
          f"{'错误':>7} | {'错误率':>9} | 可建造", flush=True)
    print("-" * 104)
    rows = []
    for S in (1000, 4000, 16000):
        # A 连续
        w, th = fit_int(Hc, z_true, S)
        eA = err_of(Hc, w, th, y)
        # B 4 bit
        q4 = np.clip(np.round((Hc + 1) * 15 / 2), 0, 15).astype(float)
        w4, th4 = fit_int(q4, z_true, S); eB = err_of(q4, w4, th4, y)
        # C 8 bit 拆分
        v8 = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)
        Zc = np.concatenate([(v8 >> 4), (v8 & 0xF)], axis=1).astype(float)
        wC, thC = fit_int(Zc, z_true, S); eC = err_of(Zc, wC, thC, y)
        # D 8 bit 不拆分
        wD, thD = fit_int(v8.astype(float), z_true, S); eD = err_of(v8.astype(float), wD, thD, y)
        for nm, Z, w_, t_, e_, ok in (
                ("A 连续 tanh（上界）", Hc, w, th, eA, "—"),
                ("B 4bit 直接量化", q4, w4, th4, eB, "是"),
                ("C 8bit 拆分（2条线）", Zc, wC, thC, eC, "是"),
                ("D 8bit 不拆分（1条线，诊断）", v8.astype(float), wD, thD, eD, "否")):
            print(f"{nm:>34} | {Z.shape[1]:8d} | {S:8d} | {np.abs(w_).max():11d} | "
                  f"{e_:7d} | {e_/N*100:8.4f}% | {ok}", flush=True)
            rows.append({"S": S, "case": nm, "dim": int(Z.shape[1]),
                         "wmax": int(np.abs(w_).max()), "err": int(e_), "buildable": ok})
        print("-" * 104, flush=True)

    print("\n[最优 S 附近的精细扫描] 只报 C（拆分）与 B（4bit）", flush=True)
    print(f"   {'S':>8} | {'B 4bit 错误':>12} | {'C 拆分错误':>11} | {'C 权重max':>10}", flush=True)
    for S in (2000, 8000, 32000, 128000, 512000):
        q4 = np.clip(np.round((Hc + 1) * 15 / 2), 0, 15).astype(float)
        w4, th4 = fit_int(q4, z_true, S); eB = err_of(q4, w4, th4, y)
        v8 = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)
        Zc = np.concatenate([(v8 >> 4), (v8 & 0xF)], axis=1).astype(float)
        wC, thC = fit_int(Zc, z_true, S); eC = err_of(Zc, wC, thC, y)
        print(f"   {S:8d} | {eB:12d} | {eC:11d} | {np.abs(wC).max():10d}", flush=True)

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump({"A_teacher": errA, "N": N, "rows": rows},
              open(os.path.join(resdir, "controlled_readout.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/controlled_readout.json", flush=True)

if __name__ == "__main__":
    main()
