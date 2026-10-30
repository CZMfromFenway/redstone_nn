#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
编译对照（干净版）：tanh 拆分 vs ReLU 线性转弱模
==================================================
第一层: 强模 LUT。坐标 i 的步长 rho_{j,i}=868*|W1_ji|/max_i|W1_ji| (若 max=0 则该坐标忽略)
        lambda_{j,i}(x)=round(x*rho_{j,i}) 幅值阶梯；负权重用下降阶梯 15*rho-lambda 保持非负
第二层:
   tanh: 非线性拆分, v=round((tanh+1)/2*255), 拆高/低半字节 -> 16 条弱模线
   ReLU: 线性缩放,   逐神经元 z_j=round(relu_j * 15/max)  -> 8 条弱模线
读出: 最小二乘后按尺度 S 取整（整数权重）
"""
import numpy as np, json, os

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def lut_lambda(W1):
    """返回 lam_up[j,i,x] 与 rho[j,i]"""
    h = W1.shape[0]
    rho = np.zeros((h, 4))
    for j in range(h):
        m = np.abs(W1[j]).max()
        if m > 1e-12:
            rho[j] = 868.0 * np.abs(W1[j]) / m
    x = np.arange(16)
    up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
    dn = np.round(15 * rho[:, :, None] - up.astype(float)).astype(np.int64)
    return up, dn, rho

def layer1_forward(X, W1, b1):
    """
    第一层正向。关键: lambda 必须非负, 所以每个坐标都用**上升阶梯** up,
    符号由输出层承担(即 S += sign(W)*up), 而不是把 lambda 本身取负。
    (下降阶梯 15rho-round(x rho) 给出的是"常数-贡献", 直接相加会错。)
    """
    up, dn, rho = lut_lambda(W1)
    h = W1.shape[0]; N = len(X); Xi = X.astype(np.int64)
    S = np.zeros((N, h), dtype=np.int64)
    for j in range(h):
        m = np.abs(W1[j]).max()
        if m <= 1e-12:
            continue
        for i in range(4):
            if abs(W1[j, i]) <= 1e-12:
                continue                      # 该坐标权重为 0, 不需要 LUT
            S[:, j] += (1 if W1[j, i] > 0 else -1) * up[j, i, Xi[:, i]]
    sc = np.array([(np.abs(W1[j]).max() / (868 * 7.5)) if np.abs(W1[j]).max() > 1e-12 else 0.0
                   for j in range(h)])
    U1 = S * sc[None, :] - W1.sum(axis=1)[None, :] + b1[None, :]
    return U1

def fit_int(Z, target, S):
    A = np.hstack([Z, np.ones((len(Z), 1))]).astype(float)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return np.round(coef[:-1] * S).astype(np.int64), -round(coef[-1] * S)

def evaluate(name, Z, target, lab, S):
    w, th = fit_int(Z, target, S)
    e = int(np.sum(((Z @ w - th) >= 0).astype(np.int8) != lab))
    print(f"   {name:>40} | {Z.shape[1]:6d} | {np.abs(w).max():8d} | {e:7d} | {e/len(lab)*100:8.4f}%",
          flush=True)
    return {"case": name, "S": S, "dim": int(Z.shape[1]), "wmax": int(np.abs(w).max()),
            "err": e}

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1o = np.array(m["W1"]); b1o = np.array(m["b1"])
    W2o = np.array(m["W2"]); b2o = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errA = int(np.sum(((np.tanh(Xn @ W1o.T + b1o) @ W2o + b2o) >= 0).astype(np.int8) != lab))

    print("=" * 104, flush=True)
    print("编译对照：tanh 非线性拆分 vs ReLU 线性转弱模", flush=True)
    print("=" * 104, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)
    print(f"参照     : 官方 tanh 老师（实数）= {errA}/{N}", flush=True)

    # ---- tanh 官方老师 ----
    U1 = layer1_forward(X, W1o, b1o)
    Hc = np.tanh(U1)
    dev = np.max(np.abs(U1 - (Xn @ W1o.T + b1o)))
    print(f"\n第一层 LUT 后预激活与真值最大偏差 = {dev:.3e}", flush=True)
    v8 = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)
    Ztanh = np.concatenate([(v8 >> 4), (v8 & 0xF)], axis=1).astype(float)
    tgt_tanh = Hc @ W2o + b2o
    e_first = int(np.sum(((Hc @ W2o + b2o) >= 0).astype(np.int8) != lab))
    print(f"仅第一层 LUT（第二层实数）错误 = {e_first}/{N}", flush=True)

    # ---- ReLU 老师：从结构训练得到（用官方真值表训练, 让它尽量拟合） ----
    rng = np.random.default_rng(0)
    W1r = rng.normal(0, 0.6, (8, 4)); b1r = np.zeros(8)
    W2r = rng.normal(0, 0.6, 8); b2r = 0.0
    ps = [W1r, b1r, W2r, np.array(b2r)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = None
    for t in range(1, 4001):
        idx = rng.choice(N, 8192, replace=False)
        xb = Xn[idx]; yb = y[idx]
        U = xb @ W1r.T + b1r; H = np.maximum(U, 0); z = H @ W2r + b2r
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); dz = (p - yb) / 8192
        dU = np.outer(dz, W2r) * (U > 0)
        g = [dU.T @ xb, dU.sum(0), H.T @ dz, np.array(dz.sum())]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= 0.02 * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0:
            Hf = np.maximum(Xn @ W1r.T + b1r, 0)
            e = int(np.sum(((Hf @ W2r + b2r) >= 0).astype(np.int8) != y))
            if best is None or e < best[0]:
                best = (e, W1r.copy(), b1r.copy(), W2r.copy(), float(b2r))
            print(f"      [relu 训练] step {t}: 错误={e} 最好={best[0]}", flush=True)
    _, W1r, b1r, W2r, b2r = best

    Ur = layer1_forward(X, W1r, b1r)
    Hr = np.maximum(Ur, 0)
    colmax = Hr.max(axis=0); colmax = np.where(colmax > 1e-12, colmax, 1.0)
    Zrelu = np.clip(np.round(Hr * 15 / colmax[None, :]), 0, 15).astype(float)
    tgt_relu = Hr @ W2r + b2r
    print(f"ReLU 老师（连续隐藏层, 拟合官方标签）错误 = {best[0]}/{N} "
          f"({best[0]/N*100:.4f}%)", flush=True)
    print(f"仅第一层 LUT 后（ReLU, 第二层实数）错误 = "
          f"{int(np.sum(((Hr @ W2r + b2r) >= 0).astype(np.int8) != y))}", flush=True)

    print(f"\n   {'方案':>40} | {'弱模通道':>6} | {'权重max':>8} | {'错误':>7} | {'错误率':>9}", flush=True)
    print("-" * 104, flush=True)
    rows = []
    for S in (2000, 8000, 32000, 128000):
        rows.append(evaluate(f"tanh 拆分（16 线） S={S}", Ztanh, tgt_tanh, lab, S))
    print("-" * 104, flush=True)
    for S in (2000, 8000, 32000, 128000):
        rows.append(evaluate(f"ReLU 线性（8 线） S={S}", Zrelu, tgt_relu, lab, S))

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump({"teacher": errA, "N": N, "relu_trained_err": int(best[0]),
               "layer1_only_tanh": e_first, "rows": rows},
              open(os.path.join(resdir, "compile_tanh_vs_relu.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/compile_tanh_vs_relu.json", flush=True)

if __name__ == "__main__":
    main()
