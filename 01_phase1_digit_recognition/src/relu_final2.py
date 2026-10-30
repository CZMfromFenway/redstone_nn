#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReLU 路线（顺式/反式正确版 + 独立评价）
=========================================
【lambda 的正确构造 (czm 确认)】
  W1_ji > 0 : 上升阶梯  lambda = round(x*rho)          <-> 反式, x=0 处输出 0
  W1_ji < 0 : 下降阶梯  lambda = round(15*rho - x*rho) <-> 顺式, x=15 处输出 0
  两者都 >= 0, 且 sum_i lambda_i(x_i) 与预激活 u 严格仿射。

【评价口径 (修正循环论证)】
  固定第一层 LUT, **独立重训第二层**(只用整数编码 Z), 用官方标签评价。
   实数重训: 允许实数权重
   整数化  : 权重取整后复评

【对照变量】
  第一层 LUT 精细度 M (满档值), 浮动/均一步长
  第二层位数 4/8/12/16
"""
import numpy as np, json, os

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def train_relu(Xn, y, h=8, seed=0, steps=8000, lr=0.02, bs=8192, log_every=4000):
    rng = np.random.default_rng(seed); N = len(Xn)
    W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
    W2 = rng.normal(0, 0.6, h); b2 = 0.0
    ps = [W1, b1, W2, np.array(b2)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]
        U = xb @ W1.T + b1; H = np.maximum(U, 0); z = H @ W2 + b2
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); dz = (p - yb) / bs
        dU = np.outer(dz, W2) * (U > 0)
        g = [dU.T @ xb, dU.sum(0), H.T @ dz, np.array(dz.sum())]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % log_every == 0 or t == steps:
            Hf = np.maximum(Xn @ W1.T + b1, 0)
            e = int(np.sum(((Hf @ W2 + b2) >= 0).astype(np.int64) != y))
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), W2.copy(), float(b2))
            print(f"      [relu s{seed}] step {t}: 错误={e} 最好={best[0]}", flush=True)
    return best

def build_Lambda(X, W1, M, floating=True):
    """
    按 czm 规则构造第一层 LUT 之和 Lambda_j = sum_i lambda_{j,i}(x_i)
      W>0: 上升 round(x*rho)            (反式)
      W<0: 下降 round(15*rho - x*rho)   (顺式), 仍 >= 0
      rho = M*|W1_ji|/max_i|W1_ji|  (floating)  或  rho = M/15 均一 (not floating)
    返回 Lambda (N,h)  以及 与 u 的仿射系数
    """
    W1 = np.asarray(W1, float); h = W1.shape[0]
    x = np.arange(16); Xi = X.astype(np.int64); N = len(X)
    Lam = np.zeros((N, h), dtype=np.int64)
    for j in range(h):
        mx = np.abs(W1[j]).max()
        if mx <= 1e-12:
            continue
        for i in range(4):
            w = W1[j, i]
            if abs(w) <= 1e-12:
                continue
            rho = (M * abs(w) / mx) if floating else (M / 15.0)
            up = np.round(x * rho)
            if w > 0:
                lam = up
            else:
                lam = np.round(15 * rho - up)
            Lam[:, j] += lam.astype(np.int64)[Xi[:, i]]
    return Lam

def affine_check(Lam, U):
    """检验 Lambda 与 u 是否仿射: 逐神经元线性拟合残差"""
    N = len(Lam)
    A = np.hstack([Lam, np.ones((N, 1))])
    coef, *_ = np.linalg.lstsq(A, U, rcond=None)
    return float(np.max(np.abs(A @ coef - U)))

def encode(Lam, bits):
    """各神经元把 Lambda 的实际范围映射到 0~lv"""
    N, h = Lam.shape; lv = (1 << bits) - 1
    lo = Lam.min(axis=0); hi = Lam.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    v = np.clip(np.round(lv * (Lam - lo[None, :]) / span[None, :]), 0, lv).astype(np.int64)
    if bits == 4:
        return v.astype(float), 1
    nch = bits // 4
    Z = np.concatenate([((v >> (4 * k)) & 0xF) for k in range(nch)], axis=1).astype(float)
    return Z, nch

def train_readout(Z, y, steps=6000, lr=0.05, seed=0):
    """
    独立重训第二层。注意: Z 的量级可能很大(Lambda 可达 1e7),
    先做标准化再用 Adam 训练, 最后把权重还原到原始尺度。
    """
    rng = np.random.default_rng(seed); N = len(Z)
    s = Z.std(axis=0)
    s = np.where(s > 1e-12, s, 1.0)
    mz = Z.mean(axis=0)
    Zn = (Z - mz) / s                      # 标准化后的编码
    w = rng.normal(0, 0.3, Zn.shape[1]); b = 0.0
    ms = np.zeros_like(w); vs = np.zeros_like(w); mb = vb = 0.0
    ys = 2 * y - 1
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, 8192, replace=False)
        Zb = Zn[idx]; yb = y[idx]; yy = ys[idx]
        z = Zb @ w + b
        if t <= steps * 3 // 4:
            m = 1 - yy * z; dz = (-yy * (m > 0)) / 8192
        else:
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); dz = (p - yb) / 8192
        gw = Zb.T @ dz; gb = dz.sum()
        ms = 0.9 * ms + 0.1 * gw; vs = 0.999 * vs + 0.001 * gw ** 2
        w -= lr * (ms / (1 - 0.9 ** t)) / (np.sqrt(vs / (1 - 0.999 ** t)) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
        if t % 2000 == 0 or t == steps:
            e = int(np.sum(((Zn @ w + b) >= 0).astype(np.int64) != y))
            if best is None or e < best[0]:
                best = (e, w.copy(), b)
    e_real, w, b = best[0], best[1], best[2]
    # 还原到原始尺度: z = Zn@w + b = (Z-mz)/s @ w + b = Z@(w/s) + (b - mz@(w/s))
    w_raw = w / s
    b_raw = b - mz @ w_raw
    # 整数化: 用最大权重定尺度
    sc = 1000.0 / max(np.abs(w_raw).max(), 1e-12)
    wq = np.round(w_raw * sc); bq = b_raw * sc
    e_int = int(np.sum(((Z @ wq + bq) >= 0).astype(np.int64) != y))
    return e_real, e_int, float(np.abs(w_raw).max()), int(np.abs(wq).max())

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    print("=" * 100, flush=True)
    print("ReLU 路线（顺式/反式正确版 + 独立重训第二层 + 官方标签评价）", flush=True)
    print("=" * 100, flush=True)

    best = None
    for s in (0, 1, 2, 3, 4):
        r = train_relu(Xn, y, 8, seed=s, steps=8000)
        if best is None or r[0] < best[0]:
            best = r
    _, W1r, b1r, W2r, b2r = best
    W1r = np.asarray(W1r, float); b1r = np.asarray(b1r, float).reshape(-1)
    U = Xn @ W1r.T + b1r
    print(f"\nReLU 老师拟合官方标签 = {best[0]}/{N} ({best[0]/N*100:.4f}%)", flush=True)
    print(f"真值 u 范围 = [{U.min():.3f},{U.max():.3f}]  死区(负值)占比 = {(U<=0).mean()*100:.1f}%",
          flush=True)

    print(f"\n[仿射性检验] 下降阶梯(顺式)是否让 Lambda 与 u 严格仿射", flush=True)
    for M in (63, 255, 868):
        Lam = build_Lambda(X, W1r, M, True)
        dev = affine_check(Lam.astype(float), U)
        print(f"   M={M:4d}: max|Lambda->u 线性拟合残差| = {dev:.3e}", flush=True)
    print(f"   (若为 ~1e-3 量级即为严格仿射; 之前用全上升阶梯时是 2.7~10.7)", flush=True)

    print(f"\n[第二层位数扫描] 第一层 M=868 浮动", flush=True)
    print(f"   {'位数':>5} | {'弱模线':>6} | {'实数重训':>9} | {'整数化':>8} | {'错误率':>9} | {'权重max':>8}",
          flush=True)
    print("-" * 74, flush=True)
    out = {"relu_teacher": int(best[0]), "N": N}
    for bits in (4, 8, 12, 16):
        Lam = build_Lambda(X, W1r, 868, True)
        Z, nch = encode(Lam, bits)
        e_real, e_int, wm, wmq = train_readout(Z, y)
        print(f"   {bits:5d} | {nch*8:6d} | {e_real:9d} | {e_int:8d} | {e_int/N*100:8.4f}% | "
              f"{wmq:8d}", flush=True)
        out[f"{bits}bit"] = {"real": int(e_real), "int": int(e_int),
                             "channels": int(nch * 8), "wmax": int(wmq)}
    print(f"\n   对照 tanh: 4bit ≈1005 (1.534%), 8bit拆分 75 (0.1144%)", flush=True)

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump(out, open(os.path.join(resdir, "relu_final2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/relu_final2.json", flush=True)

if __name__ == "__main__":
    main()
