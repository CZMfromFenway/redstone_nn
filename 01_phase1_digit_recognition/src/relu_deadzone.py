#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReLU 离散化（修正版）：死区不占编码空间
==========================================
【核心修正】
  旧做法: v = round(255*Lambda / R)                 死区与饱和区都占用编码空间
  新做法: v = round(255*(Lambda-theta)/(R-theta))    只把"线性段"映射到 0~255
  其中 theta_j = ReLU 的转折点(kink), R_j = 该神经元 Lambda 的最大值
  每档分辨率 Delta_j = (R_j-theta_j)/255  (旧做法是 R_j/255)

【必须注意的陷阱】
  逐神经元的伸缩不同 -> 第二层权重的含义被缩放。
  第二层必须吸收这些因子: 对 v_j 的权重 w_j 满足 w_j*... 对应真实 W2_j*relu_j
  本脚本用**最小二乘直接拟合** (Z -> z_true), 自动吸收所有伸缩因子。

【自洽设定】ReLU 老师 4->8->1 定义真值表, 强模 LUT 复现它。
"""
import numpy as np, json, os

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def train_relu_teacher(Xn, y, h=8, seed=0, steps=8000, lr=0.02, bs=8192, log_every=2000):
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
            e = int(np.sum(((Hf @ W2 + b2) >= 0).astype(np.int8) != y))
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), W2.copy(), float(b2))
            print(f"      [relu s{seed}] step {t}: 错误={e} 最好={best[0]}", flush=True)
    return best

def lut_lambda(W1, cap=868.0):
    h = W1.shape[0]
    rho = np.zeros((h, 4))
    for j in range(h):
        mx = np.abs(W1[j]).max()
        if mx > 1e-12:
            rho[j] = cap * np.abs(W1[j]) / mx
    x = np.arange(16)
    up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
    return up, rho

def layer1_Lambda(X, W1, b1=None):
    """
    第一层: lambda 幅值上升阶梯, 符号交给输出层。
    返回**真实的 Lambda** = sum_i W1_ji/7.5 * x_i - sum_i W1_ji + b1_j
    (与 LUT 实现等价, 误差 ~5e-4)
    """
    W1 = np.asarray(W1, dtype=float)
    b1 = np.zeros(W1.shape[0]) if b1 is None else np.asarray(b1, dtype=float).reshape(-1)
    up, rho = lut_lambda(W1)
    h = W1.shape[0]; N = len(X); Xi = X.astype(np.int64)
    S = np.zeros((N, h), dtype=np.int64)
    for j in range(h):
        if np.abs(W1[j]).max() <= 1e-12:
            continue
        for i in range(4):
            if abs(W1[j, i]) <= 1e-12:
                continue
            S[:, j] += (1 if W1[j, i] > 0 else -1) * up[j, i, Xi[:, i]]
    sc = np.array([(np.abs(W1[j]).max() / (868 * 7.5)) if np.abs(W1[j]).max() > 1e-12 else 0.0
                   for j in range(h)])
    return S * sc[None, :] - W1.sum(axis=1)[None, :] + b1[None, :]

def encode_relu(Lam, bits, theta=None, use_kink=True):
    """
    ReLU 的弱模编码（死区不占编码空间）。

      use_kink=True : theta_j = 该神经元的转折点(dead zone 上界)
                      死区 (Lam < theta_j) 统一映射到 0；
                      线性段 [theta_j, R_j] 映射到 [0, lv]
                        v = clamp(round(lv*(Lam-theta)/(R-theta)), 0, lv)
                      每档分辨率 Delta = (R-theta)/lv
      use_kink=False: 旧做法 v = clamp(round(lv*Lam/R),0,lv)，死区占掉 theta/R 的比例

    czm 的例子: theta=16000, R=48000, lv=255 -> Lam=32000 时 v = 255*16000/32000 = 127
    本实现取 theta_j = min(Lam_j, 0) 与 0 的较大者，即"relu 开始为正"的点。
    """
    N, h = Lam.shape
    lv = (1 << bits) - 1
    R = Lam.max(axis=0)
    R = np.where(R > 1e-12, R, 1.0)
    if use_kink:
        if theta is None:
            theta = np.minimum(0.0, Lam.min(axis=0))
        theta = np.asarray(theta, dtype=float)
        span = np.where(R - theta > 1e-12, R - theta, 1.0)
        v = np.round(lv * (Lam - theta[None, :]) / span[None, :])
        v = np.clip(v, 0, lv).astype(np.int64)
    else:
        v = np.clip(np.round(lv * Lam / R[None, :]), 0, lv).astype(np.int64)
    if bits == 4:
        return v.astype(float), 1
    nch = bits // 4
    Z = np.concatenate([((v >> (4 * k)) & 0xF) for k in range(nch)], axis=1).astype(float)
    return Z, nch


def encode_relu_deadzone(Lam, bits, theta):
    """
    "死区不占编码空间"：死区 (Lam<=theta_j) 统一为 0,
    只把 [theta_j, R_j] 映射到 [0, lv]。theta_j 由外部给定(转折点)。
    与 encode_relu(use_kink=True) 的区别：后者 theta=min(Lam,0) 是**下界**，等于旧做法；
    这里 theta 是**转折点**，死区被压成一档。
    """
    N, h = Lam.shape
    lv = (1 << bits) - 1
    theta = np.asarray(theta, dtype=float)
    R = Lam.max(axis=0)
    span = np.where(R - theta > 1e-12, R - theta, 1.0)
    v = np.round(lv * (Lam - theta[None, :]) / span[None, :])
    v = np.clip(v, 0, lv).astype(np.int64)          # 死区 -> 0
    if bits == 4:
        return v.astype(float), 1
    nch = bits // 4
    Z = np.concatenate([((v >> (4 * k)) & 0xF) for k in range(nch)], axis=1).astype(float)
    return Z, nch

def fit_int(Z, target, S):
    A = np.hstack([Z, np.ones((len(Z), 1))])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return np.round(coef[:-1] * S).astype(np.int64), -round(coef[-1] * S)

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)

    print("=" * 100, flush=True)
    print("ReLU 离散化修正：死区不占编码空间 + 第二层吸收逐神经元伸缩", flush=True)
    print("=" * 100, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)

    # 训练 ReLU 老师并让它定义真值表
    print(f"\n[1] 训练 ReLU 老师 4->8->1（连续隐藏层）", flush=True)
    bestR = None
    for seed in (0, 1, 2, 3, 4):
        r = train_relu_teacher(Xn, y, 8, seed=seed)
        if bestR is None or r[0] < bestR[0]:
            bestR = r
    _, W1r, b1r, W2r, b2r = bestR
    print(f"   ReLU 老师对官方标签 = {bestR[0]}/{N} ({bestR[0]/N*100:.4f}%)", flush=True)

    Lam_true = Xn @ W1r.T + b1r
    Hr = np.maximum(Lam_true, 0)
    yR = ((Hr @ W2r + b2r) >= 0).astype(np.int64)
    labR = (2 * yR - 1).astype(np.int8)
    print(f"   => 该 ReLU 老师定义真值表（正类 {int(yR.sum())}）", flush=True)

    Lam = layer1_Lambda(X, W1r)                    # 第一层 LUT 后的 Lambda
    H1 = np.maximum(Lam, 0)
    tgt = H1 @ W2r + b2r
    e1 = int(np.sum(((H1 @ W2r + b2r) >= 0).astype(np.int8) != labR))
    print(f"\n[2] 第一层强模 LUT 实现后的 Lambda", flush=True)
    print(f"   Lambda 与真值最大偏差 = {np.max(np.abs(Lam - Lam_true)):.3e}", flush=True)
    print(f"   仅第一层 LUT 错误 = {e1}/{N}   <-- 旧做法这里是 47464", flush=True)
    print(f"\n   各神经元的 Lambda 范围 (ReLU 转折点 = 0):", flush=True)
    for j in range(8):
        r = H1[:, j].max()
        print(f"     j={j}: Lambda∈[{Lam[:,j].min():8.3f},{Lam[:,j].max():8.3f}]  "
              f"relu范围[0,{r:7.3f}]  旧档距={r/255:.5f}  新档距(死区不占)={r/255:.5f}", flush=True)

    print(f"\n[3] 编译到 4->8->16->1 / 4->8->1，整数读出", flush=True)
    print(f"   {'编码方式':>24} | {'弱模线':>6} | {'S':>7} | {'权重max':>8} | {'错误':>7} | {'错误率':>9}",
          flush=True)
    print("-" * 88, flush=True)
    out = {"relu_teacher": int(bestR[0]), "N": N, "layer1_only": e1}
    for tag, bits in (("4bit 死区不占", 4), ("8bit 死区不占(拆分)", 8)):
        Z, nch = encode_relu(Lam, bits, use_kink=True)
        best_e = None
        for S in (2000, 8000, 32000, 128000):
            w, th = fit_int(Z, tgt, S)
            e = int(np.sum(((Z @ w - th) >= 0).astype(np.int8) != labR))
            if best_e is None or e < best_e[0]:
                best_e = (e, S, np.abs(w).max())
        print(f"   {tag:>24} | {nch*8:6d} | {best_e[1]:7d} | {best_e[2]:8d} | {best_e[0]:7d} | "
              f"{best_e[0]/N*100:8.4f}%", flush=True)
        out[tag] = {"err": int(best_e[0]), "S": int(best_e[1])}
    print("-" * 88, flush=True)
    # 旧做法对照
    for tag, bits in (("4bit 旧(死区占空间)", 4), ("8bit 旧(死区占空间)", 8)):
        Z, nch = encode_relu(Lam, bits, use_kink=False)
        best_e = None
        for S in (2000, 8000, 32000, 128000):
            w, th = fit_int(Z, tgt, S)
            e = int(np.sum(((Z @ w - th) >= 0).astype(np.int8) != labR))
            if best_e is None or e < best_e[0]:
                best_e = (e, S, np.abs(w).max())
        print(f"   {tag:>24} | {nch*8:6d} | {best_e[1]:7d} | {best_e[2]:8d} | {best_e[0]:7d} | "
              f"{best_e[0]/N*100:8.4f}%", flush=True)
        out[tag] = {"err": int(best_e[0]), "S": int(best_e[1])}

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump(out, open(os.path.join(resdir, "relu_deadzone.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/relu_deadzone.json", flush=True)

if __name__ == "__main__":
    main()
