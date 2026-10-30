#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：ReLU 路线 —— 线性强模->弱模 vs 非线性拆分
=================================================
【动机】
  非线性强模->弱模（tanh 的 0x00~0xFF 拆分）需要 16 单元/神经元，体积大。
  线性强模->弱模只需 3~4 单元/神经元，但激活函数只能是 ReLU。
  本实验：建 ReLU 老师，测 4->(8->16)->1 在 ReLU 下的精度，并与 tanh 对比。

【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  结构       4->8->16->1   (第一层 8 个神经元; 中间线性/非线性转弱模; 输出 1 bit)
  评价       全部 65536 状态上的最终 0/1 错误数

【四个对照】
  T1  tanh 老师（官方 N4_MLP_T1 参数）
  T2  tanh 结构重训（同结构从零训练, 隐藏层连续）
  R1  ReLU 老师（同结构从零训练, 隐藏层连续）
  R2  ReLU 结构重训（同上, 另一随机种子）
  然后：
   T1c 从 T1 直接编译到 4->8->16->1（整数读出）  <- 已验证 75 错
   R1c 从 R1 直接编译到 4->8->16->1（整数读出）
"""
import numpy as np, json, os, time

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

# ------------------------------------------------ 训练
def train(Xn, ytr, h, act, seed=0, steps=4000, lr=0.02, bs=8192, log_every=1000, tag="t",
          eval_lab=None):
    """4->h->1, act ∈ {'tanh','relu'}; 训练目标 = ytr, 早停评价 = eval_lab(默认 ytr)"""
    rng = np.random.default_rng(seed); N = len(Xn)
    ev = ytr if eval_lab is None else eval_lab
    W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
    W2 = rng.normal(0, 0.6, h); b2 = 0.0
    ps = [W1, b1, W2, np.array(b2)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    def fwd(x):
        U = x @ W1.T + b1
        return (np.tanh(U) if act == "tanh" else np.maximum(U, 0)), U
    best = None; t0 = time.time()
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = ytr[idx]
        H, U = fwd(xb)
        z = H @ W2 + b2
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); dz = (p - yb) / bs
        dH = np.outer(dz, W2)
        if act == "tanh":
            dU = dH * (1 - H ** 2)
        else:
            dU = dH * (U > 0)
        g = [dU.T @ xb, dU.sum(0), H.T @ dz, np.array(dz.sum())]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % log_every == 0 or t == steps:
            Hf, _ = fwd(Xn)
            e = int(np.sum(((Hf @ W2 + b2) >= 0).astype(np.int8) != ev))
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), W2.copy(), float(b2))
            print(f"      [{tag}] step {t:5d} ({time.time()-t0:4.0f}s): 错误={e:6d} 最好={best[0]}",
                  flush=True)
    return best

# ------------------------------------------------ 编译到 4->8->16->1
def compile_and_eval(X, Xn, W1, b1, W2, b2, act, lab, s2, bits=8):
    """
    第一层: 强模 LUT, rho=868*|W1|/max|W1|
    第二层: 若 act='tanh' -> 非线性拆分(8bit 高低半字节);
            若 act='relu' -> 线性缩放(直接 4bit 量化, 因为线性不需要查表)
    读出: 最小二乘后取整 (整数权重)
    """
    N = len(X); h = W1.shape[0]
    rho = np.zeros_like(W1, float)
    for j in range(h):
        mj = np.abs(W1[j]).max()
        rho[j] = 868.0 * np.abs(W1[j]) / mj if mj > 1e-12 else 0.0
    x = np.arange(16)
    lam = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
    # 负权重用下降阶梯(保持 lambda>=0): lam_desc = 15*rho - lam_up
    lam_up = lam
    lam_dn = np.round(15 * rho[:, :, None] - lam_up.astype(float)).astype(np.int64)
    sgn = np.where(W1 >= 0, 1, -1)
    Xi = X.astype(np.int64)
    s = np.zeros((N, h), dtype=np.int64)
    for j in range(h):
        for i in range(4):
            v = lam_up[j, i, Xi[:, i]] if sgn[j, i] > 0 else lam_dn[j, i, Xi[:, i]]
            s[:, j] += v
    scale1 = np.array([(np.abs(W1[j]).max() / (868 * 7.5)) if np.abs(W1[j]).max() > 1e-12 else 0.0
                       for j in range(h)])
    U1 = s * scale1[None, :] - W1.sum(axis=1)[None, :] + b1[None, :]   # 第一层预激活
    if act == "tanh":
        Hc = np.tanh(U1)
        v = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)
        Z = np.concatenate([(v >> 4), (v & 0xF)], axis=1).astype(float)
        zt = np.tanh(U1) @ W2 + b2
    else:
        # ReLU: 线性转弱模。逐神经元按各自实际最大值定标到 0~15（线性, 不需查表）
        Hr = np.maximum(U1, 0)
        colmax = Hr.max(axis=0)                      # 每个神经元自己的最大值
        colmax = np.where(colmax > 1e-12, colmax, 1.0)
        Z = np.clip(np.round(Hr * 15 / colmax[None, :]), 0, 15).astype(float)
        zt = Hr @ W2 + b2
    A = np.hstack([Z, np.ones((N, 1))])
    coef, *_ = np.linalg.lstsq(A, zt, rcond=None)
    w = np.round(coef[:-1] * s2).astype(np.int64); th = -round(coef[-1] * s2)
    e = int(np.sum(((Z @ w - th) >= 0).astype(np.int8) != lab))
    return e, w, th, Z.shape[1]

lab_glob = None

def main():
    global lab_glob
    m, lab, X = load_experiment()
    lab_glob = lab
    y = (lab > 0).astype(np.int64)
    W1o = np.array(m["W1"]); b1o = np.array(m["b1"])
    W2o = np.array(m["W2"]); b2o = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errA = int(np.sum(((np.tanh(Xn @ W1o.T + b1o) @ W2o + b2o) >= 0).astype(np.int8) != lab))

    print("=" * 100, flush=True)
    print("实验：ReLU 路线 vs tanh 路线", flush=True)
    print("=" * 100, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)
    print(f"参照 T1  : 官方 tanh 老师 = {errA}/{N}", flush=True)

    out = {"N": N, "T1_teacher": errA}
    print(f"\n[一] 从零训练同结构 4->8->1，比较 tanh 与 ReLU 能拟合到什么程度", flush=True)
    print(f"   {'激活':>6} | {'种子':>4} | {'连续隐藏层错误':>14} | {'错误率':>9}", flush=True)
    trained = {}
    for act in ("tanh", "relu"):
        for seed in (0, 1):
            r = train(Xn, y, 8, act, seed=seed, steps=4000, tag=f"{act} s{seed}")
            print(f"   {act:>6} | {seed:4d} | {r[0]:14d} | {r[0]/N*100:8.4f}%", flush=True)
            trained[(act, seed)] = r
    out["trained"] = {f"{a}_{s}": int(v[0]) for (a, s), v in trained.items()}

    print(f"\n[二] 用自洽真值表：ReLU 老师定义它自己的真值表，能否被同结构复现", flush=True)
    # ReLU 老师(seed 0) 输出作为真值表
    rR = trained[("relu", 0)]
    _, W1r, b1r, W2r, b2r = rR
    Ur = np.maximum(Xn @ W1r.T + b1r, 0)
    yR = ((Ur @ W2r + b2r) >= 0).astype(np.int64)
    labR = (2 * yR - 1).astype(np.int8)
    print(f"   ReLU 老师(seed0) 的正类数 = {int(yR.sum())}", flush=True)
    for act in ("relu", "tanh"):
        for seed in (0, 1, 2):
            r = train(Xn, yR, 8, act, seed=seed, steps=4000, tag=f"reluTT {act} s{seed}")
            print(f"   [{act} 拟合 ReLU 自洽真值表] seed={seed}: 错误={r[0]} "
                  f"({r[0]/N*100:.4f}%)", flush=True)

    print(f"\n[三] 编译到 4->(8->16)->1，整数读出", flush=True)
    print(f"   {'方案':>34} | {'弱模通道':>8} | {'错误':>7} | {'错误率':>9}", flush=True)
    print("-" * 76, flush=True)
    rows = []
    # tanh 官方老师
    for s2 in (2000, 8000, 32000):
        e, w, th, d = compile_and_eval(X, Xn, W1o, b1o, W2o, b2o, "tanh", lab, s2)
        print(f"   {'T1 官方tanh老师 编译 S=' + str(s2):>34} | {d:8d} | {e:7d} | {e/N*100:8.4f}%",
              flush=True)
        rows.append({"case": "T1_tanh_teacher", "S": s2, "dim": d, "err": int(e)})
    # ReLU 老师(自洽真值表)
    for s2 in (2000, 8000, 32000):
        e, w, th, d = compile_and_eval(X, Xn, W1r, b1r, W2r, b2r, "relu", labR, s2)
        print(f"   {'R1 ReLU老师 编译 S=' + str(s2):>34} | {d:8d} | {e:7d} | {e/N*100:8.4f}%",
              flush=True)
        rows.append({"case": "R1_relu_teacher", "S": s2, "dim": d, "err": int(e)})
    out["compile"] = rows

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump(out, open(os.path.join(resdir, "relu_vs_tanh.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/relu_vs_tanh.json", flush=True)

if __name__ == "__main__":
    main()
