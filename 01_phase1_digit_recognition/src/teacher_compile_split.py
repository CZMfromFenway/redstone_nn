#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：把原 Teacher 直接编译到"强模+拆分"结构（4 -> 8 -> 2h 弱模 -> 1）
======================================================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数

【编排（czm 提出）】
  1) 4->8 完全沿用原 Teacher 的 W1,b1，隐藏层 = tanh(u_j)
  2) 把 tanh(u_j) ∈ (-1,1) 编码成 b 位整数 v_j ∈ {0..2^b-1}
  3) 拆成高低半字节 v_j = sum_k nib_{j,k} * 16^k，共 (b/4)*8 条 0~15 弱模线
  4) 输出层：对半字节的线性读出。低位/高位步长成 1/16（"步长浮动"）
  5) 输出层可以再训练（hinge 3/4 + BCE 1/4）

【统一约定】
  split 返回 (N, nch*8)，列序为 [k=0 的 8 个神经元, k=1 的 8 个神经元, ...]
  编译权重按同一顺序排列。
"""
import numpy as np, json, os, time

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def encode(Hc, b):
    levels = 2 ** b
    return np.clip(np.round((Hc + 1) * (levels - 1) / 2), 0, levels - 1).astype(np.int64)

def split(v, b):
    """返回 (N, (b/4)*8)，列序 [k=0 的 8 个神经元, k=1 的 8 个神经元, ...]"""
    vv = np.asarray(v, dtype=np.int64)              # (N, 8)
    nch = b // 4
    return np.concatenate([((vv >> (4 * k)) & 0xF).astype(np.float64) for k in range(nch)],
                          axis=1)                   # (N, nch*8)

def err(Z, w, th, y):
    return int(np.sum(((Z @ w - th) >= 0).astype(np.int64) != y))

def compile_weights(W2, b, s, b2=0.0):
    """
    从 Teacher 的 W2 编译出对半字节的整数权重。
      tanh ≈ 2v/(2^b-1) - 1,  v = sum_k nib_k*16^k
      z = sum_j W2_j*tanh_j + b2
        = sum_{j,k} [ W2_j * 2*16^k / (2^b-1) ] * nib_{j,k}  + (b2 - sum_j W2_j)
    取整数尺度 s: A_{k,j} = round( s * 2 * 16^k * W2_j / (2^b - 1) )
    偏置项 (b2 - sum W2) 并入阈值，阈值乘以同一尺度 s。
    返回 (w, th)
    """
    nch = b // 4
    denom = (2 ** b - 1)
    w = np.zeros(nch * 8, dtype=np.float64)
    for k in range(nch):
        for j in range(8):
            w[k * 8 + j] = round(s * 2 * (16 ** k) * W2[j] / denom)
    th = -s * (b2 - W2.sum())
    return w, th

def train_readout(Z, y, w0, th0, steps=3000, lr=0.05, bs=8192, seed=0, log_every=1000,
                  tag="read"):
    rng = np.random.default_rng(seed); N = len(Z)
    w = np.asarray(w0, dtype=np.float64).copy(); th = float(th0)
    ms = np.zeros_like(w); vs = np.zeros_like(w); mb = vb = 0.0
    ys = 2 * y - 1
    best = (err(Z, w, th, y), w.copy(), th)
    t0 = time.time()
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        Zb = Z[idx]; yb = y[idx]; yy = ys[idx]
        z = Zb @ w - th
        if t <= steps * 3 // 4:
            m = 1 - yy * z; dz = (-yy * (m > 0)) / bs
        else:
            p = 1 / (1 + np.exp(-z)); dz = (p - yb) / bs
        gw = Zb.T @ dz; gb = -dz.sum()
        ms = 0.9 * ms + 0.1 * gw; vs = 0.999 * vs + 0.001 * gw ** 2
        w -= lr * (ms / (1 - 0.9 ** t)) / (np.sqrt(vs / (1 - 0.999 ** t)) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
        th -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
        if t % log_every == 0 or t == steps:
            e = err(Z, w, th, y)
            if e < best[0]:
                best = (e, w.copy(), th)
            print(f"      [{tag}] step {t:5d} ({time.time()-t0:4.0f}s): 错误={e:6d} 最好={best[0]}",
                  flush=True)
    return best

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1 = np.array(m["W1"]); b1 = np.array(m["b1"])
    W2 = np.array(m["W2"]); b2 = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    Hc = np.tanh(Xn @ W1.T + b1)
    z_exact = Hc @ W2 + b2
    errA = int(np.sum((z_exact >= 0).astype(np.int8) != lab))

    print("=" * 106, flush=True)
    print("实验：原 Teacher 直接编译到 强模+拆分 结构（4 -> 8 -> 2h 弱模 -> 1）", flush=True)
    print("=" * 106, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)
    print(f"目标标签 : N4_MLP_T1（正类 {int(y.sum())}）", flush=True)
    print(f"参照     : A 原 Teacher（4->8->1，隐藏层连续）= {errA}/{N}", flush=True)
    print(f"\n[先量决策余量] Teacher 的 logit 最小绝对值 = {np.abs(z_exact).min():.3e}", flush=True)
    print(f"   量化到 b 位时 logit 的最大扰动 ≈ |W2|_1 * 0.5 * 2/(2^b-1) = "
          f"{np.abs(W2).sum()*0.5*2/(2**8-1):.3e} (b=8)", flush=True)
    print(f"   => 余量远小于扰动，**无损复现不可能**；实际损失取决于有多少样本贴着边界。",
          flush=True)

    # 统一用"不拆分、只量化"作为 b 的扫描（读出用编译权重 + 训练）
    print(f"\n[基线] 隐藏层量化到 b 位、**不拆分**，读出用编译权重（并训练）", flush=True)
    print(f"   {'b':>3} | {'级数':>7} | {'编译错误':>9} | {'训练后错误':>10} | {'错误率':>9}", flush=True)
    base = []
    for b in (4, 6, 8, 12, 16):
        v = encode(Hc, b).astype(np.float64)                  # (N,8)
        denom = (2 ** b - 1)
        s = 10000
        w0 = np.round(s * 2 * W2 / denom)
        th0 = -s * (b2 - W2.sum())
        e0 = err(v, w0, th0, y)
        e1, _, _ = train_readout(v, y, w0, th0, steps=2000, seed=0,
                                 tag=f"nosplit b{b}", log_every=2000)
        print(f"   {b:3d} | {2**b:7d} | {e0:9d} | {e1:10d} | {e1/N*100:8.4f}%", flush=True)
        base.append({"b": b, "compile_err": int(e0), "train_err": int(e1)})

    print(f"\n[主实验] 拆分结构（高低半字节，权重比 16:1）", flush=True)
    print(f"   {'b':>3} | {'弱模通道':>8} | {'s':>6} | {'编译错误':>9} | {'训练后错误':>10} | "
          f"{'错误率':>9} | {'相对Teacher损失':>14}", flush=True)
    rows = []
    for b in (4, 8, 12, 16):
        v = encode(Hc, b)
        Z = split(v, b)                                       # (N, (b/4)*8)
        nch = Z.shape[1]
        for s in (16, 64, 256, 1024):
            w0, th0 = compile_weights(W2, b, s, b2)
            e0 = err(Z, w0, th0, y)
            e1, _, _ = train_readout(Z, y, w0, th0, steps=2500, seed=0,
                                     tag=f"b{b} s{s}", log_every=2500)
            loss = (e1 - errA) / N * 100
            print(f"   {b:3d} | {nch:8d} | {s:6d} | {e0:9d} | {e1:10d} | {e1/N*100:8.4f}% | "
                  f"{loss:13.4f}%", flush=True)
            rows.append({"b": b, "channels": int(nch), "s": s,
                         "compile_err": int(e0), "train_err": int(e1)})

    print("\n" + "=" * 106, flush=True)
    print("汇总", flush=True)
    print("=" * 106, flush=True)
    print(f"   {'方案':>34} | {'弱模通道':>8} | {'错误':>7} | {'错误率':>9} | {'相对损失':>10}")
    print(f"   {'A 原 Teacher（连续）':>34} | {'-':>8} | {errA:7d} | {errA/N*100:8.4f}% | {'0.0000%':>10}")
    for r in base:
        print(f"   {'不拆分 b=' + str(r['b']) + '（训练后）':>34} | {'8':>8} | {r['train_err']:7d} | "
              f"{r['train_err']/N*100:8.4f}% | {(r['train_err']-errA)/N*100:9.4f}%")
    for r in rows:
        nm = f"拆分 b={r['b']} s={r['s']}"
        print(f"   {nm:>34} | {r['channels']:8d} | {r['train_err']:7d} | "
              f"{r['train_err']/N*100:8.4f}% | {(r['train_err']-errA)/N*100:9.4f}%")

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump({"A": errA, "N": N, "base": base, "rows": rows,
               "margin": float(np.abs(z_exact).min())},
              open(os.path.join(resdir, "teacher_compile_split.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/teacher_compile_split.json", flush=True)

if __name__ == "__main__":
    main()
