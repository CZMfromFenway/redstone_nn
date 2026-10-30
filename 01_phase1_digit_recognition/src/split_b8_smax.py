#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：b=8 拆分结构，把 s 推到物理上限
======================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数
  结构       4->8（沿用 Teacher 的 W1,b1）-> 量化到 8 位 -> 拆成 2 组 0~15 弱模 -> 整数读出

【s 的确定（czm 意见）】
  对半字节 k 的权重： A_{j,k} = s * 2*16^k*W_j / (2^8-1) = s*2*16^k*W_j/255
  让**最大权重**的高位容器用满： A_hi_max = (容器上限)/15
  => s_max = A_hi_max * 255 / (2*|W|_max)
  实测 s_max 后，再扫描 s_max/2, s_max/4, ... 看精度-s 曲线。
"""
import numpy as np, json, os, time

CAP_CONT, CAP_STEP = 897, 868

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def encode(Hc, b=8):
    levels = 2 ** b
    return np.clip(np.round((Hc + 1) * (levels - 1) / 2), 0, levels - 1).astype(np.int64)

def split8(v):
    """(N,8) 的 8 位整数 -> (N,16)，列序 [hi 的 8 个神经元, lo 的 8 个神经元]"""
    hi = (v >> 4) & 0xF
    lo = v & 0xF
    return np.concatenate([hi.astype(np.float64), lo.astype(np.float64)], axis=1)

def err(Z, w, th, y):
    return int(np.sum(((Z @ w - th) >= 0).astype(np.int64) != y))

def compile_w(W2, s, b=8):
    """
    A_hi_j = round(s * 2 * 16 * W2_j / 255)   (高位, k=1)
    A_lo_j = round(s * 2 * 1  * W2_j / 255)   (低位, k=0)
    通道顺序 [hi 的 8 个, lo 的 8 个]
    阈值 th = -s*(b2 - sum W2)
    """
    denom = (2 ** b - 1)
    a_hi = np.round(s * 2 * 16 * W2 / denom)
    a_lo = np.round(s * 2 * 1 * W2 / denom)
    return np.concatenate([a_hi, a_lo]), -s * (0.0 - W2.sum())

def train_readout(Z, y, w0, th0, steps=2500, lr=0.05, bs=8192, seed=0, log_every=2500,
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
            zc = np.clip(z, -30, 30)
            p = 1 / (1 + np.exp(-zc)); dz = (p - yb) / bs
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
    errA = int(np.sum(((Hc @ W2 + b2) >= 0).astype(np.int8) != lab))

    b = 8
    v = encode(Hc, b)
    Z = split8(v)                                       # (N,16)
    denom = 2 ** b - 1

    print("=" * 104, flush=True)
    print("实验：b=8 拆分结构，把 s 推到物理上限", flush=True)
    print("=" * 104, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)
    print(f"参照     : A 原 Teacher = {errA}/{N}", flush=True)
    print(f"\nW2 = {np.round(W2,6).tolist()}", flush=True)
    print(f"b2 = {b2:.6f}    |W2|max = {np.abs(W2).max():.6f}", flush=True)

    # 物理上限
    A_hi_cap = CAP_STEP // 15                            # 最大步长 / 15 步
    s_max = A_hi_cap * denom / (2 * np.abs(W2).max())
    print(f"\n[s 的物理上限]", flush=True)
    print(f"   最大步长 {CAP_STEP} 分摊到 15 个步 => 单个半字节权重上限 A_hi <= {A_hi_cap}", flush=True)
    print(f"   A_hi_j = s*2*16*W_j/255，取最大 |W_j|={np.abs(W2).max():.4f}", flush=True)
    print(f"   => s_max = {A_hi_cap}*255/(2*{np.abs(W2).max():.4f}) = {s_max:.0f}", flush=True)
    print(f"   （容器约束 15*A_hi+15 <= {CAP_CONT} 给出 A_hi <= {(CAP_CONT-15)//15}，更松）", flush=True)

    print(f"\n[s 扫描] b=8，拆分结构，编译权重 + 输出层训练", flush=True)
    print(f"   {'s':>8} | {'A_hi范围':>16} | {'A_lo范围':>13} | {'越界?':>5} | "
          f"{'编译错误':>9} | {'训练后错误':>10} | {'错误率':>9}", flush=True)
    rows = []
    for s in (1024, 2048, 4096, 6000, int(s_max), int(s_max * 1.5)):
        w0, th0 = compile_w(W2, s, b)
        a_hi = w0[8:]; a_lo = w0[:8]
        over = "是" if (np.abs(a_hi).max() > CAP_STEP or np.abs(a_lo).max() > CAP_CONT) else "否"
        e0 = err(Z, w0, th0, y)
        e1, w1, th1 = train_readout(Z, y, w0, th0, steps=2500, seed=0,
                                    tag=f"b8 s{s}", log_every=2500)
        print(f"   {s:8d} | [{a_hi.min():6.0f},{a_hi.max():6.0f}] | "
              f"[{a_lo.min():5.0f},{a_lo.max():5.0f}] | {over:>5} | {e0:9d} | {e1:10d} | "
              f"{e1/N*100:8.4f}%", flush=True)
        rows.append({"s": s, "a_hi_max": float(np.abs(a_hi).max()),
                     "a_lo_max": float(np.abs(a_lo).max()),
                     "over": over == "是", "compile_err": int(e0), "train_err": int(e1)})

    # 最优 s 下的权重是否可编译成整数权重
    best = min(rows, key=lambda r: r["train_err"])
    print(f"\n[最优 s = {best['s']}]", flush=True)
    w0, th0 = compile_w(W2, best["s"], b)
    a_hi, a_lo = w0[8:], w0[:8]
    print(f"   高位权重 A_hi = {a_hi.astype(int).tolist()}", flush=True)
    print(f"   低位权重 A_lo = {a_lo.astype(int).tolist()}", flush=True)
    print(f"   比值 (应约为 16): {[round(a/b,2) if b!=0 else None for a,b in zip(a_hi,a_lo)]}", flush=True)
    print(f"   阈值 th = {th0:.1f}", flush=True)

    print("\n" + "=" * 104, flush=True)
    print("汇总", flush=True)
    print("=" * 104, flush=True)
    print(f"   {'方案':>20} | {'编译错误':>9} | {'训练后错误':>10} | {'错误率':>9} | {'相对Teacher损失':>14}")
    print(f"   {'A 原Teacher（连续）':>20} | {'-':>9} | {errA:10d} | {errA/N*100:8.4f}% | {'0.0000%':>14}")
    for r in rows:
        print(f"   {'b=8 拆分 s=' + str(r['s']):>20} | {r['compile_err']:9d} | {r['train_err']:10d} | "
              f"{r['train_err']/N*100:8.4f}% | {(r['train_err']-errA)/N*100:13.4f}%")

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump({"A": errA, "N": N, "s_max": float(s_max), "rows": rows},
              open(os.path.join(resdir, "split_b8_smax.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/split_b8_smax.json", flush=True)

if __name__ == "__main__":
    main()
