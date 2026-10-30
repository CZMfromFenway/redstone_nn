#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：强模+拆分结构（4 -> 8强模 -> 16弱模 -> 1）
=================================================
【新结构（czm 提出）】
  把一个强模信号按 tanh 的方式变成两个 0~15 弱模信号：
  相当于把 tanh 的输出映射到 0x00~0xFF，再拆成高/低半字节各一条弱模线。
  这样层间带宽从 4 bit 提到 8 bit，代价是纵向通道数翻倍。
  拆分层的体积等效于一个"16 输入 x h 输出"的强模 LUT 层。

【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数

  对比方案
    A 基线      隐藏层量化到 16 级 (4 bit)
    B 新结构    隐藏层量化到 256 级 (8 bit)，拆成 (hi, lo) 两组 0~15 弱模
    C 连续      隐藏层连续 tanh（上界）
    D 加宽      同样 4 bit，但把隐藏单元数加到等效体积（16倍）——只看精度趋势
  最后一层输入都是 0~15 整数，输出不量化（高精度阈值）。

  体积模型（按 czm 的尺度）
    单元截面 4(输入线) x 1(神经元)，长 32~40
    每层单元数 = 输入线数 x 神经元数
    与"强度"相关的长度成本按容器峰值计
"""
import numpy as np, json, os, time

L = 16
BS = 8192

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def quant(Hc, levels):
    return np.clip(np.round((Hc + 1) * (levels - 1) / 2), 0, levels - 1)

def featurize(Hq, levels):
    """把一个 level 级整数编码展开成若干 0~15 的弱模通道"""
    if levels <= 16:
        return Hq.astype(np.float64)
    n_byte = int(np.log2(levels) // 4)          # 256 -> 2 个弱模通道
    cols = []
    v = Hq.astype(np.int64)
    for b in range(n_byte):
        cols.append((v >> (4 * b)) & 0xF)
    return np.stack(cols, axis=2).reshape(Hq.shape[0], -1).astype(np.float64)

def train(Xn, y, h, levels, seed=0, steps=2500, lr=0.03, bs=BS, tag="run", log_every=500):
    """
    训练 4->h->1；隐藏层输出量化到 levels 级；用 featurize 展开成 0~15 通道；
    输出层线性判别（不量化），hinge(3/4) + BCE(1/4)
    """
    rng = np.random.default_rng(seed); N = len(Xn)
    W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
    d = featurize(np.zeros((1, h), dtype=np.int64), levels).shape[1]
    v = rng.normal(0, 0.6, d); th = 0.0
    ps = [W1, b1, v, np.array(th)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    s = (levels - 1) / 2; ys = 2 * y - 1
    best = None; t0 = time.time()
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]; yy = ys[idx]
        Hc = np.tanh(xb @ W1.T + b1)
        Hq = quant(Hc, levels)
        F = featurize(Hq, levels)                      # (bs, d)
        z = F @ v - th
        if t <= steps * 3 // 4:
            m = 1 - yy * z; dz = (-yy * (m > 0)) / bs
        else:
            p = 1 / (1 + np.exp(-z)); dz = (p - yb) / bs
        gv = F.T @ dz; gth = np.array(-dz.sum())
        # 反传到 Hq（要对每个通道的贡献求和），再经 STE 与 tanh
        if levels <= 16:
            dHq = np.outer(dz, v)
        else:
            dHq = np.zeros((len(xb), h))
            nb = d // h
            for b in range(nb):
                dHq += np.outer(dz, v[b * h:(b + 1) * h])   # 每通道共享同一份 Hq
        dH = dHq * (s * (1 - Hc ** 2))
        gW1 = dH.T @ xb; gb1 = dH.sum(0)
        for i, (pp, gg) in enumerate(zip(ps, [gW1, gb1, gv, gth])):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % log_every == 0 or t == steps:
            Hq_all = quant(np.tanh(Xn @ W1.T + b1), levels)
            F_all = featurize(Hq_all, levels)
            e = int(np.sum(((F_all @ v - th) >= 0).astype(np.int64) != y))
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), v.copy(), th)
            print(f"      [{tag}] step {t:5d} ({time.time()-t0:4.0f}s): 错误={e:6d} 最好={best[0]}",
                  flush=True)
    return best

def volume(hidden_units, weak_channels_per_unit, n_in=4, L_cell=36):
    """
    体积核算（单元数 x 单位体积）
      每层单元数 = 输入线数 x 神经元数
      拆分层：从 1 个强模信号读出 2 个弱模，等效于 16 输入 x h 输出的一层
    返回各层单元数与总单元数
    """
    layers = {}
    # 第一层：4 个弱模输入 -> hidden_units 个强模神经元
    layers["第1层 4->h"] = n_in * hidden_units
    # 拆分层：等效 16 输入 x hidden_units 输出
    if weak_channels_per_unit > 1:
        layers["拆分层 16->h"] = 16 * hidden_units
    # 最后一层：弱模输入 -> 1 个输出
    layers["输出层"] = weak_channels_per_unit * hidden_units * 1
    return layers, sum(layers.values())

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1t = np.array(m["W1"]); b1t = np.array(m["b1"]); W2t = np.array(m["W2"]); b2t = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errA = int(np.sum(((np.tanh(Xn @ W1t.T + b1t) @ W2t + b2t) >= 0).astype(np.int8) != lab))

    print("=" * 104, flush=True)
    print("实验：强模+拆分结构（4 -> h强模 -> 2h弱模 -> 1）", flush=True)
    print("=" * 104, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)
    print(f"目标标签 : N4_MLP_T1（正类 {int(y.sum())}）", flush=True)
    print(f"参照     : A 原 Teacher（隐藏层连续）= {errA}/{N}", flush=True)

    plan = [
        ("4->8->1      (4bit 基线)", 8, 16),
        ("4->16->1     (4bit)", 16, 16),
        ("4->32->1     (4bit)", 32, 16),
        ("4->8->16->1  (8bit 拆分)", 8, 256),
        ("4->16->32->1 (8bit 拆分)", 16, 256),
        ("4->32->64->1 (8bit 拆分)", 32, 256),
        ("4->64->1     (4bit)", 64, 16),
    ]
    results = []
    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    for name, h, levels in plan:
        print(f"\n{'#'*100}\n### {name}\n{'#'*100}", flush=True)
        best = None
        for seed in (0, 1):
            r = train(Xn, y, h, levels, seed=seed, steps=2000, tag=f"{name} s{seed}")
            if best is None or r[0] < best[0]:
                best = r
        nch = featurize(np.zeros((1, h), dtype=np.int64), levels).shape[1] // h
        lay, tot = volume(h, nch)
        print(f"   => 最好错误 = {best[0]}/{N} ({best[0]/N*100:.4f}%)   "
              f"弱模通道/单元 = {nch}   单元数 = {tot}  {lay}", flush=True)
        results.append({"name": name, "h": h, "levels": levels, "channels": nch,
                        "err": int(best[0]), "cells": int(tot), "layers": lay,
                        "cell_per_err": float(tot / max(best[0], 1))})
        json.dump({"A": errA, "N": N, "results": results},
                  open(os.path.join(resdir, "split_structure.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    print("\n" + "=" * 110, flush=True)
    print("汇总", flush=True)
    print("=" * 110, flush=True)
    print(f"   {'方案':>26} | {'通道':>4} | {'单元数':>7} | {'错误':>7} | {'错误率':>9} | {'单元/错误':>10}")
    print(f"   {'A 原Teacher(连续)':>26} | {'-':>4} | {'-':>7} | {errA:7d} | {errA/N*100:8.4f}% | {'-':>10}")
    for r in results:
        print(f"   {r['name']:>26} | {r['channels']:>4} | {r['cells']:>7} | {r['err']:>7} | "
              f"{r['err']/N*100:8.4f}% | {r['cell_per_err']:>10.1f}")
    print("\n已写出: results/split_structure.json", flush=True)

if __name__ == "__main__":
    main()
