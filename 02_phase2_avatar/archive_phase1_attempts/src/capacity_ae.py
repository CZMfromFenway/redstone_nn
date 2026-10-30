#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
8x8 像素画生成器的**容量下界**：自编码器瓶颈实验
====================================================
【上一版的错误】用随机噪声 z 直接监督预测目标图像 —— z 与目标无关，
  网络只能学出条件均值，MSE 恒等于基准。测的是记忆容量，不是表示容量。

【正确做法】自编码器: x -> Enc -> z(d 维) -> Dec -> x'
  重建质量随 d 的变化 = **表示这个数据分布所需的内在维度**
  这是生成器容量的一个良定义下界:
    如果连"表示"都要 d 维，那么"生成"至少需要 d 维的隐藏层。

调色板: 16 色 / 48 色（按亮度排序，使颜色成为有序一维量）
数据: 本地 MC 材质切出的 8x8 图块（>=6 色）
"""
import numpy as np, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def load_structured(min_colors=6, want=12000, seed=0):
    X = np.load(os.path.join(DATA, "mc8.npy"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:200000]
    keep = []
    for i in idx:
        if len(np.unique(X[i].reshape(-1, 3), axis=0)) >= min_colors:
            keep.append(i)
        if len(keep) >= want:
            break
    return X[keep].astype(np.float32) / 255.0


def quantize(A, C, seed=0):
    px = A.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    cen = px[rng.choice(len(px), C, replace=False)].copy()
    for _ in range(20):
        lab = ((px[:, None] - cen[None]) ** 2).sum(2).argmin(1)
        for k in range(C):
            m = lab == k
            if m.sum():
                cen[k] = px[m].mean(0)
    lab = ((px[:, None] - cen[None]) ** 2).sum(2).argmin(1)
    lum = cen @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum); remap = np.zeros(C, int); remap[order] = np.arange(C)
    return remap[lab].reshape(len(A), 64), cen[order]


def mlp(dims, seed=0, scale=0.5):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        P.append(rng.normal(0, scale, (dims[i], dims[i + 1])))
        P.append(np.zeros(dims[i + 1]))
    return P


def fwd(P, X):
    """返回 各层激活(ReLU, 最后一层线性) 与 输入"""
    acts = [X]; h = X
    L = len(P) // 2
    for i in range(L):
        W, b = P[2 * i], P[2 * i + 1]
        h = h @ W + b
        if i < L - 1:
            h = np.maximum(h, 0)
        acts.append(h)
    return acts


def train_ae(P0, Xtr, Xte, steps=5000, lr=0.02, bs=128, seed=0):
    P = [p.copy() for p in P0]
    ms = [np.zeros_like(p) for p in P]; vs = [np.zeros_like(p) for p in P]
    rng = np.random.default_rng(seed); n = len(Xtr)
    best = (1e9, None)
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Xb = Xtr[idx]
        acts = fwd(P, Xb)
        d = 2 * (acts[-1] - Xb) / bs
        L = len(P) // 2
        for i in range(L - 1, -1, -1):
            W, b = P[2 * i], P[2 * i + 1]
            gW = acts[i].T @ d; gb = d.sum(0)
            if i > 0:
                dh = d @ W.T
                d = dh * (acts[i] > 0)
            gi = (i, gW, gb)
            ms[2 * i] = 0.9 * ms[2 * i] + 0.1 * gW; vs[2 * i] = 0.999 * vs[2 * i] + 0.001 * gW ** 2
            ms[2 * i + 1] = 0.9 * ms[2 * i + 1] + 0.1 * gb; vs[2 * i + 1] = 0.999 * vs[2 * i + 1] + 0.001 * gb ** 2
            P[2 * i] -= lr * (ms[2 * i] / (1 - 0.9 ** t)) / (np.sqrt(vs[2 * i] / (1 - 0.999 ** t)) + 1e-8)
            P[2 * i + 1] -= lr * (ms[2 * i + 1] / (1 - 0.9 ** t)) / (np.sqrt(vs[2 * i + 1] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            ae = fwd(P, Xte)[-1]
            mse = float(((ae - Xte) ** 2).mean())
            if mse < best[0]:
                best = (mse, [p.copy() for p in P])
    return best


def main():
    print("=" * 100)
    print("8x8 生成器容量下界：自编码器瓶颈实验")
    print("=" * 100)
    A = load_structured(6, 12000)
    rng = np.random.default_rng(0); A = A[rng.permutation(len(A))]
    Atr, Ate = A[:8000], A[8000:10000]
    print(f"数据: train {Atr.shape} test {Ate.shape}")

    out = {}
    for C in (16, 48):
        lab_tr, pal = quantize(Atr, C, 1)
        px = Ate.reshape(-1, 3)
        lab_te = ((px[:, None] - pal[None]) ** 2).sum(2).argmin(1).reshape(len(Ate), 64)
        Ttr = (lab_tr / (C - 1)).astype(np.float32)
        Tte = (lab_te / (C - 1)).astype(np.float32)
        # 量化本身的不可约误差
        q_err = float(((pal[lab_tr.reshape(-1)].reshape(-1, 8, 8, 3) - Atr) ** 2).mean())
        base = float(((Tte - Ttr.mean(0)) ** 2).mean())
        print(f"\n{'#'*100}")
        print(f"### C={C} 色  量化不可约 MSE={q_err:.6f}  基准(均值图) MSE={base:.6f}")
        print(f"{'#'*100}")
        print(f"  {'瓶颈d':>6} | {'解码器h':>7} | {'参数':>7} | {'红石单元':>8} | {'训练MSE':>9} | {'测试MSE':>9} | {'相对基准':>8}")
        print("  " + "-" * 82)
        rows = []
        for d_z, h in [(2, 32), (4, 32), (8, 32), (8, 64), (16, 64), (16, 128),
                       (32, 128), (48, 128), (64, 128), (64, 256)]:
            t0 = time.time()
            dims = [64, h, d_z, h, 64]
            best = (1e9, None)
            for sd in (0, 1):
                P0 = mlp(dims, seed=sd)
                r = train_ae(P0, Ttr, Tte, steps=4000, seed=sd)
                if r[0] < best[0]:
                    best = r
            nparam = sum(p.size for p in mlp(dims))
            cells = h * 64 + d_z * h + h * d_z + 64 * h
            print(f"  {d_z:>6} | {h:>7} | {nparam:>7} | {cells:>8} | {best[0]:>9.6f} | "
                  f"{best[0]:>9.6f} | {best[0]/base:>7.2%}  ({time.time()-t0:.0f}s)", flush=True)
            rows.append({"C": C, "d": d_z, "h": h, "params": nparam, "cells": cells,
                         "mse": best[0], "rel": best[0] / base})
        out[f"C{C}"] = {"quant_err": q_err, "baseline": base, "rows": rows}

    json.dump(out, open(os.path.join(RES, "capacity_ae.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: {RES}/capacity_ae.json")


if __name__ == "__main__":
    main()
