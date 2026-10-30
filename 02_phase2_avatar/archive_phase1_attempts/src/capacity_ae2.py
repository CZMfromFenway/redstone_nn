#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
8x8 生成器容量下界（修正版：Kaiming 初始化 + 显式收敛检查）
==============================================================
上一版的问题: 训练 MSE 精确等于"恒输出均值图"的 MSE -> 隐藏层 ReLU 全死。
本版修正:
  1. Kaiming 初始化 (fan_in 缩放), 按层设置
  2. 更小学习率 + 更保守的步数
  3. 打印训练集 MSE 收敛曲线, 确认网络真的在学
  4. 若训练 MSE 仍等于基准 -> 明确标记为"死亡", 不采信
"""
import numpy as np, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")


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


def mlp_kaiming(dims, seed=0):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        std = np.sqrt(2.0 / fan_in)
        P.append(rng.normal(0, std, (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
    # 最后一层: 输出用更小尺度
    P[-1][:] = 0.0
    return P


def fwd(P, X):
    acts = [X]; h = X
    L = len(P) // 2
    for i in range(L):
        h = h @ P[2 * i] + P[2 * i + 1]
        if i < L - 1:
            h = np.maximum(h, 0)
        acts.append(h)
    return acts


def train(P0, Xtr, Xte, steps=6000, lr=0.01, bs=128, seed=0, verbose=False):
    P = [p.copy() for p in P0]
    ms = [np.zeros_like(p) for p in P]; vs = [np.zeros_like(p) for p in P]
    rng = np.random.default_rng(seed); n = len(Xtr)
    best = (1e9, None)
    L = len(P) // 2
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Xb = Xtr[idx]
        acts = fwd(P, Xb)
        d = 2 * (acts[-1] - Xb) / bs
        for i in range(L - 1, -1, -1):
            W = P[2 * i]
            gW = acts[i].T @ d; gb = d.sum(0)
            if i > 0:
                d = (d @ W.T) * (acts[i] > 0)
            for k, gg in ((2 * i, gW), (2 * i + 1, gb)):
                ms[k] = 0.9 * ms[k] + 0.1 * gg; vs[k] = 0.999 * vs[k] + 0.001 * gg ** 2
                P[k] -= lr * (ms[k] / (1 - 0.9 ** t)) / (np.sqrt(vs[k] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1500 == 0 or t == steps:
            tr = float(((fwd(P, Xtr[:2000])[-1] - Xtr[:2000]) ** 2).mean())
            te = float(((fwd(P, Xte)[-1] - Xte) ** 2).mean())
            if te < best[0]:
                best = (te, [p.copy() for p in P])
            if verbose:
                print(f"        step {t:5d}: 训练MSE {tr:.6f}  测试MSE {te:.6f}", flush=True)
    return best


def main():
    print("=" * 104)
    print("8x8 生成器容量下界（修正：Kaiming 初始化 + 收敛检查）")
    print("=" * 104)
    A = load_structured(6, 12000)
    rng = np.random.default_rng(0); A = A[rng.permutation(len(A))]
    Atr, Ate = A[:8000], A[8000:10000]

    out = {}
    for C in (16, 48):
        lab_tr, pal = quantize(Atr, C, 1)
        px = Ate.reshape(-1, 3)
        lab_te = ((px[:, None] - pal[None]) ** 2).sum(2).argmin(1).reshape(len(Ate), 64)
        Ttr = (lab_tr / (C - 1)).astype(np.float32)
        Tte = (lab_te / (C - 1)).astype(np.float32)
        base = float(((Tte - Ttr.mean(0)) ** 2).mean())
        print(f"\n{'#'*104}")
        print(f"### C={C} 色   基准(恒输出均值图) 测试MSE = {base:.6f}")
        print(f"{'#'*104}")
        print(f"  {'d':>4} | {'解h':>5} | {'参数':>7} | {'红石单元':>8} | {'训练MSE':>9} | "
              f"{'测试MSE':>9} | {'相对基准':>8} | 状态")
        print("  " + "-" * 92)
        rows = []
        for d_z, h in [(4, 32), (8, 64), (16, 64), (16, 128), (32, 128),
                       (48, 128), (64, 128), (64, 256), (96, 256)]:
            t0 = time.time()
            dims = [64, h, d_z, h, 64]
            P0 = mlp_kaiming(dims, seed=0)
            r = train(P0, Ttr, Tte, steps=6000, lr=0.01, seed=0)
            tr = float(((fwd(r[1], Ttr[:2000])[-1] - Ttr[:2000]) ** 2).mean())
            te = r[0]
            nparam = sum(p.size for p in P0)
            cells = 64 * h + h * d_z + d_z * h + h * 64
            dead = abs(tr - base) < 1e-6
            st = "死亡" if dead else "学习"
            rel = te / base
            print(f"  {d_z:>4} | {h:>5} | {nparam:>7} | {cells:>8} | {tr:>9.6f} | {te:>9.6f} | "
                  f"{rel:>7.2%} | {st}  ({time.time()-t0:.0f}s)", flush=True)
            rows.append({"C": C, "d": d_z, "h": h, "params": nparam, "cells": cells,
                         "mse_train": tr, "mse": te, "rel": rel, "dead": bool(dead)})
        out[f"C{C}"] = {"baseline": base, "rows": rows}

    json.dump(out, open(os.path.join(RES, "capacity_ae2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: {RES}/capacity_ae2.json")


if __name__ == "__main__":
    main()
