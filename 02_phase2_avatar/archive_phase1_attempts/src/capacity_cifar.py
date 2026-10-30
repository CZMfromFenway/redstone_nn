#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-10 cat 类的 8x8 容量下界（独立验证 MC 材质结论）
========================================================
与 capacity_ae2.py 同一套方法，只换数据源:
  CIFAR-10 (32x32 彩色) -> 8x8 降采样 -> k-means 量化到 16/48 色
  取 cat 类 (label=3)
目的: 检验"内在维度约 8 维"这个结论是否依赖数据集。
"""
import numpy as np, os, json, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")


def load_cifar_cat():
    tgz = os.path.join(DATA, "cifar-10-binary.tar.gz")
    wanted = [f"data_batch_{i}.bin" for i in range(1, 6)] + ["test_batch.bin"]
    imgs = []
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar.getmembers():
            base = os.path.basename(m.name)
            if base in wanted:
                buf = tar.extractfile(m).read()
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 3073)
                sel = arr[arr[:, 0] == 3, 1:]
                imgs.append(sel.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                print(f"  读入 {base}: {len(sel)} 张 cat")
    return np.concatenate(imgs).astype(np.float32)


def down8(X32):
    n = len(X32)
    return X32.reshape(n, 8, 4, 8, 4, 3).mean(axis=(2, 4)) / 255.0


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
        std = np.sqrt(2.0 / dims[i])
        P.append(rng.normal(0, std, (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
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


def train(P0, Xtr, Xte, steps=6000, lr=0.01, bs=128, seed=0):
    P = [p.copy() for p in P0]
    ms = [np.zeros_like(p) for p in P]; vs = [np.zeros_like(p) for p in P]
    rng = np.random.default_rng(seed); n = len(Xtr); L = len(P) // 2
    best = (1e9, None)
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
            te = float(((fwd(P, Xte)[-1] - Xte) ** 2).mean())
            if te < best[0]:
                best = (te, [p.copy() for p in P])
    return best


def main():
    print("=" * 104)
    print("CIFAR-10 cat 类 8x8 容量下界（独立验证）")
    print("=" * 104)
    X32 = load_cifar_cat()
    print(f"CIFAR cat 原始: {X32.shape}")
    A = down8(X32)
    rng = np.random.default_rng(0); A = A[rng.permutation(len(A))]
    Atr, Ate = A[:5000], A[5000:6000]
    print(f"8x8: train {Atr.shape} test {Ate.shape}")

    out = {}
    for C in (16, 48):
        lab_tr, pal = quantize(Atr, C, 1)
        px = Ate.reshape(-1, 3)
        lab_te = ((px[:, None] - pal[None]) ** 2).sum(2).argmin(1).reshape(len(Ate), 64)
        Ttr = (lab_tr / (C - 1)).astype(np.float32)
        Tte = (lab_te / (C - 1)).astype(np.float32)
        base = float(((Tte - Ttr.mean(0)) ** 2).mean())
        print(f"\n{'#'*104}\n### C={C} 色   基准(均值图) MSE = {base:.6f}\n{'#'*104}")
        print(f"  {'d':>4} | {'h':>5} | {'参数':>7} | {'红石单元':>8} | {'测试MSE':>9} | "
              f"{'相对基准':>8} | 状态")
        print("  " + "-" * 78)
        rows = []
        for d_z, h in [(4, 32), (8, 64), (16, 64), (32, 128), (48, 128), (64, 128)]:
            t0 = time.time()
            dims = [64, h, d_z, h, 64]
            P0 = mlp_kaiming(dims, seed=0)
            r = train(P0, Ttr, Tte, steps=6000, lr=0.01, seed=0)
            tr = float(((fwd(r[1], Ttr[:2000])[-1] - Ttr[:2000]) ** 2).mean())
            cells = 64 * h + h * d_z + d_z * h + h * 64
            dead = abs(tr - base) < 1e-6
            print(f"  {d_z:>4} | {h:>5} | {sum(p.size for p in P0):>7} | {cells:>8} | "
                  f"{r[0]:>9.6f} | {r[0]/base:>7.2%} | {'死亡' if dead else '学习'}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            rows.append({"C": C, "d": d_z, "h": h, "cells": cells,
                         "mse": r[0], "rel": r[0] / base, "dead": bool(dead)})
        out[f"C{C}"] = {"baseline": base, "rows": rows}

    json.dump(out, open(os.path.join(RES, "capacity_cifar.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: {RES}/capacity_cifar.json")


if __name__ == "__main__":
    main()
