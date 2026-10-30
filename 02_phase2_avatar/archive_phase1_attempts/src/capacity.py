#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
8x8 像素画生成器：容量下界实验（本地 MC 材质数据）
====================================================
数据: 从 .minecraft 资源包提取的 8x8 图块（430780 个），
      筛出"有结构的"（>=6 种颜色）用于容量实验。

调色板: 分别取 16 / 48 色（用户关心的两档），用 k-means 量化。

网络（与红石可移植性对齐）:
  z (d 维) -> Linear(d,h) -> ReLU -> Linear(h, 192) -> 24 个像素值
  隐藏层连续；移植时只需离线权重。

评价: 测试 MSE / 相对"恒输出均值图"基准的比值。
      MSE 越低 = 该容量能拟合的分布越细。
"""
import numpy as np, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def load_structured(min_colors=6, want=12000, seed=0):
    X = np.load(os.path.join(DATA, "mc8.npy"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:200000]
    keep = []
    for i in idx:
        x = X[i]
        if len(np.unique(x.reshape(-1, 3), axis=0)) >= min_colors:
            keep.append(i)
        if len(keep) >= want:
            break
    return X[keep].astype(np.float32) / 255.0


def quantize_palette(A, C, seed=0):
    px = A.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    cen = px[rng.choice(len(px), C, replace=False)].copy()
    for it in range(20):
        d = ((px[:, None, :] - cen[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for k in range(C):
            m = lab == k
            if m.sum():
                cen[k] = px[m].mean(0)
    d = ((px[:, None, :] - cen[None, :, :]) ** 2).sum(2)
    lab = d.argmin(1)
    # 按亮度排序（让颜色成为有序一维量 -> 红石输出层只需 24 个单元）
    lum = cen @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    remap = np.zeros(C, dtype=int); remap[order] = np.arange(C)
    lab = remap[lab]; cen = cen[order]
    return lab.reshape(len(A), 8, 8), cen


def target(lab, C):
    return (lab.astype(np.float32) / max(C - 1, 1)).reshape(len(lab), -1)


def init_mlp(d, h, out, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, scale, (d, h)), np.zeros(h),
            rng.normal(0, scale, (h, out)), np.zeros(out)]


def forward(P, Z):
    W1, b1, W2, b2 = P
    H = np.maximum(Z @ W1 + b1, 0)
    return H, H @ W2 + b2


def train_ls(P0, Zt, Tt, steps=4000, lr=0.02, bs=128, seed=0):
    W1, b1, W2, b2 = [p.copy() for p in P0]
    ps = [W1, b1, W2, b2]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    rng = np.random.default_rng(seed)
    n = len(Zt); best = (1e9, None)
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Zb, Tb = Zt[idx], Tt[idx]
        H, Y = forward(ps, Zb)
        dY = 2 * (Y - Tb) / bs
        gW2 = H.T @ dY; gb2 = dY.sum(0)
        dH = (dY @ W2.T) * (H > 0)
        gW1 = Zb.T @ dH; gb1 = dH.sum(0)
        for i, (pp, gg) in enumerate(zip(ps, [gW1, gb1, gW2, gb2])):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            _, Yf = forward(ps, Zt)
            mse = float(((Yf - Tt) ** 2).mean())
            if mse < best[0]:
                best = (mse, [p.copy() for p in ps])
    return best


def main():
    print("=" * 98)
    print("8x8 像素画生成器容量实验（本地 MC 材质数据）")
    print("=" * 98)
    A = load_structured(min_colors=6, want=12000)
    print(f"有结构图块（>=6 色）: {len(A)}")
    rng = np.random.default_rng(0); A = A[rng.permutation(len(A))]
    Atr, Ate = A[:8000], A[8000:10000]
    print(f"train {Atr.shape}  test {Ate.shape}")

    out = {}
    for C in (16, 48):
        lab_tr, pal = quantize_palette(Atr, C, seed=1)
        px = Ate.reshape(-1, 3)
        d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(2)
        lab_te = d.argmin(1).reshape(len(Ate), 8, 8)
        Ttr, Tte = target(lab_tr, C), target(lab_te, C)
        mean_img = Ttr.mean(0)
        base = float(((Tte - mean_img) ** 2).mean())
        print(f"\n{'#'*98}\n### C = {C} 色   基准(恒输出均值图) 测试 MSE = {base:.6f}\n{'#'*98}")
        print(f"  {'d':>4} | {'h':>5} | {'参数量':>8} | {'红石单元':>9} | {'测试MSE':>10} | "
              f"{'相对基准':>9} | {'用时':>6}")
        print("  " + "-" * 76)
        rows = []
        for d_z, h in [(2, 8), (4, 16), (4, 32), (8, 32), (8, 64), (16, 64),
                       (16, 128), (32, 128), (32, 256), (64, 256), (64, 512)]:
            t0 = time.time()
            rg = np.random.default_rng(7)
            Ztr = rg.normal(0, 1, (len(Ttr), d_z)).astype(np.float32)
            Zte = rg.normal(0, 1, (len(Tte), d_z)).astype(np.float32)
            best = (1e9, None)
            for sd in (0, 1):
                P0 = init_mlp(d_z, h, 64, seed=sd)
                r = train_ls(P0, Ztr, Ttr, steps=3000, seed=sd)
                if r[0] < best[0]:
                    best = r
            _, Yte = forward(best[1], Zte)
            mse = float(((Yte - Tte) ** 2).mean())
            nparam = d_z * h + h + h * 64 + 64
            cells = h * d_z + 24 * h        # 红石: 隐藏层 h 神经元 x d 张表 + 输出层 64 像素 x 1 表
            print(f"  {d_z:>4} | {h:>5} | {nparam:>8} | {cells:>9} | {mse:>10.6f} | "
                  f"{mse/base:>8.2%} | {time.time()-t0:>5.0f}s", flush=True)
            rows.append({"C": C, "d": d_z, "h": h, "params": nparam, "cells": cells,
                         "mse": mse, "rel": mse / base})
        out[f"C{C}"] = {"baseline": base, "rows": rows}
    json.dump(out, open(os.path.join(RES, "capacity.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: {RES}/capacity.json")


if __name__ == "__main__":
    main()
