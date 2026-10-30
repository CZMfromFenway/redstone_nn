#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实 8x8 头像的生成器容量下界（1605 张皮肤脸）
================================================
数据: 从 110 万皮肤数据集抽取的 1605 张，取脸部 (8,8)-(16,16) = 8x8
调色板: MC 16 色混凝土（czm 色环顺序）/ MC 48 色（混凝土+羊毛+陶瓦）
        另附 k-means 自由调色板作对照（看"MC 色系"这个约束的代价）

方法: 自编码器瓶颈实验（表示容量下界）
  关键指标: 相对 MSE，且必须与"量化不可约误差"比较
"""
import os, json, numpy as np, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")


def mlp_kaiming(dims, seed=0):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        std = np.sqrt(2.0 / dims[i])
        P.append(rng.normal(0, std, (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
    P[-1][:] = 0.0
    return P


def fwd(P, X):
    acts = [X]; h = X; L = len(P) // 2
    for i in range(L):
        h = h @ P[2 * i] + P[2 * i + 1]
        if i < L - 1:
            h = np.maximum(h, 0)
        acts.append(h)
    return acts


def train(P0, Xtr, Xte, steps=6000, lr=0.01, bs=64, seed=0):
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


def quant_to_palette(F, pal):
    flat = F.reshape(-1, 3)
    lab = ((flat[:, None] - np.asarray(pal)[None]) ** 2).sum(2).argmin(1)
    rec = np.asarray(pal)[lab].reshape(F.shape)
    qerr = float(((rec - F) ** 2).mean())
    return lab.reshape(len(F), 64), qerr


def kmeans_pal(F, C, seed=0):
    px = F.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    cen = px[rng.choice(len(px), C, replace=False)].copy()
    for _ in range(30):
        lab = ((px[:, None] - cen[None]) ** 2).sum(2).argmin(1)
        for k in range(C):
            m = lab == k
            if m.sum():
                cen[k] = px[m].mean(0)
    lab = ((px[:, None] - cen[None]) ** 2).sum(2).argmin(1)
    rec = cen[lab].reshape(F.shape)
    return cen, lab.reshape(len(F), 64), float(((rec - F) ** 2).mean())


def main():
    print("=" * 100)
    print("真实 8x8 头像（1605 张皮肤脸）生成器容量下界")
    print("=" * 100)
    F = np.load(os.path.join(DATA, 'faces8_rgb.npy')).astype(np.float32)
    print(f"头像数据: {F.shape}")
    rng = np.random.default_rng(0); F = F[rng.permutation(len(F))]
    Atr, Ate = F[:1200], F[1200:]
    print(f"train {Atr.shape}  test {Ate.shape}")

    pals = json.load(open(os.path.join(DATA, 'palettes.json')))
    schemes = [("MC16 混凝土(色环)", pals['pal16']),
               ("MC48 混凝土+羊毛+陶瓦", pals['pal48'])]

    out = {}
    for name, pal in schemes:
        lab_tr, qerr = quant_to_palette(Atr, pal)
        lab_te = quant_to_palette(Ate, pal)[0]
        C = len(pal)
        Ttr = (lab_tr / (C - 1)).astype(np.float32)
        Tte = (lab_te / (C - 1)).astype(np.float32)
        base = float(((Tte - Ttr.mean(0)) ** 2).mean())
        print(f"\n{'#'*100}")
        print(f"### {name}  ({C} 色)")
        print(f"###   量化不可约 MSE = {qerr:.5f} (RMS {np.sqrt(qerr):.1f}/255)"
              f"   基准(均值图) MSE = {base:.5f}")
        print(f"{'#'*100}")
        print(f"  {'d':>4} | {'h':>5} | {'参数':>7} | {'红石单元':>8} | {'测试MSE':>9} | "
              f"{'相对基准':>8} | {'vs量化下限':>10} | 状态")
        print("  " + "-" * 90)
        rows = []
        for d_z, h in [(2, 16), (4, 32), (8, 32), (8, 64), (16, 64), (16, 128), (32, 128)]:
            t0 = time.time()
            dims = [64, h, d_z, h, 64]
            P0 = mlp_kaiming(dims, seed=0)
            r = train(P0, Ttr, Tte, steps=6000, lr=0.01, seed=0)
            tr = float(((fwd(r[1], Ttr[:400])[-1] - Ttr[:400]) ** 2).mean())
            te = r[0]
            cells = 64 * h + h * d_z + d_z * h + h * 64
            dead = abs(tr - base) < 1e-6
            print(f"  {d_z:>4} | {h:>5} | {sum(p.size for p in P0):>7} | {cells:>8} | "
                  f"{te:>9.6f} | {te/base:>7.2%} | {te/max(qerr,1e-9):>9.2f}x | "
                  f"{'死亡' if dead else '学习'}  ({time.time()-t0:.0f}s)", flush=True)
            rows.append({"scheme": name, "C": C, "d": d_z, "h": h, "cells": cells,
                         "mse": te, "rel": te / base, "vs_qerr": te / max(qerr, 1e-9),
                         "dead": bool(dead)})
        out[name] = {"C": C, "quant_err": qerr, "baseline": base, "rows": rows}

    # 自由调色板对照
    print(f"\n{'#'*100}\n### 对照: k-means 自由 16 / 48 色（不受 MC 色系约束）\n{'#'*100}")
    for C in (16, 48):
        cen, lab_tr, qerr = kmeans_pal(Atr, C, 1)
        lab_te = ((Ate.reshape(-1, 3)[:, None] - cen[None]) ** 2).sum(2).argmin(1).reshape(len(Ate), 64)
        Ttr = (lab_tr / (C - 1)).astype(np.float32); Tte = (lab_te / (C - 1)).astype(np.float32)
        print(f"  C={C}: 自由调色板量化误差 MSE={qerr:.5f} (RMS {np.sqrt(qerr):.1f})")
        out[f"自由{C}色"] = {"C": C, "quant_err": qerr}
    print(f"\n  -> 对比'MC 色系'的量化误差，即可看出色系约束的代价")

    json.dump(out, open(os.path.join(RES, "capacity_avatar.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: {RES}/capacity_avatar.json")


if __name__ == "__main__":
    main()
