#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MNIST 15x15 二值 —— K=1 的降体积实验
======================================
两个方向（czm 提议）:
  (1) 核的形状: 学到的 / 全 1 / 只用非负
  (2) 空间降采样: 卷积 stride 与池化块大小

体积记账（按 docs/05）:
  每个 LUT 单元长度 L≈36; 每层体积 V ≈ 4n·h·L    (n=输入线数, h=神经元数)
  -> 单元数 = n·h ;  体积 = 单元数 × 4 × 36 = 单元数 × 144

结构: 15x15 二值 -> 3x3 卷积(stride s1, padding) -> ReLU(0..15)
      -> T 阈值二值化 -> 块内求和(池化 p) -> 致密层 -> 10
"""
import numpy as np, os, struct, json, time

RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'


def load_idx(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))


def to15(imgs):
    n = len(imgs)
    a = imgs.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    out = np.zeros((n, 15, 15), dtype=np.float32)
    out[:, :14, :14] = a
    out[:, 14, :14] = a[:, 13, :]
    out[:, :14, 14] = a[:, :, 13]
    out[:, 14, 14] = a[:, 13, 13]
    return out


Btr = (to15(Xtr) > to15(Xtr).mean()).astype(np.float32)
Bte = (to15(Xte) > to15(Xtr).mean()).astype(np.float32)
rng = np.random.default_rng(0)
i_tr = rng.choice(len(Btr), 8000, replace=False)
i_te = rng.choice(len(Bte), 2000, replace=False)
Btr, ytr_s = Btr[i_tr], ytr[i_tr]
Bte, yte_s = Bte[i_te], yte[i_te]
print(f"数据: 15x15 二值, train {len(Btr)} test {len(Bte)}")


def conv(B, W, stride):
    n = len(B)
    P = np.zeros((n, 17, 17), dtype=np.float32); P[:, 1:16, 1:16] = B
    osz = (15 - 1) // stride + 1
    out = np.zeros((n, osz, osz), dtype=np.float32)
    for dr in range(3):
        for dc in range(3):
            sl = P[:, dr:dr + 15:stride, dc:dc + 15:stride]
            out += W[dr, dc] * sl[:, :osz, :osz]
    return out, P


def reduce_map(R, p):
    """R:(b,m,m) -> (b, g, g) 块内求和，块大小 p，不足则丢弃余数"""
    b, m, _ = R.shape
    g = m // p
    if g < 1:
        return R.reshape(b, 1, -1).sum(2)
    Rc = R[:, :g * p, :g * p]
    return Rc.reshape(b, g, p, g, p).sum(axis=(2, 4))


def feats(R, taus, p1, p2=None):
    """
    T 阈值二值化 -> 两级块内求和
      p1: 第一级块大小; p2: 第二级(可选)
    """
    b = R.shape[0]
    Fs = []
    for ta in taus:
        Bt = (R >= ta).astype(np.float32)
        S = reduce_map(Bt, p1)
        if p2:
            S = reduce_map(S, p2)
        Fs.append(S.reshape(b, -1))
    return np.concatenate(Fs, axis=1)


def run(kernel_mode, stride, taus, p1, p2=None, steps=3000, lr=0.02, bs=64, seed=0):
    """
    训练: 核(可学) + 致密层. kernel_mode: 'learn' | 'ones' | 'nonneg'
    """
    n = len(Btr); rng = np.random.default_rng(seed)
    W = np.ones((3, 3), dtype=np.float64) if kernel_mode == 'ones' else rng.normal(0, 1.0, (3, 3))
    R0, _ = conv(Btr[:8], W, stride)
    nf = feats(R0, taus, p1, p2).shape[1]
    Wd = rng.normal(0, 0.1, (nf, 10)); bd = np.zeros(10)
    ps = [Wd, bd] if kernel_mode == 'ones' else [W, Wd, bd]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = 0.0
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Bb, yb = Btr[idx], ytr_s[idx]
        Rc, Pb = conv(Bb, W, stride)
        R = np.clip(Rc, 0, 15)
        F = feats(R, taus, p1, p2)
        Z = F @ Wd + bd
        Pm = np.exp(Z - Z.max(1, keepdims=True)); Pm /= Pm.sum(1, keepdims=True)
        dZ = Pm.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        gWd = F.T @ dZ; gbd = dZ.sum(0)
        g = [gWd, gbd]
        if kernel_mode != 'ones':
            dF = dZ @ Wd.T
            dR = np.zeros_like(R)
            # 反传: 阈值 STE + 池化均分
            off = 0
            per = dF.shape[1] // len(taus)
            dR = np.zeros_like(R)
            for ti, ta in enumerate(taus):
                dS = dF[:, ti * per:(ti + 1) * per]
                # 展开到 (b,g,g) -> 还原到 R 的尺寸
                b_, m_, _ = R.shape
                g_ = m_ // p1
                dM = dS.reshape(b_, g_, g_) if (p2 is None) else None
                if p2 is not None:
                    raise NotImplementedError
                dM = np.repeat(np.repeat(dM / (p1 * p1), p1, axis=1), p1, axis=2)
                dR[:, :g_ * p1, :g_ * p1] += dM
            dRc = dR * ((Rc > 0) & (Rc < 15))
            gW = np.zeros((3, 3))
            for dr in range(3):
                for dc in range(3):
                    gW[dr, dc] = float((dRc * Pb[:, dr:dr + 15:stride, dc:dc + 15:stride]
                                        [:, :R.shape[1], :R.shape[2]]).sum())
            g = [gW] + g
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            Wc = np.ones((3, 3)) if kernel_mode == 'ones' else np.clip(np.round(W), -6, 6)
            if kernel_mode == 'nonneg':
                Wc = np.clip(np.round(W), 0, 6)
            Rc2, _ = conv(Bte, Wc, stride); R2 = np.clip(Rc2, 0, 15)
            F2 = feats(R2, taus, p1, p2)
            a = float(((F2 @ Wd + bd).argmax(1) == yte_s).mean())
            best = max(best, a)
    # 成本
    m = (15 - 1) // stride + 1
    g1 = m // p1
    if p2:
        g2 = g1 // p2
    else:
        g2 = g1
    nfeat = len(taus) * g2 * g2
    conv_cells = m * m * 9
    thr_cmp = len(taus) * m * m
    dense_cells = nfeat * 10
    cells = conv_cells + dense_cells
    return best, cells, nfeat, m, W


taus4 = [1.0, 4.0, 8.0, 12.0]
print("\n" + "=" * 104)
print("扫: 核形状 × 卷积 stride × 池化块   (T=4 固定)")
print("=" * 104)
print(f"{'核':>8} | {'s':>2} | {'p':>4} | {'卷积图':>7} | {'特征数':>6} | {'卷积单元':>8} | "
      f"{'致密单元':>8} | {'总单元':>7} | {'体积':>9} | {'准确率':>7}")
print("-" * 104)
rows = []
for mode in ('learn', 'ones', 'nonneg'):
    for stride in (1, 2):
        for p1 in (3, 5, 1):
            try:
                t0 = time.time()
                acc, cells, nf, m, W = run(mode, stride, taus4, p1, steps=3000)
                vol = cells * 144
                conv_cells = m * m * 9
                print(f"{mode:>8} | {stride:>2} | {p1:>4} | {m:>3}x{m:<3} | {nf:>6} | "
                      f"{conv_cells:>8} | {nf*10:>8} | {cells:>7} | {vol:>9} | {acc:>7.4f}",
                      flush=True)
                rows.append({"mode": mode, "stride": stride, "p": p1, "m": m,
                             "nfeat": nf, "conv_cells": conv_cells,
                             "dense_cells": nf * 10, "cells": cells,
                             "vol": vol, "acc": acc})
            except Exception as e:
                print(f"{mode:>8} | {stride:>2} | {p1:>4} | 失败: {type(e).__name__} {e}")

json.dump(rows, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "results", "mnist_sweep.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/mnist_sweep.json")
