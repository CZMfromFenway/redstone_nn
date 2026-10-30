#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
整数权重版 225 -> 10 线性分类器：完整分析 + 存表
====================================================
约束: 权重 w ∈ Z ∩ [-882, 882]，输入 x ∈ {0,1}，输出 z 为**精确整数**

分析:
  1. 训练整数权重（QAT，直通估计）
  2. 参数分布（权重、偏置）-> CSV
  3. **全样本空间（2^225 个输入）的输出精确极值**
       z_max = b + sum(max(w,0))      （因为 x 可任取 0/1）
       z_min = b + sum(min(w,0))
     —— 这是精确的，不需要枚举
  4. 判对样本的 top1-top2 差值（margin）最小值与完整分布
"""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
LO, HI = -882, 882


def load_idx(p):
    import struct
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def to15(im):
    n = len(im)
    a = im.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def train_int(Ftr, ytr, Fte, yte, steps=6000, lr=0.05, bs=128, seed=0,
              lo=LO, hi=HI, best_track=True):
    d_in = Ftr.shape[1]; n = len(Ftr)
    rng = np.random.default_rng(seed)
    W = np.zeros((d_in, 10)); b = np.zeros(10)
    M = [np.zeros_like(W), np.zeros_like(b)]; V = [np.zeros_like(W), np.zeros_like(b)]
    best = (0.0, None, None)
    for t in range(1, steps + 1):
        Wq = np.clip(np.round(W), lo, hi); bq = np.clip(np.round(b), lo, hi)
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Z = Fb @ Wq + bq
        mx = Z.max(1, keepdims=True)
        e = np.exp(Z - mx); P = e / e.sum(1, keepdims=True)
        dZ = P.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        g = [Fb.T @ dZ, dZ.sum(0)]
        for i, (pp, gg) in enumerate(zip([W, b], g)):
            M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
            pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            acc = float(((Fte @ Wq + bq).argmax(1) == yte).mean())
            if acc > best[0]:
                best = (acc, Wq.copy(), bq.copy())
    return best


def main():
    print("=" * 92)
    print(f"整数权重 225->10 线性分类器   w ∈ Z ∩ [{LO},{HI}]")
    print("=" * 92)
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B16 = to15(Xte); thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B16 > thr).astype(np.float64).reshape(len(B16), -1)
    d_in = Ftr.shape[1]
    print(f"数据 train {Ftr.shape}  test {Fte.shape}\n")

    acc, Wi, bi = train_int(Ftr, ytr, Fte, yte)
    print(f"整数权重准确率 {acc:.4f}")

    # ---------- 1. 参数分布 ----------
    print(f"\n[1] 整数权重分布（{Wi.size} 个）")
    print(f"  范围 [{Wi.min():.0f}, {Wi.max():.0f}]   均值 {Wi.mean():.3f}   "
          f"标准差 {Wi.std():.3f}   中位数 {np.median(Wi):.0f}")
    u, c = np.unique(Wi, return_counts=True)
    print(f"  取值个数 {len(u)}")
    print(f"  {'值':>5} | {'计数':>6} | {'占比':>7}")
    print("  " + "-" * 24)
    for v, cc in zip(u, c):
        print(f"  {v:>5.0f} | {cc:>6} | {cc/Wi.size:>6.2%}")
    print(f"  非零权重 {(Wi != 0).sum()} / {Wi.size}  ({(Wi!=0).mean():.1%})")
    print(f"  偏置: 范围 [{bi.min():.0f}, {bi.max():.0f}]  "
          f"值 = {[int(x) for x in bi]}")

    # ---------- 2. 全样本空间精确输出范围 ----------
    pos = np.maximum(Wi, 0).sum(0)
    neg = np.minimum(Wi, 0).sum(0)
    zmax = bi + pos
    zmin = bi + neg
    Zte = Fte @ Wi + bi
    print(f"\n[2] 全样本空间（2^225 个输入）的输出范围 —— **精确极值**")
    print(f"  {'通道':>4} | {'理论最小':>9} | {'理论最大':>9} | {'跨度':>7} | "
          f"{'测试集最小':>10} | {'测试集最大':>10} | {'测试集跨度':>9}")
    print("  " + "-" * 78)
    for k in range(10):
        print(f"  {k:>4} | {zmin[k]:>9.0f} | {zmax[k]:>9.0f} | {zmax[k]-zmin[k]:>7.0f} | "
              f"{Zte[:,k].min():>10.0f} | {Zte[:,k].max():>10.0f} | "
              f"{Zte[:,k].max()-Zte[:,k].min():>9.0f}")

    # ---------- 3. margin 分析 ----------
    order = np.argsort(-Zte, axis=1)
    top1 = order[:, 0]; top2 = order[:, 1]
    margin = Zte[np.arange(len(Zte)), top1] - Zte[np.arange(len(Zte)), top2]
    correct = top1 == yte
    mc = margin[correct]
    print(f"\n[3] 判对样本的 top1 - top2 差值（margin）")
    print(f"  判对 {correct.sum()} / {len(yte)}   准确率 {correct.mean():.4f}")
    print(f"  **最小 margin = {mc.min():.0f}**（整数，故最小可能为 1）")
    print(f"  最大 {mc.max():.0f}   均值 {mc.mean():.3f}   中位数 {np.median(mc):.0f}")
    u2, c2 = np.unique(mc, return_counts=True)
    print(f"\n  margin 完整分布（{len(u2)} 个取值）")
    print(f"  {'margin':>7} | {'样本数':>7} | {'占比':>7} | {'累计':>7}")
    print("  " + "-" * 38)
    cum = 0
    for v, cc in zip(u2, c2):
        cum += cc
        if cc > 20 or v <= 5:
            print(f"  {v:>7.0f} | {cc:>7} | {cc/len(mc):>6.2%} | {cum/len(mc):>6.2%}")
    print(f"\n  margin = 1 的样本数: {(mc == 1).sum()}   （最危险的样本）")
    for thr_ in (1, 2, 3, 5, 10):
        print(f"  margin <= {thr_:>2}: {(mc <= thr_).sum():>5} 个 "
              f"({(mc <= thr_).mean():.3%})")

    # ---------- 4. 存表 ----------
    with open(os.path.join(RES, 'Wint_225x10.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pixel_r', 'pixel_c', 'input_index']
                   + [f'w_digit{k}' for k in range(10)])
        for p in range(d_in):
            w.writerow([p // 15, p % 15, p] + [int(Wi[p, k]) for k in range(10)])
    print(f"\n[4] 已写出:")
    print(f"  results/Wint_225x10.csv         全部 {Wi.size} 个整数权重 + 像素坐标")

    with open(os.path.join(RES, 'Wint_matrix_15x15.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for k in range(10):
            w.writerow([f'--- digit {k} (bias={int(bi[k])}) ---'])
            for r in range(15):
                w.writerow([int(Wi[r*15+c, k]) for c in range(15)])
    print(f"  results/Wint_matrix_15x15.csv   每个数字一个 15x15 整数权重矩阵")

    with open(os.path.join(RES, 'Wint_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['digit', 'bias', 'w_pos_sum', 'w_neg_sum',
                    'z_min_exact', 'z_max_exact', 'z_span',
                    'z_min_test', 'z_max_test', 'z_mean_test', 'z_std_test',
                    'w_min', 'w_max', 'w_nnz'])
        for k in range(10):
            w.writerow([k, int(bi[k]), int(pos[k]), int(neg[k]),
                        int(zmin[k]), int(zmax[k]), int(zmax[k]-zmin[k]),
                        int(Zte[:,k].min()), int(Zte[:,k].max()),
                        f'{Zte[:,k].mean():.3f}', f'{Zte[:,k].std():.3f}',
                        int(Wi[:,k].min()), int(Wi[:,k].max()), int((Wi[:,k]!=0).sum())])
    print(f"  results/Wint_summary.csv        每通道统计")

    np.savez(os.path.join(RES, 'Wint_model.npz'), W=Wi.astype(np.int32),
             b=bi.astype(np.int32), acc=acc)
    print(f"  results/Wint_model.npz          整数权重（可直接用）")

    with open(os.path.join(RES, 'Wint_margins.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'true', 'top1', 'top2', 'z_top1', 'z_top2', 'margin', 'correct'])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]), int(top1[i]), int(top2[i]),
                        int(Zte[i, top1[i]]), int(Zte[i, top2[i]]),
                        int(margin[i]), int(correct[i])])
    print(f"  results/Wint_margins.csv        {len(yte)} 个样本的 margin")

    json.dump({'accuracy': acc, 'w_min': int(Wi.min()), 'w_max': int(Wi.max()),
               'w_nnz': int((Wi != 0).sum()),
               'bias': [int(x) for x in bi],
               'z_min_exact': [int(x) for x in zmin],
               'z_max_exact': [int(x) for x in zmax],
               'margin_min': int(mc.min()), 'margin_median': float(np.median(mc)),
               'margin_mean': float(mc.mean()),
               'margin_eq1': int((mc == 1).sum()),
               'margin_le3': int((mc <= 3).sum())},
              open(os.path.join(RES, 'Wint_analysis.json'), 'w'), indent=2)
    print(f"  results/Wint_analysis.json")


if __name__ == '__main__':
    main()
