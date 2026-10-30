#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
225 -> 10 (argmax) 线性分类器的完整参数与输出分析
====================================================
输出:
  1. weights.csv      —— 全部 2250 个权重（按 像素位置 -> 数字 组织）
  2. biases.csv       —— 10 个偏置
  3. summary.csv      —— 每个输出通道的统计
  4. margins.csv      —— 每个测试样本的 top1/top2 差值（判对时）
  5. analysis.txt     —— 汇总报告

关键分析:
  * 全样本空间的输出范围: 对每个通道，
      max = bias + sum(max(w,0))      下界
      min = bias + sum(min(w,0))      上界
    （因为输入是 0/1 二值，这是**精确**极值，不需要枚举 2^225）
  * 测试集上的实际分布
  * 判对样本的 margin 最小值及其分布
"""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


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


def main():
    print("=" * 90)
    print("225 -> 10 (argmax) 线性分类器：参数与输出分析")
    print("=" * 90)
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B16 = to15(Xte); thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B16 > thr).astype(np.float64).reshape(len(B16), -1)
    d_in = Ftr.shape[1]; n = len(Ftr)
    print(f"数据 train {Ftr.shape}  test {Fte.shape}   前景占比 {Ftr.mean():.4f}")

    # ---------- 训练 ----------
    rng = np.random.default_rng(0)
    W = np.zeros((d_in, 10)); b = np.zeros(10)
    M = [np.zeros_like(W), np.zeros_like(b)]; V = [np.zeros_like(W), np.zeros_like(b)]
    steps, lr, bs = 4000, 0.05, 128
    best = 0.0
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Z = Fb @ W + b
        mx = Z.max(1, keepdims=True)
        e = np.exp(Z - mx); P = e / e.sum(1, keepdims=True)
        dZ = P.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        g = [Fb.T @ dZ, dZ.sum(0)]
        for i, (pp, gg) in enumerate(zip([W, b], g)):
            M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
            pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 2000 == 0 or t == steps:
            best = max(best, float(((Fte @ W + b).argmax(1) == yte).mean()))
    print(f"训练完成: 测试准确率 {best:.4f}")

    # ---------- 1. 权重分布 ----------
    print(f"\n[1] 权重分布（{W.size} 个）")
    print(f"  范围 [{W.min():.4f}, {W.max():.4f}]   均值 {W.mean():.5f}   "
          f"标准差 {W.std():.4f}")
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print(f"  分位数: " + "  ".join(f"P{q}={np.percentile(W,q):.4f}" for q in qs))
    print(f"  正权重 {(W>0).sum()}  负权重 {(W<0).sum()}  零 {np.isclose(W,0).sum()}")

    # ---------- 2. 全样本空间的**精确**输出范围 ----------
    # 输入是 0/1 二值 -> 每个通道的极值可精确计算（不需枚举 2^225）
    pos_sum = np.maximum(W, 0).sum(0)      # 该通道全取 +1 时的最大增量
    neg_sum = np.minimum(W, 0).sum(0)      # 全取 -0（即输入全 0）时的最小增量
    z_max = b + pos_sum                     # 精确上界
    z_min = b + neg_sum                     # 精确下界
    print(f"\n[2] 全样本空间（2^225 个输入）的输出范围 —— 精确极值")
    print(f"  {'通道':>4} | {'理论最小':>10} | {'理论最大':>10} | {'跨度':>9} | "
          f"{'测试集最小':>10} | {'测试集最大':>10}")
    print("  " + "-" * 74)
    Zte = Fte @ W + b
    for k in range(10):
        print(f"  {k:>4} | {z_min[k]:>10.3f} | {z_max[k]:>10.3f} | "
              f"{z_max[k]-z_min[k]:>9.3f} | {Zte[:,k].min():>10.3f} | {Zte[:,k].max():>10.3f}")

    # ---------- 3. margin 分析 ----------
    order = np.argsort(-Zte, axis=1)
    top1 = order[:, 0]; top2 = order[:, 1]
    margin = Zte[np.arange(len(Zte)), top1] - Zte[np.arange(len(Zte)), top2]
    correct = top1 == yte
    mc = margin[correct]
    print(f"\n[3] 判对样本的 top1 - top2 差值（margin）")
    print(f"  判对样本数 {correct.sum()} / {len(yte)}   准确率 {correct.mean():.4f}")
    print(f"  **最小 margin = {mc.min():.6f}**   （最大值 {mc.max():.3f}）")
    print(f"  均值 {mc.mean():.4f}   标准差 {mc.std():.4f}")
    mq = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    print(f"  分位数: " + "  ".join(f"P{q}={np.percentile(mc,q):.4f}" for q in mq))
    print(f"\n  最接近的 10 个 margin: {np.sort(mc)[:10].round(6).tolist()}")
    print(f"  margin < 0.1 的样本数: {(mc < 0.1).sum()}")
    print(f"  margin < 0.01 的样本数: {(mc < 0.01).sum()}")
    print(f"  margin < 0.001 的样本数: {(mc < 0.001).sum()}")

    # 判错样本的 margin（应为负）
    mw = margin[~correct]
    print(f"\n  判错样本 {len(mw)} 个，margin 范围 [{mw.min():.3f}, {mw.max():.3f}]")

    # ---------- 4. 写出 CSV ----------
    with open(os.path.join(RES, 'W225x10.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pixel_r', 'pixel_c', 'input_index'] + [f'w_to_digit{k}' for k in range(10)])
        for p in range(d_in):
            w.writerow([p // 15, p % 15, p] + [f'{W[p,k]:.6f}' for k in range(10)])
    print(f"\n[4] 已写出 results/W225x10.csv  （{d_in} 行 x 10 列权重 + 位置列）")

    with open(os.path.join(RES, 'bias10.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['digit', 'bias'])
        for k in range(10):
            w.writerow([k, f'{b[k]:.6f}'])

    with open(os.path.join(RES, 'channel_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['digit', 'bias', 'w_sum_positive', 'w_sum_negative',
                    'z_min_exact', 'z_max_exact', 'z_span',
                    'z_min_testset', 'z_max_testset',
                    'z_mean_testset', 'z_std_testset',
                    'w_min', 'w_max', 'w_mean', 'w_abs_sum'])
        for k in range(10):
            w.writerow([k, f'{b[k]:.6f}', f'{pos_sum[k]:.6f}', f'{neg_sum[k]:.6f}',
                        f'{z_min[k]:.6f}', f'{z_max[k]:.6f}', f'{z_max[k]-z_min[k]:.6f}',
                        f'{Zte[:,k].min():.6f}', f'{Zte[:,k].max():.6f}',
                        f'{Zte[:,k].mean():.6f}', f'{Zte[:,k].std():.6f}',
                        f'{W[:,k].min():.6f}', f'{W[:,k].max():.6f}',
                        f'{W[:,k].mean():.6f}', f'{np.abs(W[:,k]).sum():.6f}'])
    print(f"    results/bias10.csv   results/channel_summary.csv")

    with open(os.path.join(RES, 'margins_testset.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'true_label', 'top1', 'top2', 'z_top1', 'z_top2',
                    'margin', 'correct'])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]), int(top1[i]), int(top2[i]),
                        f'{Zte[i,top1[i]]:.6f}', f'{Zte[i,top2[i]]:.6f}',
                        f'{margin[i]:.6f}', int(correct[i])])
    print(f"    results/margins_testset.csv  （{len(yte)} 行）")

    # 权重按位置的可视化数据
    with open(os.path.join(RES, 'W_matrix_15x15.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for k in range(10):
            w.writerow([f'--- digit {k} ---'])
            for r in range(15):
                w.writerow([f'{W[r*15+c,k]:.4f}' for c in range(15)])
    print(f"    results/W_matrix_15x15.csv  （每个数字一个 15x15 权重图）")

    summary = {
        'accuracy': best, 'n_weights': int(W.size),
        'w_min': float(W.min()), 'w_max': float(W.max()),
        'w_mean': float(W.mean()), 'w_std': float(W.std()),
        'z_min_exact': z_min.tolist(), 'z_max_exact': z_max.tolist(),
        'margin_min_correct': float(mc.min()),
        'margin_mean_correct': float(mc.mean()),
        'margin_median_correct': float(np.median(mc)),
        'margin_p1': float(np.percentile(mc, 1)),
        'margin_p5': float(np.percentile(mc, 5)),
    }
    json.dump(summary, open(os.path.join(RES, 'W225x10_analysis.json'), 'w'), indent=2)
    print(f"    results/W225x10_analysis.json")


if __name__ == '__main__':
    main()
