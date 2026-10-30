#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
笔画加粗对准确率的影响（诊断"总是识别成 3/5"）
==================================================
做法:
  1. 对 MNIST 测试集做形态学膨胀（笔画变粗 1~4 格）
  2. 测准确率
  3. 统计各层 F 的均值随粗细的变化 -> 看哪一层"占便宜"
  4. 统计误判分布（被判成什么）
  5. 模拟"在 15x15 上直接用粗笔刷画"
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
TARGET = 128
DISK = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
        if dy*dy + dx*dx <= 4]


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


def dilate_batch(B, times):
    """B: (n,15,15) bool -> 膨胀 times 次"""
    out = B.copy()
    for _ in range(times):
        acc = out.copy()
        for dy, dx in DISK:
            sh = np.roll(np.roll(out, dy, axis=1), dx, axis=2)
            acc |= sh
        out = acc
    return out


def main():
    zz = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = zz['W'].astype(int); bias = zz['b'].astype(int)

    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    B0 = (to15(Xte) > thr)

    print("=" * 100)
    print("笔画加粗对准确率的影响")
    print("=" * 100)

    print(f"\n  {'加粗':>4} | {'平均点亮':>8} | {'准确率':>8} | "
          f"{'最常误判为':>12} | 各层 F 均值 (0..9)")
    print("  " + "-" * 96)
    res = {}
    for it in range(0, 5):
        B = dilate_batch(B0, it)
        X = B.reshape(len(B), -1).astype(np.int64)
        Z = X @ W + bias
        F = Z + TARGET
        pred = F.argmax(1)
        acc = float((pred == yte).mean())
        lit = X.sum(1).mean()
        # 误判分布
        wrong = pred[yte != pred]
        if len(wrong):
            vals, cnt = np.unique(wrong, return_counts=True)
            top = vals[np.argmax(cnt)]
            tops = f"{top} ({cnt.max()/len(wrong)*100:.0f}%)"
        else:
            tops = "-"
        fm = F.mean(0)
        print(f"  {it:>4} | {lit:>8.1f} | {acc:>8.4f} | {tops:>12} | " +
              " ".join(f"{v:>5.1f}" for v in fm))
        res[it] = {'acc': acc, 'lit': float(lit), 'Fmean': fm.tolist(),
                   'pred_dist': {int(a): int(b) for a, b in
                                 zip(*np.unique(pred, return_counts=True))}}

    print(f"\n{'='*100}")
    print("各层 F 均值随加粗的变化（起点 = 未加粗）")
    print("=" * 100)
    print(f"  {'层(数字)':>8} | " + " | ".join(f"加粗{i}" for i in range(5))
          + " |   Δ(加粗4 - 原)")
    print("  " + "-" * 74)
    base = np.array(res[0]['Fmean'])
    for k in range(10):
        row = [res[i]['Fmean'][k] for i in range(5)]
        print(f"  {k+1:>4}({k})  | " + " | ".join(f"{v:>6.1f}" for v in row) +
              f" |   {row[4]-row[0]:>+8.1f}")

    print(f"\n[解读] Δ 最小（最不吃亏）的层 = 笔画变粗时最容易胜出")
    deltas = [(res[4]['Fmean'][k] - res[0]['Fmean'][k], k) for k in range(10)]
    deltas.sort(reverse=True)
    for d, k in deltas:
        print(f"    Δ = {d:>+7.1f}  层{k+1}（数字{k}）  "
              f"{'<- 最占便宜' if d == deltas[0][0] else ''}")

    json.dump({str(k): v for k, v in res.items()},
              open(os.path.join(RES, 'stroke_boldness.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/stroke_boldness.json")


if __name__ == '__main__':
    main()
