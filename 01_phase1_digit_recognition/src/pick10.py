#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
挑选 10 个识别正确的样本: 展示二值像素图 + 每层 F 值
=======================================================
输出:
  results/pick10_digits.png   10 个样本的 15x15 二值图（放大）
  results/pick10_fvalues.csv  每个样本的 10 层 F 值
控制台: 打印每个样本的像素图（文字）与 F 表
"""
import os, json, csv
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
TARGET = 128


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
    zz = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = zz['W'].astype(int); bias = zz['b'].astype(int)
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte_raw = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A15 = to15(Xte_raw)
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    B = (A15 > thr)
    Xin = B.astype(np.int64).reshape(len(B), -1)
    lit_all = (B.reshape(len(B), -1) > 0)
    Z = Xin.astype(np.float64) @ W + bias

    # 各层 F
    F = np.zeros((len(Xin), 10), dtype=np.int64)
    F0s = []
    for k in range(10):
        parity = '奇数' if (k + 1) % 2 == 1 else '偶数'
        c_all = (15 + W[:, k]) if parity == '奇数' else (15 - W[:, k])
        Gsum = np.zeros(len(Xin), dtype=np.int64)
        gmaxs = []
        for gi, (r0, nr) in enumerate(GROUPS):
            pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
            cg = c_all[pxs]
            g0 = int(np.maximum(cg, 15).sum())
            gmaxs.append(g0 - int(np.minimum(cg, 15).sum()))
            lit = lit_all[:, pxs]
            out = np.where(lit, 15, cg[None, :]) if parity == '偶数' \
                else np.where(lit, cg[None, :], 15)
            Gsum += g0 - out.sum(1)
        F0_old = sum(gmaxs)
        off = int(((F0_old - Gsum) - Z[:, k]).mean())
        F0 = F0_old - off + TARGET
        F0s.append(F0)
        F[:, k] = F0 - Gsum

    correct = (F.argmax(1) == yte)
    print(f"识别正确样本: {int(correct.sum())} / {len(yte)}  ({correct.mean():.4f})")

    # 选 10 个: 覆盖不同数字，且 margin 适中（避免平局）
    idxs = []
    srt = np.sort(F, 1)
    margin = srt[:, -1] - srt[:, -2]
    for d in range(10):
        cand = np.flatnonzero(correct & (yte == d) & (margin >= 3))
        if len(cand) == 0:
            cand = np.flatnonzero(correct & (yte == d))
        idxs.append(int(cand[0]))
    idxs = np.array(idxs)
    print(f"选中样本索引: {idxs.tolist()}")
    print(f"对应真实数字: {yte[idxs].tolist()}")
    print(f"对应预测数字: {F[idxs].argmax(1).tolist()}")
    print(f"决策余量: {margin[idxs].tolist()}")

    # 文字打印每个样本
    for n, i in enumerate(idxs):
        print(f"\n{'='*62}")
        print(f"样本 {n+1}: 测试集索引 {i}, 真实数字 {yte[i]}, "
              f"识别为 {F[i].argmax()} (margin {margin[i]})")
        print(f"{'='*62}")
        for r in range(15):
            line = ''.join('#' if B[i, r, c] else '.' for c in range(15))
            print('  ' + line)
        print(f"  {'层':>3}(数字) | " + " | ".join(f"F{k}" for k in range(10)))
        print("  " + "-" * 76)
        print("       F  | " + " | ".join(f"{F[i,k]:>3}" for k in range(10)))
        mx = F[i].argmax()
        print(f"  -> F 最大的是第 {mx+1} 层（数字 {mx}），F={F[i,mx]}，"
              f"次大 F={sorted(F[i])[-2]}")

    # 拼图
    SC = 16
    cols = 5
    rows_ = 2
    canvas = np.full((rows_ * 15, cols * 15, 3), 255, dtype=np.uint8)
    for n, i in enumerate(idxs):
        r_, c_ = divmod(n, cols)
        tile = np.where(B[i], 0, 255).astype(np.uint8)
        canvas[r_*15:(r_+1)*15, c_*15:(c_+1)*15] = tile[:, :, None]
    big = np.kron(canvas, np.ones((SC, SC, 1), dtype=np.uint8))
    for r_ in range(rows_ + 1):
        big[r_*15*SC:r_*15*SC+3, :] = 200
    for c_ in range(cols + 1):
        big[:, c_*15*SC:c_*15*SC+3] = 200
    Image.fromarray(big).save(os.path.join(RES, 'pick10_digits.png'))
    print(f"\n已保存图片: results/pick10_digits.png  "
          f"({big.shape[1]}x{big.shape[0]}, 每格 15x15 放大 {SC} 倍)")

    # CSV
    with open(os.path.join(RES, 'pick10_fvalues.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'test_index', 'true_digit', 'predicted', 'margin']
                   + [f'F_layer{k+1}_digit{k}' for k in range(10)])
        for n, i in enumerate(idxs):
            w.writerow([n + 1, i, int(yte[i]), int(F[i].argmax()), int(margin[i])]
                       + [int(F[i, k]) for k in range(10)])
    print(f"已保存数据: results/pick10_fvalues.csv")

    # 也保存像素图（文本格式便于查看）
    with open(os.path.join(RES, 'pick10_pixels.txt'), 'w', encoding='utf-8') as f:
        for n, i in enumerate(idxs):
            f.write(f"sample {n+1}  index {i}  true {yte[i]}  "
                    f"pred {F[i].argmax()}\n")
            for r in range(15):
                f.write(''.join('#' if B[i, r, c] else '.' for c in range(15)) + '\n')
            f.write('\n')
    print(f"已保存像素: results/pick10_pixels.txt")

    json.dump({'indices': idxs.tolist(), 'F_0': F0s,
               'true': yte[idxs].tolist(),
               'pred': F[idxs].argmax(1).tolist(),
               'F': F[idxs].tolist()},
              open(os.path.join(RES, 'pick10.json'), 'w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
