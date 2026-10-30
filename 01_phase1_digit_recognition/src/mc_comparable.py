#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
验证: 10 层的 F 可直接比较，且 max(F) 对应识别数字
=====================================================
要求（czm）:
  1. 每层 F 输出后可直接比较
  2. F 最大的层 = 识别到的数字

设计保证: F_k = z_k + 128（各层平移量相同）
  => argmax_k F_k = argmax_k z_k

逐项验证:
  A. 各层 F 是否 = 各层 z + 128（逐样本逐层）
  B. argmax over 10 层 F 是否等于 argmax over 10 层 z
  C. 准确率是否与网络一致
  D. 各层 F 的实测范围（是否同一量纲可比）
"""
import os, json
import numpy as np

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
    W = np.load(os.path.join(RES, 'Wint_model.npz'))['W'].astype(int)
    bias = np.load(os.path.join(RES, 'Wint_model.npz'))['b'].astype(int)
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    Xin = (to15(Xte) > thr).astype(np.int64).reshape(len(Xte), -1)
    lit_all = Xin.astype(bool)
    Z = Xin.astype(np.float64) @ W + bias

    # 各层 F_0（新）与 F
    F0s = []
    F = np.zeros((len(Xin), 10), dtype=np.int64)
    for k in range(10):
        parity = '奇数' if (k + 1) % 2 == 1 else '偶数'
        c_all = (15 + W[:, k]) if parity == '奇数' else (15 - W[:, k])
        Gsum = np.zeros(len(Xin), dtype=np.int64)
        gmaxs = []
        for gi, (r0, nr) in enumerate(GROUPS):
            pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
            cg = c_all[pxs]
            g0 = int(np.maximum(cg, 15).sum())
            gmax = g0 - int(np.minimum(cg, 15).sum())
            gmaxs.append(gmax)
            lit = lit_all[:, pxs]
            out = np.where(lit, 15, cg[None, :]) if parity == '偶数' \
                else np.where(lit, cg[None, :], 15)
            Gsum += g0 - out.sum(1)
        F0_old = sum(gmaxs)
        # 实测 offset 保证各层平移量 = 128
        F_old = F0_old - Gsum
        off = int((F_old - Z[:, k]).mean())
        F0_new = F0_old - off + TARGET
        F[:, k] = F0_new - Gsum
        F0s.append(F0_new)

    print("=" * 96)
    print("验证: 10 层 F 可直接比较，max(F) 对应识别数字")
    print("=" * 96)

    print(f"\n[A] 逐层核对 F == z + 128")
    print(f"  {'层':>3} | {'F_0':>5} | {'F-z 范围':>14} | {'是否恒 =128':>12}")
    print("  " + "-" * 46)
    okA = True
    for k in range(10):
        d = F[:, k] - Z[:, k]
        good = (d.min() == d.max() == TARGET)
        okA &= good
        print(f"  {k+1:>3} | {F0s[k]:>5} | [{d.min():.0f}, {d.max():.0f}] | "
              f"{'是 OK' if good else '否'}")
    print(f"  全部层平移量 = 128: {'是 OK' if okA else '否'}")

    print(f"\n[B] argmax 逐样本对比")
    am_F = F.argmax(1); am_z = Z.argmax(1)
    same = np.array_equal(am_F, am_z)
    print(f"  argmax(F) == argmax(z): {'是 OK' if same else '否'}")
    print(f"  不一致样本数: {int((am_F != am_z).sum())}")

    print(f"\n[C] 准确率")
    az = float((am_z == yte).mean()); aF = float((am_F == yte).mean())
    print(f"  原网络 argmax(z)   {az:.4f}")
    print(f"  10 层 argmax(F)    {aF:.4f}   {'一致 OK' if az == aF else '不一致'}")
    print(f"  -> 识别到的数字 = F 最大的那一层，准确率 {aF:.4f}")

    print(f"\n[D] 各层 F 的实测范围（可比性检查）")
    print(f"  {'层':>3}(数字) | {'F_0':>5} | {'F 实测范围':>14} | {'F 均值':>8} | {'F 标准差':>8}")
    print("  " + "-" * 62)
    for k in range(10):
        print(f"  {k+1:>3}({k}) | {F0s[k]:>5} | [{F[:,k].min():>4}, {F[:,k].max():>4}] | "
              f"{F[:,k].mean():>8.1f} | {F[:,k].std():>8.1f}")

    print(f"\n[E] 每个样本的 max(F) 与次大值之差（决策余量）")
    srt = np.sort(F, 1)
    margin = srt[:, -1] - srt[:, -2]
    print(f"  最小 {margin.min()}  中位 {np.median(margin):.0f}  平均 {margin.mean():.1f}")
    print(f"  margin = 0 的样本数: {int((margin==0).sum())}  （平局）")
    for t in (1, 2, 3, 5):
        print(f"  margin <= {t}: {int((margin<=t).sum())}")

    print(f"\n{'='*96}")
    print("### 结论")
    print(f"  1. 各层 F = z + 128（平移量一致）-> 可直接比较  "
          f"{'OK' if okA else 'FAIL'}")
    print(f"  2. argmax(F) == argmax(z) 逐样本一致  {'OK' if same else 'FAIL'}")
    print(f"  3. F 最大的层 = 识别数字，准确率 {aF:.4f}")
    print(f"  4. 各层 F 范围 [{F.min()}, {F.max()}]，量纲一致（都是 z+128）")

    json.dump({'F_0': F0s, 'target': TARGET,
               'argmax_match': bool(same), 'accuracy': aF,
               'F_range': [int(F.min()), int(F.max())],
               'margin_min': int(margin.min()),
               'margin_median': float(np.median(margin)),
               'margin_ties': int((margin == 0).sum())},
              open(os.path.join(RES, 'mc_comparable_check.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_comparable_check.json")


if __name__ == '__main__':
    main()
