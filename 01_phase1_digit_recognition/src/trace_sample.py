#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
样本 #1（数字 0，测试集索引 3）的逐层 F 期望值
=================================================
同时给出第 1 层（数字 0）的 G_1..G_8 中间量，便于逐级核对

args:
  python trace_sample.py [index]      默认 3
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
TARGET = 128
ORDER = [13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0]


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
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    W = np.load(os.path.join(RES, 'Wint_model.npz'))['W'].astype(int)
    bias = np.load(os.path.join(RES, 'Wint_model.npz'))['b'].astype(int)
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    B = (to15(Xte) > thr)
    x = B[idx].reshape(-1).astype(bool)
    y = int(yte[idx])

    print("=" * 100)
    print(f"样本 #{idx}  真实数字 {y}")
    print("=" * 100)
    print("\n[二值像素图]（# = 点亮）")
    for r in range(15):
        print('  ' + ''.join('#' if B[idx, r, c] else '.' for c in range(15)))
    print(f"\n  点亮像素数 {int(x.sum())} / 225")

    # 逐层
    print(f"\n[逐层期望输出] F = z + 128（10 层可比）")
    print(f"  {'层':>3}(数字) | {'y':>5} | {'奇偶':>4} | {'ΣG_i':>6} | {'F_0':>5} | "
          f"{'F':>5} | {'z':>5}")
    print("  " + "-" * 62)
    Fs = []
    for k in range(10):
        ln = k + 1
        parity = '偶数' if ln % 2 == 0 else '奇数'
        yy = -(1 + 2 * (10 - ln))
        c_all = (15 - W[:, k]) if parity == '偶数' else (15 + W[:, k])
        Gs = []
        Gsum = 0
        for (r0, nr) in GROUPS:
            pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
            cg = c_all[pxs]
            g0 = int(np.maximum(cg, 15).sum())
            L = x[pxs]
            out = np.where(L, 15, cg) if parity == '偶数' else np.where(L, cg, 15)
            g = g0 - int(out.sum())
            Gs.append(g); Gsum += g
        z = int(x.astype(np.int64) @ W[:, k] + bias[k])
        off = int(((sum(gs.max() for gs in [np.maximum(c_all[pxs], 15) for _ in [0]]) if False else 0)))
        # 直接用已存参数: F_0 = z + 128 + Gsum  (因为 F = F_0 - Gsum = z + 128)
        F0 = z + TARGET + Gsum
        F = F0 - Gsum
        Fs.append(F)
        print(f"  {ln:>3}({k}) | {yy:>5} | {parity:>4} | {Gsum:>6} | {F0:>5} | "
              f"{F:>5} | {z:>5}")
    Fs = np.array(Fs)
    print(f"\n  argmax(F) = 第 {Fs.argmax()+1} 层（数字 {Fs.argmax()}） F={Fs.max()}")
    print(f"  真实数字 {y} -> {'正确' if Fs.argmax() == y else '错误'}")
    srt = np.sort(Fs)
    print(f"  margin = {srt[-1]} - {srt[-2]} = {srt[-1]-srt[-2]}")

    # 第 1 层（数字 0）明细
    print(f"\n{'='*100}")
    print(f"第 1 层（数字 0，y=-19，奇数层）逐支线明细")
    print("=" * 100)
    k = 0; parity = '奇数'
    c_all = 15 + W[:, k]
    print(f"  {'支线':>5} | {'像素行':>8} | {'n':>3} | {'G_i0':>5} | {'Σ输出':>7} | "
          f"{'G_i':>5}")
    print("  " + "-" * 56)
    Gsum = 0
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
        cg = c_all[pxs]
        g0 = int(np.maximum(cg, 15).sum())
        L = x[pxs]
        out = np.where(L, cg, 15)          # 奇数层: 点亮输出 c, 不亮输出 15
        s = int(out.sum())
        g = g0 - s
        Gsum += g
        print(f"  G_{gi+1:<3} | {r0:>3}-{r0+nr-1:<3} | {len(pxs):>3} | {g0:>5} | "
              f"{s:>7} | {g:>5}")
    z0 = int(x.astype(np.int64) @ W[:, 0] + bias[0])
    print(f"\n  ΣG_i = {Gsum}   z = {z0}   F_0 = {z0 + TARGET + Gsum}")
    print(f"  F = F_0 - ΣG_i = {z0 + TARGET}")
    print(f"  容器信号范围: [{int(c_all.min())}, {int(c_all.max())}]")

    # 目标层（真实数字 y）的明细
    print(f"\n{'='*100}")
    print(f"第 {y+1} 层（数字 {y}，y={-(1+2*(10-(y+1)))}, "
          f"{'偶数' if (y+1)%2==0 else '奇数'}层）逐支线明细")
    print("=" * 100)
    par = '偶数' if (y + 1) % 2 == 0 else '奇数'
    c_all = (15 - W[:, y]) if par == '偶数' else (15 + W[:, y])
    print(f"  {'支线':>5} | {'G_i0':>5} | {'Σ输出':>7} | {'G_i':>5}")
    print("  " + "-" * 36)
    Gsum = 0
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
        cg = c_all[pxs]
        g0 = int(np.maximum(cg, 15).sum())
        L = x[pxs]
        out = np.where(L, 15, cg) if par == '偶数' else np.where(L, cg, 15)
        s = int(out.sum()); g = g0 - s; Gsum += g
        print(f"  G_{gi+1:<3} | {g0:>5} | {s:>7} | {g:>5}")
    zz = int(x.astype(np.int64) @ W[:, y] + bias[y])
    print(f"\n  ΣG_i = {Gsum}   z = {zz}   F_0 = {zz + TARGET + Gsum}   "
          f"F = {zz + TARGET}")


if __name__ == '__main__':
    main()
