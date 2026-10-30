#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
可视化：MNIST 按原方案缩到 15x15 并二值化后的样子
====================================================
图 1 (900 数字):
  30 行 x 30 列 = 900 个数字
  每 3 行对应一个数字类别（3x30=90 个样本/类），共 10 类 x 3 行 = 30 行
  每 3 行左侧有类别标签

图 2 (各类别放大):
  每个数字挑 10 个，放大显示，便于看清笔画
"""
import os, struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
BASE_THR = 29.09          # 原方案的二值化阈值（训练集 15x15 的均值）


def load_idx(p):
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


def get_font(sz):
    for nm in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(nm, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    tr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    A15 = to15(tr.astype(np.float32))
    B = A15 > BASE_THR
    print(f"训练集 {B.shape}  平均点亮 {B.reshape(len(B),-1).sum(1).mean():.1f}/225")

    # 每类取 90 个（在训练集里顺序取，保证多样）
    picks = {}
    for d in range(10):
        idx = np.flatnonzero(ytr == d)[:90]
        picks[d] = idx
        print(f"  数字 {d}: {len(idx)} 个样本")

    # ---------- 图 1: 900 个数字 ----------
    CELL = 20            # 每个像素 20x20
    PADPIX = 2           # 数字之间留白（像素）
    LBLW = 52            # 左侧标签宽度
    ROWS_PER_DIGIT = 3
    COLS = 30
    DIG = 15             # 每个数字 15x15

    tile = DIG * CELL + PADPIX          # 单个数字占的像素
    Wt = LBLW + COLS * tile + PADPIX
    Ht = 34 + 10 * ROWS_PER_DIGIT * tile + PADPIX
    canvas = np.full((Ht, Wt, 3), 255, dtype=np.uint8)

    for d in range(10):
        for r in range(ROWS_PER_DIGIT):
            for c in range(COLS):
                n = r * COLS + c
                if n >= len(picks[d]):
                    break
                img = B[picks[d][n]]
                y0 = 34 + (d * ROWS_PER_DIGIT + r) * tile
                x0 = LBLW + c * tile
                cell = np.where(img, 0, 255).astype(np.uint8)
                big = np.kron(cell, np.ones((CELL, CELL), dtype=np.uint8))
                canvas[y0:y0 + DIG*CELL, x0:x0 + DIG*CELL] = big[:, :, None]
        # 类别分隔线
        ysep = 34 + (d + 1) * ROWS_PER_DIGIT * tile - PADPIX
        if d < 9:
            canvas[ysep:ysep + 2, :] = (200, 200, 200)

    out = Image.fromarray(canvas)
    dr = ImageDraw.Draw(out)
    f14 = get_font(14)
    f16 = get_font(16)
    dr.text((10, 8), "MNIST 下采样到 15x15 + 二值化（阈值 29.09）  "
                     "900 个样本 = 30x30，每 3 行一个数字类别",
            fill=(0, 0, 0), font=f14)
    for d in range(10):
        yc = 34 + (d * ROWS_PER_DIGIT + 1.5) * tile
        dr.text((6, int(yc) - 10), str(d), fill=(180, 0, 0), font=f16)
    out.save(os.path.join(RES, 'mnist15_binarized_900.png'))
    print(f"已存: results/mnist15_binarized_900.png  ({out.size[0]}x{out.size[1]})")

    # ---------- 图 2: 每类 10 个放大 ----------
    CELL2, PAD2, LBL2 = 16, 4, 40
    PER = 10
    tile2 = 15 * CELL2 + PAD2
    W2 = LBL2 + PER * tile2 + PAD2
    H2 = 30 + 10 * tile2 + PAD2
    c2 = np.full((H2, W2, 3), 255, dtype=np.uint8)
    for d in range(10):
        for c in range(PER):
            n = c * 9                     # 分散取样本
            if n >= len(picks[d]):
                n = c % len(picks[d])
            img = B[picks[d][n]]
            y0 = 30 + d * tile2
            x0 = LBL2 + c * tile2
            big = np.kron(np.where(img, 0, 255).astype(np.uint8),
                          np.ones((CELL2, CELL2), dtype=np.uint8))
            c2[y0:y0+15*CELL2, x0:x0+15*CELL2] = big[:, :, None]
        ysep = 30 + (d + 1) * tile2 - PAD2
        if d < 9:
            c2[ysep:ysep+2, :] = (200, 200, 200)
    o2 = Image.fromarray(c2)
    dr2 = ImageDraw.Draw(o2)
    dr2.text((8, 6), "每个数字 10 个样本（放大 16 倍）",
             fill=(0, 0, 0), font=get_font(14))
    for d in range(10):
        dr2.text((6, 30 + d * tile2 + 100), str(d), fill=(180, 0, 0),
                 font=get_font(18))
    o2.save(os.path.join(RES, 'mnist15_by_digit.png'))
    print(f"已存: results/mnist15_by_digit.png  ({o2.size[0]}x{o2.size[1]})")

    # 文本形式：每类前 20 个的字符画
    with open(os.path.join(RES, 'mnist15_ascii.txt'), 'w', encoding='utf-8') as f:
        for d in range(10):
            f.write(f"\n{'='*72}\n数字 {d}（前 20 个样本）\n{'='*72}\n")
            for n in range(20):
                if n >= len(picks[d]):
                    break
                img = B[picks[d][n]]
                f.write(f"\n样本 {n+1}  (点亮 {int(img.sum())} 格)\n")
                for r in range(15):
                    f.write('  ' + ''.join('#' if img[r, c] else '.'
                                           for c in range(15)) + '\n')
    print(f"已存: results/mnist15_ascii.txt")

    # 统计
    print(f"\n各类别的平均点亮格数（15x15 二值化后）:")
    for d in range(10):
        s = B[picks[d]].reshape(len(picks[d]), -1).sum(1)
        print(f"  数字 {d}: {s.mean():.1f} 格  (范围 {s.min()}~{s.max()})")


if __name__ == '__main__':
    main()
