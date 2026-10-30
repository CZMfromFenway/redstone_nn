#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
10 层权重热图：每个数字在 15x15 屏幕上的权重分布
====================================================
输出:
  results/heatmap_all.png        2x5 拼图（每个数字一张）
  results/heatmap_digit<K>.png   单张（放大）
  results/weights_10x225.csv     原始权重矩阵

配色: 蓝=负权重, 白=0, 红=正权重（对称归一化到全局最大绝对值）
每张图上标注: 数字、bias、W正和、W负和
"""
import os, json, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')

# 配色（发散型）
NEG = np.array([30, 90, 200], dtype=float)      # 蓝
MID = np.array([255, 255, 255], dtype=float)    # 白
POS = np.array([200, 40, 40], dtype=float)      # 红


def colorize(w, vmax):
    """w: (15,15) -> (15,15,3) uint8"""
    t = np.clip(w / vmax, -1, 1)[:, :, None]        # (15,15,1)
    out = np.where(t < 0, MID + (NEG - MID) * (-t), MID + (POS - MID) * t)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    W = np.load(os.path.join(RES, 'Wint_model.npz'))['W'].astype(int)
    bias = np.load(os.path.join(RES, 'Wint_model.npz'))['b'].astype(int)
    vmax = int(np.abs(W).max())
    print(f"权重范围 [{W.min()}, {W.max()}]  全局绝对最大 {vmax}")

    SC = 22           # 每像素放大倍数
    PAD = 2           # 格线
    LABEL = 34        # 顶部文字区高度

    def render(k):
        """返回单张图 (h,w,3)"""
        img = np.full((15, 15, 3), 255, dtype=np.uint8)
        img = colorize(W[:, k].reshape(15, 15), vmax)
        big = np.kron(img, np.ones((SC, SC, 1), dtype=np.uint8))
        # 画格线
        for i in range(16):
            big[i * SC - 1:i * SC + 1, :] = 120
            big[:, i * SC - 1:i * SC + 1] = 120
        return big

    # 单张（带文字标签）
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_s = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        font_s = font

    for k in range(10):
        tile = render(k)
        H, Wd = tile.shape[:2]
        canvas = Image.new('RGB', (Wd, H + LABEL), (255, 255, 255))
        canvas.paste(Image.fromarray(tile), (0, LABEL))
        d = ImageDraw.Draw(canvas)
        wpos = int(W[W[:, k] > 0, k].sum())
        wneg = int(W[W[:, k] < 0, k].sum())
        d.text((6, 6), f"digit {k}   bias={bias[k]}   "
                       f"W+={wpos}  W-={wneg}",
               fill=(0, 0, 0), font=font_s)
        canvas.save(os.path.join(RES, f'heatmap_digit{k}.png'))
        print(f"  heatmap_digit{k}.png   W+={wpos} W-={wneg}")

    # 2x5 拼图
    tiles = [render(k) for k in range(10)]
    th, tw = tiles[0].shape[:2]
    gap = 14
    cols, rows = 5, 2
    Wt = cols * tw + (cols + 1) * gap
    Ht = rows * (th + LABEL) + (rows + 1) * gap
    big = Image.new('RGB', (Wt, Ht), (245, 245, 245))
    for k in range(10):
        r, c = divmod(k, cols)
        x = gap + c * (tw + gap)
        y = gap + r * (th + LABEL + gap)
        big.paste(Image.fromarray(tiles[k]), (x, y + LABEL))
        d = ImageDraw.Draw(big)
        wpos = int(W[W[:, k] > 0, k].sum())
        wneg = int(W[W[:, k] < 0, k].sum())
        d.text((x + 4, y + 6), f"digit {k}  bias={bias[k]}  "
                               f"W+={wpos} W-={wneg}",
               fill=(0, 0, 0), font=font_s)
    big.save(os.path.join(RES, 'heatmap_all.png'))
    print(f"  heatmap_all.png  ({big.size[0]}x{big.size[1]})")

    # CSV
    with open(os.path.join(RES, 'weights_10x225.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['row', 'col', 'pixel'] + [f'digit{k}' for k in range(10)])
        for r in range(15):
            for c in range(15):
                p = r * 15 + c
                w.writerow([r, c, p] + [int(W[p, k]) for k in range(10)])
    print(f"  weights_10x225.csv")

    # 文字版：每个数字的 15x15 权重（便于直接看数）
    with open(os.path.join(RES, 'weights_heatmap.txt'), 'w', encoding='utf-8') as f:
        for k in range(10):
            f.write(f"{'='*60}\n digit {k}   bias={bias[k]}   "
                    f"W+={int(W[W[:,k]>0,k].sum())}  "
                    f"W-={int(W[W[:,k]<0,k].sum())}\n{'='*60}\n")
            f.write("      " + "".join(f"{c:>4}" for c in range(15)) + "\n")
            for r in range(15):
                f.write(f"  {r:>3} " + "".join(f"{int(W[r*15+c,k]):>4}"
                                               for c in range(15)) + "\n")
            f.write("\n")
    print(f"  weights_heatmap.txt")


if __name__ == '__main__':
    main()
