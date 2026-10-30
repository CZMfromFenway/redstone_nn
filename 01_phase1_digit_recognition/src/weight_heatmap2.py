#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
权重热图（增强对比度版）+ 偏置/G_i0 可视化
=============================================
1. heatmap_all_boost.png  : 每张图按自己的最大绝对值归一化（看清内部结构）
2. bias_chart.png         : 10 层的 bias / W正和 / W负和 / F_0 / G_i0
3. heatmap_row0.png       : 第 0 行的 10 个数字权重对比（示例）
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
NEG = np.array([20, 70, 190], dtype=float)
MID = np.array([252, 252, 252], dtype=float)
POS = np.array([200, 30, 30], dtype=float)


def colorize(w, vmax):
    t = np.clip(w / max(vmax, 1), -1, 1)[:, :, None]
    out = np.where(t < 0, MID + (NEG - MID) * (-t), MID + (POS - MID) * t)
    return np.clip(out, 0, 255).astype(np.uint8)


def font(sz):
    try:
        return ImageFont.truetype("arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def main():
    d = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = d['W'].astype(int); bias = d['b'].astype(int)
    SC, LABEL, GAP = 22, 36, 14

    # 逐层容器值范围（用于标注）
    def cinfo(k):
        ln = k + 1
        par = 'even' if ln % 2 == 0 else 'odd'
        c = (15 - W[:, k]) if par == 'even' else (15 + W[:, k])
        return par, int(c.min()), int(c.max())

    def render(k, vmax):
        img = colorize(W[:, k].reshape(15, 15), vmax)
        big = np.kron(img, np.ones((SC, SC, 1), dtype=np.uint8))
        for i in range(16):
            big[max(0, i*SC-1):i*SC+1, :] = 130
            big[:, max(0, i*SC-1):i*SC+1] = 130
        return big

    # ---- 增强对比度拼图（每图单独归一化） ----
    tiles = [render(k, int(np.abs(W[:, k]).max())) for k in range(10)]
    th, tw = tiles[0].shape[:2]
    cols, rows = 5, 2
    Wt = cols * tw + (cols + 1) * GAP
    Ht = rows * (th + LABEL) + (rows + 1) * GAP
    big = Image.new('RGB', (Wt, Ht), (243, 243, 243))
    dr = ImageDraw.Draw(big)
    f = font(15)
    for k in range(10):
        r, c = divmod(k, 5)
        x = GAP + c * (tw + GAP)
        y = GAP + r * (th + LABEL + GAP)
        big.paste(Image.fromarray(tiles[k]), (x, y + LABEL))
        par, cmin, cmax = cinfo(k)
        wpos = int(W[W[:, k] > 0, k].sum()); wneg = int(W[W[:, k] < 0, k].sum())
        dr.text((x + 4, y + 2), f"digit {k}   layer {k+1} ({par})   "
                                f"bias={bias[k]}", fill=(0, 0, 0), font=f)
        dr.text((x + 4, y + 19), f"W+={wpos}  W-={wneg}   "
                                f"c=[{cmin},{cmax}]", fill=(70, 70, 70), font=f)
    big.save(os.path.join(RES, 'heatmap_all_boost.png'))
    print(f"heatmap_all_boost.png  {big.size}")

    # ---- 第 0 行（顶部像素行）10 个数字的权重对比 ----
    def cellrow(k, row=0, vmax=12, sc=30):
        img = colorize(W[row*15:(row+1)*15, k].reshape(1, 15), vmax)
        big = np.kron(img, np.ones((sc, sc, 1), dtype=np.uint8))
        for i in range(16):
            big[:, max(0, i*sc-1):i*sc+1] = 130
        return big

    strips = [cellrow(k) for k in range(10)]
    sh, sw = strips[0].shape[:2]
    canvas = Image.new('RGB', (sw + 80, 10 * (sh + 8) + 10), (255, 255, 255))
    dd = ImageDraw.Draw(canvas)
    for k in range(10):
        y = 6 + k * (sh + 8)
        canvas.paste(Image.fromarray(strips[k]), (76, y))
        dd.text((4, y + 6), f"digit {k}", fill=(0, 0, 0), font=font(16))
    canvas.save(os.path.join(RES, 'heatmap_row0_compare.png'))
    print(f"heatmap_row0_compare.png  {canvas.size}")

    # ---- 偏置图表 ----
    f2 = font(14)
    CW, CH = 96, 150
    canvas = Image.new('RGB', (10 * CW + 40, 3 * CH + 120), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)

    def bar(x, y, w, h, val, vmax, col):
        bh = int(abs(val) / vmax * (h - 24))
        if val >= 0:
            dr.rectangle([x, y + h - 24 - bh, x + w, y + h - 24], fill=col)
        else:
            dr.rectangle([x, y + h - 24, x + w, y + h - 24 + bh], fill=col)

    # 子图 1: bias
    dr.text((6, 4), "bias", fill=(0, 0, 0), font=f2)
    vb = max(abs(bias).max(), 1)
    for k in range(10):
        x = 40 + k * CW
        dr.rectangle([x, 40, x + CW - 18, 40 + CH - 24], outline=(200, 200, 200))
        bar(x + 2, 40 + (CH - 24) // 2, CW - 22, CH, int(bias[k]), vb, (60, 120, 220))
        dr.text((x, 40 + CH - 16), f"{bias[k]:>4}", fill=(0, 0, 0), font=f2)
        dr.text((x, 40 - 16), f"L{k+1}", fill=(0, 0, 0), font=f2)
    # 子图 2: W正和 / W负和
    y0 = CH + 50
    dr.text((6, y0 - 18), "W+ (red, up)  /  W- (blue, down)", fill=(0, 0, 0), font=f2)
    vw = 300
    for k in range(10):
        x = 40 + k * CW
        dr.rectangle([x, y0 + 40, x + CW - 18, y0 + CH - 24], outline=(200, 200, 200))
        wp = int(W[W[:, k] > 0, k].sum()); wn = int(W[W[:, k] < 0, k].sum())
        bar(x + 2, y0 + 40 + (CH - 24) // 2, CW - 22, CH, wp, vw, (200, 60, 60))
        bar(x + CW // 2 - 9, y0 + 40 + (CH - 24) // 2, CW - 22, CH, wn, vw, (60, 120, 220))
        dr.text((x, y0 + 40 + CH - 16), f"{wp:>4}", fill=(160, 0, 0), font=f2)
        dr.text((x, y0 + 40 - 16), f"{wn:>4}", fill=(0, 0, 160), font=f2)
        dr.text((x, y0 - 2), f"L{k+1}", fill=(0, 0, 0), font=f2)
    canvas.save(os.path.join(RES, 'bias_chart.png'))
    print(f"bias_chart.png  {canvas.size}")

    # ---- G_i0 表格图 ----
    canvas = Image.new('RGB', (10 * 96 + 40, 4 * 34 + 60), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 4), "G_i0 (i=1..8) and F_0 per layer", fill=(0, 0, 0), font=font(15))
    hdr = ["L", "G10", "G20", "G30", "G40", "G50", "G60", "G70", "G80", "F0"]
    for j, h in enumerate(hdr):
        dr.text((10 + j * 92, 30), h, fill=(0, 0, 0), font=f2)
    for k in range(10):
        c = (15 - W[:, k]) if (k + 1) % 2 == 0 else (15 + W[:, k])
        gs = []
        for (r0, nr) in GROUPS:
            pxs = [r * 15 + cc for r in range(r0, r0 + nr) for cc in range(15)]
            cg = c[pxs]
            gs.append(int(np.maximum(cg, 15).sum()))
        F0 = None
        row = [f"L{k+1}"] + [str(g) for g in gs] + ["?"]
        for j, v in enumerate(row):
            dr.text((10 + j * 92, 56 + k * 34), v, fill=(0, 0, 0), font=f2)
    canvas.save(os.path.join(RES, 'G_i0_chart.png'))
    print(f"G_i0_chart.png  {canvas.size}")


if __name__ == '__main__':
    main()
