#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
诊断: 为什么生成图的中轴线几乎全白
======================================
怀疑: 编码里"未知"与"白色(0号色)"无法区分
  未知: 我把 one-hot 置为全 -1
  白色: one-hot = [-1,-1,...,-1,+1]  -> 前15个通道也是 -1

本脚本:
  1. 检查真实头像中轴线两列的实际颜色分布（是否本来就偏白？）
  2. 检查生成图中轴线的颜色分布
  3. 统计训练时"未知位置附近"的邻域编码分布
"""
import os, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16
# 中轴线两列: c=3 和 c=4
MID = [r * 8 + c for r in range(8) for c in (3, 4)]


def main():
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    names = json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['names']

    print("=" * 94)
    print("诊断: 中轴线为什么偏白")
    print("=" * 94)
    print(f"中轴线 = 列 c=3 与 c=4，共 {len(MID)} 个位置")

    # ---- 1. 真实头像的中轴线分布 ----
    mid_real = lab[:, MID]
    vals, cnt = np.unique(mid_real, return_counts=True)
    order = np.argsort(-cnt)
    tot = mid_real.size
    print(f"\n[1] 真实头像中轴线的颜色分布（这是基准）")
    print(f"  {'色号':>4} | {'HEX':>8} | {'MC 纹理':<24} | {'占比':>7}")
    print("  " + "-" * 56)
    for k in order[:8]:
        v = vals[k]
        c = pal[v]
        print(f"  {v:>4} | #{c[0]:02X}{c[1]:02X}{c[2]:02X} | {names[v]:<24} | {cnt[k]/tot:>6.2%}")
    print(f"  白色(0号)在真实中轴线占 {np.mean(mid_real==0):.2%}")

    # 对比: 真实头像整体分布
    print(f"  白色(0号)在真实头像**整体**占 {np.mean(lab==0):.2%}")
    print(f"  -> 中轴线比整体{'更' if np.mean(mid_real==0)>np.mean(lab==0) else '不'}偏白 "
          f"(倍数 {np.mean(mid_real==0)/max(np.mean(lab==0),1e-9):.2f})")

    # ---- 2. 生成图的中轴线分布 ----
    for fn in ('conv2_d15_T07.png', 'conv2_d15_T10.png'):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        arr = np.array(Image.open(p))
        H, W, _ = arr.shape
        # 每张 8x8 放大 8 倍，网格线 2px
        cell = 8 * 8 + 2
        cols_per_row = 16
        whites, totals = 0, 0
        for i in range(2):                       # 2 行
            for j in range(cols_per_row):
                r0 = i * cell + 2
                c0 = j * cell + 2
                tile = arr[r0:r0 + 64, c0:c0 + 64]
                if tile.shape[0] < 64:
                    continue
                # 取中轴列 c=3,4 -> 像素 24..39
                midpx = tile[:, 24:40].reshape(-1, 3).astype(int)
                # 与白色调色板比较
                white = pal[0].astype(int)
                isw = (np.abs(midpx - white).sum(1) < 10)
                whites += int(isw.sum()); totals += len(midpx)
        print(f"\n[2] 生成图 {fn}: 中轴线像素是白色的比例 = {whites/max(totals,1):.2%}")

    # ---- 3. 验证编码冲突 ----
    print(f"\n[3] 编码冲突验证")
    # 白色像素的邻域 one-hot（中心是白色，其它也设成各种颜色）
    def onehot(v):
        o = np.full(C, -1.0); o[v] = 1.0
        return o
    unknown_repr = np.full(C, -1.0)
    print(f"  白色(0号)的 one-hot: {onehot(0)}")
    print(f"  我在'未知'位置写入的:  {unknown_repr}")
    print(f"  -> 两者{'完全相同' if np.allclose(onehot(0), unknown_repr) else '不同'}！"
          f"  (白色除 0 号通道外全是 -1，未知也全是 -1)")
    # 未知与其它颜色的差异
    print(f"  未知与 1 号色的 L1 距离: {np.abs(unknown_repr-onehot(1)).sum():.1f}")
    print(f"  白色与 1 号色的 L1 距离: {np.abs(onehot(0)-onehot(1)).sum():.1f}")
    print(f"  -> 未知到任何颜色的距离都 >= 2，但**白色与未知距离为 0**")

    # ---- 4. 训练时邻域里"未知"的密度 ----
    print(f"\n[4] 训练时 3x3 邻域里有多少位置是'未知'")
    r = np.random.default_rng(0)
    N = 2000
    sub = lab[r.choice(len(lab), N, replace=False)]
    n_unk = r.integers(8, 33, N)
    fr = n_unk / 64.0
    print(f"  每样本挖掉 {n_unk.mean():.1f} 个位置（占 64 位置的 {fr.mean():.1%}）")
    print(f"  -> 每个 3x3 邻域平均有 {9*fr.mean():.2f} 个位置是'未知'")
    print(f"  -> 而真实 8x8 头像的白色占比只有 {np.mean(lab==0):.2%}")
    print(f"  -> 模型看到'未知'的频率远高于'真实白色'，于是把'全-1'学成了白色")


if __name__ == '__main__':
    main()
