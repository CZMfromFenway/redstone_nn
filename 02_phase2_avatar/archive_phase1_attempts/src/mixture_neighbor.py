#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C 方案（修正）：混合近邻模型 —— 保持空间结构
==============================================
上一版的问题: 朴素贝叶斯（64 像素条件独立）-> 空间结构被破坏 -> 噪点

本版: 每个"成分"用**一个真实头像**作为原型，采样时:
   1. 随机选一个原型 a（按数据分布，可加权）
   2. 找它最近的 k 个邻居
   3. 逐像素从"邻居在该位置出现的颜色"中按距离加权采样
   -> 颜色来自真实头像的**局部统计**，空间结构得以保留

这仍然是一个**概率生成模型**（不依赖对抗），但成分是"数据点邻域"
而不是"独立的逐像素分布"。
"""
import os, json, numpy as np
from PIL import Image
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16


def main():
    print("=" * 92)
    print("C 方案（修正）：混合近邻生成 8x8 头像")
    print("=" * 92)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    print(f"数据: {lab.shape}")

    rng = np.random.default_rng(0)
    n = len(lab)

    def sample_heads(count, k_nb, temp=1.0, seed=1):
        """k_nb 近邻逐像素采样"""
        r = np.random.default_rng(seed)
        out = np.zeros((count, 64), dtype=np.int64)
        for j in range(count):
            a = r.integers(0, n)
            d = (lab != lab[a]).sum(1)          # 汉明距离
            nb = np.argsort(d)[:k_nb]
            w = np.exp(-d[nb] / max(temp, 1e-6))
            w = w / w.sum()
            for i in range(64):
                vals = lab[nb, i]
                # 按权重取该位置的颜色分布
                u = r.random()
                cum = 0.0
                pick = vals[0]
                for v, ww in zip(vals, w):
                    cum += ww
                    if u <= cum:
                        pick = v; break
                out[j, i] = pick
        return out

    # 多样性/一致性指标
    def stats(S):
        uc = np.array([len(np.unique(s)) for s in S])
        # 与最近真实样本的汉明距离
        dd = np.array([(lab != s).sum(1).min() for s in S[:500]])
        return uc.mean(), dd.mean()

    real_uc, _ = stats(lab[:2000])
    print(f"真实头像: 平均颜色数 {real_uc:.2f}")

    print(f"\n  {'k近邻':>6} | {'温度':>5} | {'生成颜色数':>10} | {'到最近真实的汉明距离':>20}")
    print("  " + "-" * 56)
    best = None
    for k_nb in (3, 8, 20, 50):
        for temp in (0.5, 2.0, 10.0):
            S = sample_heads(600, k_nb, temp, seed=k_nb * 10 + int(temp * 10))
            uc, dd = stats(S)
            # 选优标准: 颜色数接近真实, **且** 汉明距离足够大(有真正的新颖性)
            score = abs(uc - real_uc) - 0.15 * min(dd, 8.0)
            print(f"  {k_nb:>6} | {temp:>5.1f} | {uc:>10.2f} | {dd:>20.1f} | 评分 {score:>7.3f}",
                  flush=True)
            if best is None or score < best[0]:
                best = (score, k_nb, temp)

    print(f"\n  最佳配置(兼顾颜色数与新颖性): k={best[1]}, 温度={best[2]}")
    S = sample_heads(48, best[1], best[2], seed=123)

    def to_img(row):
        return pal[row].reshape(8, 8, 3)

    SC = 12
    def grid(imgs, cols=12):
        rows = (len(imgs) + cols - 1) // cols
        cv = np.full((rows * 8, cols * 8, 3), 40, dtype=np.uint8)
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            cv[r*8:(r+1)*8, c*8:(c+1)*8] = im
        big = np.kron(cv, np.ones((SC, SC, 1), dtype=np.uint8))
        for r in range(rows + 1):
            big[r*8*SC:r*8*SC+1, :] = 90
        for c in range(cols + 1):
            big[:, c*8*SC:c*8*SC+1] = 90
        return big

    Image.fromarray(grid([to_img(r) for r in S])).save(
        os.path.join(RES, 'mixture_samples.png'))
    R = lab[rng.choice(n, 48, replace=False)]
    Image.fromarray(grid([to_img(r) for r in R])).save(
        os.path.join(RES, 'mixture_real.png'))
    print(f"\n已保存: {RES}/mixture_samples.png  mixture_real.png")


if __name__ == "__main__":
    main()
