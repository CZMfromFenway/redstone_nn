#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合近邻：新颖性对照（固定 k=30，扫温度）
==========================================
输出三档: T=0.5(近似复制) / T=2.0(微调) / T=8.0(重组, 有新像素)
目的是看清"质量 vs 新颖性"的实际权衡，而不是只看数字。
"""
import os, json, time
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16


def main():
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    rng0 = np.random.default_rng(0)
    POOL = 20000
    pool = lab[rng0.choice(len(lab), POOL, replace=False)]

    def gen(count, k_nb, temp, seed):
        r = np.random.default_rng(seed)
        out = np.zeros((count, 64), dtype=np.int64)
        for j in range(count):
            a = r.integers(0, POOL)
            d = np.empty(POOL, dtype=np.int32)
            for s in range(0, POOL, 5000):
                d[s:s+5000] = (pool[s:s+5000] != pool[a]).sum(1)
            nb = np.argpartition(d, k_nb)[:k_nb]
            w = np.exp(-d[nb] / max(temp, 1e-6)); w /= w.sum()
            vals = pool[nb]
            cum = np.cumsum(w)[None, :]
            pick = np.clip((r.random(64)[:, None] > cum).sum(1), 0, k_nb - 1)
            out[j] = vals[pick, np.arange(64)]
        return out

    def to_img(l): return pal[l].reshape(8, 8, 3)

    SC = 12
    def grid(imgs, cols=12):
        rows_ = (len(imgs) + cols - 1) // cols
        cv = np.full((rows_ * 8, cols * 8, 3), 40, dtype=np.uint8)
        for i, im in enumerate(imgs):
            r_, c_ = divmod(i, cols)
            cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = im
        big = np.kron(cv, np.ones((SC, SC, 1), dtype=np.uint8))
        for r_ in range(rows_ + 1):
            big[r_*8*SC:r_*8*SC+1, :] = 90
        for c_ in range(cols + 1):
            big[:, c_*8*SC:c_*8*SC+1] = 90
        return big

    for temp, nm, kn in ((0.5, 'copy_hard', 30), (2.0, 'light', 30),
                         (5.0, 'mid', 30), (12.0, 'recombine', 30),
                         (5.0, 'mid_k100', 100)):
        S = gen(60, kn, temp, seed=int(temp * 100) + kn)
        dd = np.mean([(pool != s).sum(1).min() for s in S])
        uc = np.mean([len(np.unique(s)) for s in S])
        Image.fromarray(grid([to_img(v) for v in S], 12)).save(
            os.path.join(RES, f'mix50k_{nm}.png'))
        print(f"  k={kn:<4} T={temp:<5} 颜色数 {uc:.2f}  到最近真实 {dd:5.1f}  -> mix50k_{nm}.png",
              flush=True)

    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in pool[r2.choice(POOL, 60, replace=False)]], 12)).save(
        os.path.join(RES, 'mix50k_real.png'))
    print("  真实对照 -> mix50k_real.png")


if __name__ == '__main__':
    main()
