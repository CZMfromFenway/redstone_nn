#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合近邻生成（5 万张数据版）
==============================
模型: p(x) = sum_k pi_k p_k(x)
  每个成分 = 一个真实头像原型，采样时逐像素从 k 个最近邻的
  对应位置按距离加权取色 -> 颜色永远来自真实头像的局部统计

与 5484 张版的区别: 邻域覆盖更密，同样 k 下邻居更相似 -> 更协调
"""
import os, json, numpy as np, time, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16


def main():
    print("=" * 92)
    print("混合近邻生成（5 万张数据）")
    print("=" * 92)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    n = len(lab)
    print(f"数据: {lab.shape}")

    # 为了速度，只用 20000 张做原型池（距离矩阵 20000x20000 太大，按需算）
    POOL = min(n, 20000)
    rng0 = np.random.default_rng(0)
    pool = lab[rng0.choice(n, POOL, replace=False)]
    print(f"原型池: {POOL}")

    def sample_heads(count, k_nb, temp, seed=1):
        r = np.random.default_rng(seed)
        out = np.zeros((count, 64), dtype=np.int64)
        for j in range(count):
            a = r.integers(0, POOL)
            # 分块算汉明距离
            d = np.empty(POOL, dtype=np.int32)
            for s in range(0, POOL, 5000):
                d[s:s+5000] = (pool[s:s+5000] != pool[a]).sum(1)
            nb = np.argpartition(d, k_nb)[:k_nb]
            w = np.exp(-d[nb] / max(temp, 1e-6)); w /= w.sum()
            vals = pool[nb]                        # (k,64)
            # 按权重逐位置采样
            u = r.random(64)
            cum = np.cumsum(w)[None, :]            # (1,k)
            pick = (u[:, None] > cum).sum(1)       # 每个位置选第几个邻居
            pick = np.clip(pick, 0, k_nb - 1)
            out[j] = vals[pick, np.arange(64)]
        return out

    def diversity(S):
        A = S[:, None, :] != S[None, :, :]
        m = len(S)
        return float(A.sum(2).sum() / (m * (m - 1)))

    real_div = diversity(pool[:120])
    real_uc = np.mean([len(np.unique(s)) for s in pool[:3000]])
    print(f"真实: 颜色数 {real_uc:.2f}   样本间多样性 {real_div:.1f}")

    print(f"\n  {'k':>4} | {'温度':>5} | {'颜色数':>7} | {'多样性':>7} | {'到最近真实':>9} | {'用时':>6}")
    print("  " + "-" * 62)
    rows = []
    for k_nb in (3, 10, 30, 100, 300):
        for temp in (0.5, 2.0, 8.0):
            t0 = time.time()
            S = sample_heads(400, k_nb, temp, seed=k_nb * 10 + int(temp * 10))
            uc = np.mean([len(np.unique(s)) for s in S])
            dv = diversity(S)
            dd = np.mean([(pool != s).sum(1).min() for s in S])
            print(f"  {k_nb:>4} | {temp:>5.1f} | {uc:>7.2f} | {dv:>7.1f} | {dd:>9.1f} | "
                  f"{time.time()-t0:>5.0f}s", flush=True)
            rows.append({'k': k_nb, 'temp': temp, 'uc': float(uc), 'div': dv,
                         'near': float(dd)})

    # 选: 颜色数接近真实 且 多样性高 且 到最近真实足够远
    best = max(rows, key=lambda r: r['div'] - 20 * abs(r['uc'] - real_uc))
    print(f"\n  最佳配置: k={best['k']}, 温度={best['temp']}  "
          f"(颜色数 {best['uc']:.2f}, 多样性 {best['div']:.1f})")

    S = sample_heads(60, best['k'], best['temp'], seed=123)

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

    Image.fromarray(grid([to_img(v) for v in S], 12)).save(
        os.path.join(RES, 'mix50k_samples.png'))
    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in pool[r2.choice(POOL, 60, replace=False)]], 12)).save(
        os.path.join(RES, 'mix50k_real.png'))
    print(f"已保存: mix50k_samples.png  mix50k_real.png")
    json.dump(rows, open(os.path.join(RES, 'mix50k_results.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
