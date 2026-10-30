#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对称约束的混合近邻生成（czm 诊断的修正）
==========================================
问题: 逐像素独立采样 -> 左右眼/嘴各自独立 -> 温度一高就不对称
修正: 把 64 个像素按左右镜像配对成 32 对，**每对一起采样**

  X_sym  = (X + M(X))/2      对称分量（每对两像素共享）
  X_anti = (X - M(X))/2      反对称分量（每对两像素相反）

采样策略:
  A. 完全对称:   逐对只抽一个颜色，两像素都用它      (X_anti = 0)
  B. 对称+微扰:  逐对抽一个主色，再以概率 p 给一边抽个邻近色
  C. 无约束:     逐像素独立（旧做法，作对照）

配对方式: (i, 7-i) 共 4 对/行 * 8 行 = 32 对
"""
import os, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16

# 左右镜像配对: 位置 p = r*8+c  <->  r*8+(7-c)
PAIRS = []
for r in range(8):
    for c in range(4):
        PAIRS.append((r * 8 + c, r * 8 + (7 - c)))
PAIRS = np.array(PAIRS)          # (32,2)
IDX_ALL = np.arange(64)


def main():
    print("=" * 96)
    print("对称约束的混合近邻生成")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    POOL = 20000
    rng0 = np.random.default_rng(0)
    pool = lab[rng0.choice(len(lab), POOL, replace=False)]
    print(f"数据 {lab.shape}  原型池 {POOL}")

    # 真实头像的对称性有多强？先量化
    sym_part = pool[:, PAIRS[:, 0]]
    anti_part = pool[:, PAIRS[:, 1]]
    same = float((sym_part == anti_part).mean())
    print(f"真实头像中，左右镜像对**颜色相同**的比例 = {same:.3f}")
    print(f"  -> 反对称分量占比 {(1-same)*100:.1f}%（这就是真实的不对称程度）")

    def gen(count, k_nb, temp, mode, p_asym=0.0, seed=1):
        r = np.random.default_rng(seed)
        out = np.zeros((count, 64), dtype=np.int64)
        for j in range(count):
            a = r.integers(0, POOL)
            d = np.empty(POOL, dtype=np.int32)
            for s in range(0, POOL, 5000):
                d[s:s+5000] = (pool[s:s+5000] != pool[a]).sum(1)
            nb = np.argpartition(d, k_nb)[:k_nb]
            w = np.exp(-d[nb] / max(temp, 1e-6)); w /= w.sum()
            cum = np.cumsum(w)[None, :]
            vals = pool[nb]                       # (k,64)
            if mode == 'free':
                pick = np.clip((r.random(64)[:, None] > cum).sum(1), 0, k_nb - 1)
                out[j] = vals[pick, IDX_ALL]
            else:
                # 逐对采样：一对只抽一个邻居（对称），或两边各抽（不对称）
                pickL = np.clip((r.random(32)[:, None] > cum).sum(1), 0, k_nb - 1)
                colL = vals[pickL, PAIRS[:, 0]]
                if mode == 'sym':
                    colR = colL
                else:  # sym_jitter
                    use = r.random(32) < p_asym
                    pickR = np.clip((r.random(32)[:, None] > cum).sum(1), 0, k_nb - 1)
                    colR = np.where(use, vals[pickR, PAIRS[:, 1]], colL)
                out[j, PAIRS[:, 0]] = colL
                out[j, PAIRS[:, 1]] = colR
        return out

    def metrics(S):
        uc = np.mean([len(np.unique(s)) for s in S])
        A = S[:, None, :] != S[None, :, :]
        div = float(A.sum(2).sum() / (len(S) * (len(S) - 1)))
        # 对称性: 镜像对相同的比例
        sym = float((S[:, PAIRS[:, 0]] == S[:, PAIRS[:, 1]]).mean())
        dd = float(np.mean([(pool != s).sum(1).min() for s in S]))
        return uc, div, sym, dd

    def to_img(l): return pal[l].reshape(8, 8, 3)

    SC = 8
    def grid(imgs, cols=16):
        rows_ = (len(imgs) + cols - 1) // cols
        cv = np.full((rows_ * 8, cols * 8, 3), 40, dtype=np.uint8)
        for i, im in enumerate(imgs):
            r_, c_ = divmod(i, cols)
            cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = im
        big = np.kron(cv, np.ones((SC, SC, 1), dtype=np.uint8))
        for r_ in range(rows_ + 1):
            big[r_*8*SC:r_*8*SC+2, :] = 200
        for c_ in range(cols + 1):
            big[:, c_*8*SC:c_*8*SC+2] = 200
        return big

    print(f"\n  {'配置':<26} | {'颜色数':>6} | {'多样性':>6} | {'对称率':>6} | {'到最近真实':>9}")
    print("  " + "-" * 74)
    rows = []
    for mode, p_asym, temp, kn, nm in (
            ('free', 0, 0.5, 30, 'T=0.5 无约束(复制)'),
            ('free', 0, 5.0, 30, 'T=5.0 无约束(旧)'),
            ('sym', 0, 5.0, 30, 'T=5.0 **完全对称**'),
            ('sym', 0, 12.0, 30, 'T=12  完全对称'),
            ('sym', 0, 30.0, 30, 'T=30  完全对称'),
            ('sym_jitter', 0.15, 8.0, 30, 'T=8  对称+15%微扰'),
            ('sym_jitter', 0.30, 8.0, 30, 'T=8  对称+30%微扰'),
            ('sym_jitter', 0.15, 20.0, 30, 'T=20 对称+15%微扰'),
    ):
        S = gen(64, kn, temp, mode, p_asym, seed=hash(nm) % 10000)
        uc, div, sym, dd = metrics(S)
        print(f"  {nm:<26} | {uc:>6.2f} | {div:>6.1f} | {sym:>6.2f} | {dd:>9.1f}")
        Image.fromarray(grid([to_img(v) for v in S], 16)).save(
            os.path.join(RES, f'sym_{mode}_{int(temp)}_{int(p_asym*100)}.png'))
        rows.append({'name': nm, 'uc': uc, 'div': div, 'sym': sym, 'near': dd})

    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in pool[r2.choice(POOL, 64, replace=False)]], 16)).save(
        os.path.join(RES, 'sym_REAL.png'))
    print(f"\n  真实头像对称率 = {same:.3f}  (作为参考)")
    print(f"  已保存 sym_*.png  (文件名含模式/温度/微扰率)")
    json.dump(rows, open(os.path.join(RES, 'sym_results.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
