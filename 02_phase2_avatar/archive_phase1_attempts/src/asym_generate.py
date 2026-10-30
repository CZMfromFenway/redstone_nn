#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对称 + 反对称分解的生成
=========================
分解（左半 s，反对称差 d）:
    s_i = x_i                 (左半 32 个位置)
    d_i = x_{M(i)} - x_i      (右 - 左，索引有序故差值有颜色意义)
    还原: 左 = s,  右 = clamp(s + d, 0, 15)

本脚本对比两档 d 的范围:
    d ∈ [-3,+3]   7 档   （红石友好）
    d ∈ [-15,+15] 31 档  （无损）

生成方式（分解版混合近邻）:
    1. 随机取一个原型 (s,d)
    2. 在 (s,d) 联合空间找 k 近邻
    3. 逐位置按距离加权采样 s 和 d（两者一起采样，保持配对）
    4. 合成头像

输出文件:
    asym_d3_<档位>.png         d量化到±3
    asym_d15_<档位>.png        d量化到±15
    asym_compare_d3.png        总表（行=温度，列=同源头像）
    asym_compare_d15.png       总表
    asym_real.png              真实对照
"""
import os, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16

LEFT = [(r, c) for r in range(8) for c in range(4)]
LIDX = np.array([r * 8 + c for r, c in LEFT], dtype=int)
RIDX = np.array([r * 8 + (7 - c) for r, c in LEFT], dtype=int)

NCOL = 16
TEMPS = [0.5, 2.0, 5.0, 8.0, 15.0]
K_NB = 30


def main():
    print("=" * 96)
    print("对称 + 反对称分解的生成")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)

    s_all = lab[:, LIDX]                      # (n,32)
    d_all = lab[:, RIDX] - lab[:, LIDX]       # (n,32)
    print(f"数据 {lab.shape}   s {s_all.shape}  d {d_all.shape}")
    print(f"d 范围 [{d_all.min()},{d_all.max()}]   |d|=0 占比 {(d_all==0).mean():.2%}")

    rng0 = np.random.default_rng(0)
    POOL = 20000
    sel = rng0.choice(len(lab), POOL, replace=False)
    S = s_all[sel]; D = d_all[sel]

    # 固定一批原型（供对照表用同一列）
    r = np.random.default_rng(2024)
    proto = r.choice(POOL, NCOL, replace=False)

    def gen_from(pi, k_nb, temp, dclip, seed):
        """从原型 pi 生成一张"""
        rr = np.random.default_rng(seed)
        # 联合空间距离 = s 上的差异 + d 上的差异
        ds = np.empty(POOL, dtype=np.int32); dd = np.empty(POOL, dtype=np.int32)
        for t in range(0, POOL, 5000):
            ds[t:t+5000] = (S[t:t+5000] != S[pi]).sum(1)
            dd[t:t+5000] = (D[t:t+5000] != D[pi]).sum(1)
        dist = ds + dd
        nb = np.argpartition(dist, k_nb)[:k_nb]
        w = np.exp(-dist[nb] / max(temp, 1e-6)); w /= w.sum()
        cum = np.cumsum(w)[None, :]
        ps = np.clip((rr.random(32)[:, None] > cum).sum(1), 0, k_nb - 1)
        pd = np.clip((rr.random(32)[:, None] > cum).sum(1), 0, k_nb - 1)
        sv = S[nb][ps, np.arange(32)]
        dv = np.clip(D[nb][pd, np.arange(32)], -dclip, dclip)
        X = np.zeros(64, dtype=np.int64)
        X[LIDX] = sv
        X[RIDX] = np.clip(sv + dv, 0, C - 1)
        return X

    def to_img(l): return pal[l].reshape(8, 8, 3)

    SC = 8
    def grid(imgs, cols=NCOL):
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

    # 真实对照（用同一批原型）
    real64 = lab[sel[proto]]
    Image.fromarray(grid([to_img(v) for v in real64], NCOL)).save(
        os.path.join(RES, 'asym_real.png'))
    print(f"  已保存 asym_real.png （真实原型，{NCOL} 张）")

    summary = {}
    for dclip, tag in ((3, 'd3'), (15, 'd15')):
        print(f"\n{'#'*96}")
        print(f"### d 量化到 ±{dclip}   （{2*dclip+1} 档）")
        print(f"{'#'*96}")
        rows_img = []
        stats = []
        for temp in TEMPS:
            imgs, Sg = [], []
            for ci in range(NCOL):
                X = gen_from(proto[ci], K_NB, temp, dclip, seed=ci * 977 + int(temp * 10))
                Sg.append(X)
                imgs.append(to_img(X))
            Sg = np.array(Sg)
            # 指标
            uc = np.mean([len(np.unique(v)) for v in Sg])
            symrate = float((Sg[:, LIDX] == Sg[:, RIDX]).mean())
            dsrc = float((Sg != real64).sum(1).mean())
            dnear = float(np.mean([(lab != v).sum(1).min() for v in Sg]))
            stats.append((temp, uc, symrate, dsrc, dnear))
            rows_img.append(imgs)
            Image.fromarray(grid(imgs, NCOL)).save(
                os.path.join(RES, f'asym_{tag}_T{int(temp*10):03d}.png'))
            print(f"  T={temp:<6} 颜色数 {uc:.2f}  对称率 {symrate:.2f}  "
                  f"到源头像 {dsrc:5.1f}  到最近真实 {dnear:5.1f}  -> asym_{tag}_T{int(temp*10):03d}.png",
                  flush=True)

        # 总表
        allimgs = [im for row in rows_img for im in row]
        Image.fromarray(grid(allimgs, NCOL)).save(
            os.path.join(RES, f'asym_compare_{tag}.png'))
        summary[tag] = {'dclip': dclip, 'stats': stats,
                        'file': f'asym_compare_{tag}.png'}
        print(f"  总表 -> asym_compare_{tag}.png （{len(TEMPS)} 行 × {NCOL} 列，行=温度）")

    # 真实头像的对称率参考
    sym_real = float((lab[:, LIDX] == lab[:, RIDX]).mean())
    print(f"\n真实头像对称率 = {sym_real:.3f}")
    json.dump(summary, open(os.path.join(RES, 'asym_summary.json'), 'w'),
              ensure_ascii=False, indent=2)
    print("已写出 results/asym_summary.json")


if __name__ == '__main__':
    main()
