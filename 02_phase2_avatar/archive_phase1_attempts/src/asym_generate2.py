#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对称+反对称分解的生成（修正：s 与 d 必须配对）
================================================
上一版的错误: s 和 d 各自独立采样 -> s 来自邻居 A、d 来自邻居 B
  -> 配对关系丢失 -> 脸部左右不一致（尽管 d 很小）

本版修正:
  1. s 和 d 从**同一个邻居**取出（保持配对）
  2. d 用**逐位置**从邻居集合中采样（而非整行取），保留变化
  3. 关键约束: d 只在"真实 d 允许的位置"非零

输出:
  asym2_d3_<档位>.png / asym2_d15_<档位>.png    单档
  asym2_compare_d3.png / asym2_compare_d15.png  总表（行=温度）
  asym2_real.png                                真实对照
"""
import os, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16
LIDX = np.array([r * 8 + c for r in range(8) for c in range(4)], dtype=int)
RIDX = np.array([r * 8 + (7 - c) for r in range(8) for c in range(4)], dtype=int)
NCOL = 16
TEMPS = [0.5, 2.0, 5.0, 8.0, 15.0]
K_NB = 30


def main():
    print("=" * 96)
    print("对称+反对称分解生成 v2（s 与 d 配对）")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    s_all = lab[:, LIDX]; d_all = lab[:, RIDX] - lab[:, LIDX]

    rng0 = np.random.default_rng(0)
    POOL = 20000
    sel = rng0.choice(len(lab), POOL, replace=False)
    S, D = s_all[sel], d_all[sel]
    r = np.random.default_rng(2024)
    proto = r.choice(POOL, NCOL, replace=False)

    def gen_from(pi, temp, dclip, seed):
        rr = np.random.default_rng(seed)
        ds = np.empty(POOL, dtype=np.int32); dd = np.empty(POOL, dtype=np.int32)
        for t in range(0, POOL, 5000):
            ds[t:t+5000] = (S[t:t+5000] != S[pi]).sum(1)
            dd[t:t+5000] = (D[t:t+5000] != D[pi]).sum(1)
        dist = ds + dd
        nb = np.argpartition(dist, K_NB)[:K_NB]
        w = np.exp(-dist[nb] / max(temp, 1e-6)); w /= w.sum()
        cum = np.cumsum(w)[None, :]
        # 关键修正: s 和 d 用**同一个 pick**，保证来自同一邻居
        pick = np.clip((rr.random(32)[:, None] > cum).sum(1), 0, K_NB - 1)
        sv = S[nb][pick, np.arange(32)]
        dv = np.clip(D[nb][pick, np.arange(32)], -dclip, dclip)
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

    real64 = lab[sel[proto]]
    Image.fromarray(grid([to_img(v) for v in real64])).save(os.path.join(RES, 'asym2_real.png'))
    print(f"真实对照 -> asym2_real.png")

    sym_real = float((real64[:, LIDX] == real64[:, RIDX]).mean())
    print(f"（这批原型的对称率 {sym_real:.2f}）\n")

    for dclip, tag in ((3, 'd3'), (15, 'd15')):
        print(f"{'#'*96}\n### d ∈ [±{dclip}]  ({2*dclip+1} 档)\n{'#'*96}")
        rows_img, stats = [], []
        for temp in TEMPS:
            imgs, Sg = [], []
            for ci in range(NCOL):
                X = gen_from(proto[ci], temp, dclip, seed=ci * 977 + int(temp * 10))
                Sg.append(X); imgs.append(to_img(X))
            Sg = np.array(Sg)
            uc = np.mean([len(np.unique(v)) for v in Sg])
            symrate = float((Sg[:, LIDX] == Sg[:, RIDX]).mean())
            dsrc = float((Sg != real64).sum(1).mean())
            dnear = float(np.mean([(lab != v).sum(1).min() for v in Sg]))
            # 不对称强度: |d| 均值
            dm = float(np.abs(Sg[:, RIDX] - Sg[:, LIDX]).mean())
            stats.append({'temp': temp, 'uc': float(uc), 'sym': symrate,
                          'd_src': dsrc, 'd_near': dnear, 'absd': dm})
            rows_img.append(imgs)
            Image.fromarray(grid(imgs)).save(os.path.join(RES, f'asym2_{tag}_T{int(temp*10):03d}.png'))
            print(f"  T={temp:<5} 颜色数 {uc:.2f}  对称率 {symrate:.2f}  |d|均值 {dm:.2f}  "
                  f"到源头像 {dsrc:5.1f}  到最近真实 {dnear:5.1f}", flush=True)
        Image.fromarray(grid([im for row in rows_img for im in row])).save(
            os.path.join(RES, f'asym2_compare_{tag}.png'))
        print(f"  总表 -> asym2_compare_{tag}.png\n")


if __name__ == '__main__':
    main()
