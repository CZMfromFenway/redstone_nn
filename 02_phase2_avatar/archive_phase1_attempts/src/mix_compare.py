#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
温度对照总表
==============
布局: 行 = 配置（真实 / 各温度 / 各k），列 = 同一批原型
  每一行都用**同一批源头像**，所以纵向对比就是"同一张脸在各温度下变成什么"
输出: mix_compare.png （大图，每张 8x8 放大 8 倍）

同时打印每档的两个距离:
  到源头像      —— 偏离基准多少（小=几乎没动）
  到最近真实    —— 是否有真正的"新像素"（0=复制训练样本）
"""
import os, json
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

    NCOL = 16
    r = np.random.default_rng(2024)
    proto_idx = r.choice(POOL, NCOL, replace=False)     # 固定一批原型
    protos = pool[proto_idx]

    def gen_from(proto, k_nb, temp, seed):
        """从指定原型出发生成一张"""
        rr = np.random.default_rng(seed)
        d = np.empty(POOL, dtype=np.int32)
        for s in range(0, POOL, 5000):
            d[s:s+5000] = (pool[s:s+5000] != proto).sum(1)
        nb = np.argpartition(d, k_nb)[:k_nb]
        w = np.exp(-d[nb] / max(temp, 1e-6)); w /= w.sum()
        vals = pool[nb]
        cum = np.cumsum(w)[None, :]
        pick = np.clip((rr.random(64)[:, None] > cum).sum(1), 0, k_nb - 1)
        return vals[pick, np.arange(64)]

    configs = [
        ('真实原型', None),
        ('T=0.5  (复制)', (30, 0.5)),
        ('T=2.0  (微调)', (30, 2.0)),
        ('T=5.0  (重组)', (30, 5.0)),
        ('T=8.0', (30, 8.0)),
        ('T=12.0 (噪)', (30, 12.0)),
        ('k=100 T=5', (100, 5.0)),
        ('k=300 T=5', (300, 5.0)),
    ]

    SC = 8
    rows_img = []
    stats = []
    for name, cfg in configs:
        if cfg is None:
            imgs = [pal[p].reshape(8, 8, 3) for p in protos]
            stats.append((name, 0.0, 0.0))
        else:
            k_nb, temp = cfg
            S = [gen_from(protos[i], k_nb, temp, seed=i * 977 + int(temp * 10)) for i in range(NCOL)]
            S = np.array(S)
            imgs = [pal[s].reshape(8, 8, 3) for s in S]
            d_src = float((S != protos).sum(1).mean())
            d_near = float(np.mean([(pool != s).sum(1).min() for s in S]))
            stats.append((name, d_src, d_near))
        rows_img.append(imgs)

    # 拼图: 每行上方留标签空间 → 直接拼，标签用文件名表达
    R = len(rows_img)
    canvas = np.full((R * 8, NCOL * 8, 3), 40, dtype=np.uint8)
    for ri, imgs in enumerate(rows_img):
        for ci, im in enumerate(imgs):
            canvas[ri*8:(ri+1)*8, ci*8:(ci+1)*8] = im
    big = np.kron(canvas, np.ones((SC, SC, 1), dtype=np.uint8))
    for ri in range(R + 1):
        big[ri*8*SC:ri*8*SC+2, :] = 200
    for ci in range(NCOL + 1):
        big[:, ci*8*SC:ci*8*SC+2] = 200
    Image.fromarray(big).save(os.path.join(RES, 'mix_compare.png'))

    # 每档单独出一张（方便单看）
    for (name, cfg), imgs in zip(configs, rows_img):
        tag = name.split()[0].replace('=', '').replace('(', '').replace(')', '')
        cv = np.full((8, NCOL * 8, 3), 40, dtype=np.uint8)
        for ci, im in enumerate(imgs):
            cv[:, ci*8:(ci+1)*8] = im
        b2 = np.kron(cv, np.ones((SC, SC, 1), dtype=np.uint8))
        for ci in range(NCOL + 1):
            b2[:, ci*8*SC:ci*8*SC+2] = 200
        Image.fromarray(b2).save(os.path.join(RES, f'mixrow_{tag}.png'))

    print("=" * 84)
    print("温度对照总表  (mix_compare.png: 行=配置, 列=同一批源头像)")
    print("=" * 84)
    print(f"  {'配置':<18} | {'到源头像(像素)':>14} | {'到最近真实(像素)':>16} | 说明")
    print("  " + "-" * 78)
    for name, ds, dn in stats:
        if name.startswith('真实'):
            note = "基准"
        elif dn < 0.5:
            note = "≈复制训练样本"
        elif dn < 8:
            note = "有新像素，仍可辨认"
        else:
            note = "噪点化"
        print(f"  {name:<18} | {ds:>14.1f} | {dn:>16.1f} | {note}")
    print(f"\n  已保存: mix_compare.png  (整表)   mixrow_*.png  (单行)")


if __name__ == '__main__':
    main()
