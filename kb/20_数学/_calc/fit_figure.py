# -*- coding: utf-8 -*-
"""重画"拟合效果"图：同一个应用的 4 个通道，一张图 4 行。

数据来源与实验时一致：
    目标直方图  <- 02_phase2_avatar/webapp/styles_gauss.json 里的 (mu, chol) 采样
    模型 pmf    <- results/direct_shape_spec.json 里的 (M, c, w, abc) 卷积

用法：python kb/20_数学/_calc/fit_figure.py
输出：kb/figures/dropper_fit_shape.png
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

ROOT = r'E:\redstone_nn'
P2 = os.path.join(ROOT, '02_phase2_avatar', 'webapp')
SPEC = os.path.join(ROOT, '03_phase3_quantized_network', 'results',
                    'direct_shape_spec.json')
OUT = os.path.join(ROOT, 'kb', 'figures', 'dropper_fit_4ch.png')

STYLE = '少女'          # 这一页要举的那个应用
DELTA, NB, NG = 7.0, 256, 400_000


def target_hist(mu, chol, d, rng):
    """按与实验相同的口径采出第 d 个通道的目标直方图。"""
    x = rng.standard_normal((NG, 4)) @ np.asarray(chol).T + np.asarray(mu)
    xa = x[np.all(np.abs(x) <= 3.5, axis=1)]
    X = np.clip(np.floor((xa + 3.5) / DELTA * NB), 0, NB - 1).astype(int)
    p = np.bincount(X[:, d], minlength=NB).astype(float)
    return p / p.sum()


def model_pmf(ch):
    """三层循环等价于迭代卷积：每个投掷器把分布复制三份平移。"""
    cur = np.array([1.0])
    for c in ch['channels']:
        w, p = c['w'], np.array(c['abc']) / 9.0
        nxt = np.zeros(len(cur) + 2 * w)
        for j in range(3):
            if p[j] > 0:
                nxt[j * w:j * w + len(cur)] += cur * p[j]
        cur = nxt
    return cur


def lift(pk, c):
    """平移 c 并截断到 [0, NB)。"""
    out = np.zeros(NB)
    idx = c + np.arange(len(pk))
    ok = (idx >= 0) & (idx < NB)
    out[idx[ok]] = pk[ok]
    if (idx < 0).any():
        out[0] += pk[idx < 0].sum()
    if (idx >= NB).any():
        out[NB - 1] += pk[idx >= NB].sum()
    return out


def main():
    styles = json.load(open(os.path.join(P2, 'styles_gauss.json'),
                            encoding='utf-8'))['styles']
    spec = json.load(open(SPEC, encoding='utf-8'))['styles']
    e = [s for s in styles if s['name'] == STYLE][0]
    rng = np.random.default_rng(19)

    rows, info = [], []
    for d, ch in enumerate(spec[STYLE]):
        tgt = target_hist(e['mu'], e['chol'], d, rng)
        mod = lift(model_pmf(ch), ch['c'])
        k = np.arange(NB)
        mu_t = (k * tgt).sum()
        sd_t = np.sqrt(((k - mu_t) ** 2 * tgt).sum())
        mu_m = (k * mod).sum()
        sd_m = np.sqrt(((k - mu_m) ** 2 * mod).sum())
        j = np.abs(np.cumsum(mod) - np.cumsum(tgt)).max()
        rows.append((f"ch{d+1}  M={ch['M']:2d}  c={ch['c']:+4d}  J={j:.4f}",
                     tgt, mod))
        info.append((d + 1, mu_t, sd_t, mu_m, sd_m, ch['M'], ch['c'], j))

    W, H = 1000, 240 * len(rows)
    im = Image.new('RGB', (W, H), (20, 20, 24))
    dr = ImageDraw.Draw(im)
    for r, (title, tgt, mod) in enumerate(rows):
        y0 = r * 240
        dr.text((10, y0 + 4), title, fill=(235, 235, 245))
        pw, ph = 620, 170
        bx, by = 20, y0 + 26
        mx = max(tgt.max(), mod.max()) * 1.1
        for xx in range(pw):
            x = int(xx / pw * NB)
            h1 = int(tgt[x] / mx * ph)
            h2 = int(mod[x] / mx * ph)
            if h1:
                dr.line([(bx + xx, by + ph), (bx + xx, by + ph - h1)],
                        fill=(150, 150, 160))
            if h2:
                dr.line([(bx + xx, by + ph), (bx + xx, by + ph - h2)],
                        fill=(90, 190, 255))
        dr.rectangle([bx, by, bx + pw, by + ph], outline=(70, 70, 80))
        dr.text((bx + 4, by + ph + 4), 'grey = target   blue = model',
                fill=(170, 170, 180))
        cx0, cy0, cw, chh = 690, by, 290, ph
        dc = np.cumsum(mod) - np.cumsum(tgt)
        for xx in range(cw):
            x = int(xx / cw * NB)
            v = dc[x]
            yy = cy0 + chh // 2 - int(v / 0.05 * (chh // 2))
            yy = max(cy0, min(cy0 + chh, yy))
            dr.line([(cx0 + xx, cy0 + chh // 2), (cx0 + xx, yy)],
                    fill=(255, 120, 120))
        dr.rectangle([cx0, cy0, cx0 + cw, cy0 + chh], outline=(70, 70, 80))
        dr.line([(cx0, cy0 + chh // 2), (cx0 + cw, cy0 + chh // 2)],
                fill=(60, 60, 70))
        dr.text((cx0 + 4, cy0 + chh + 4), 'CDF diff  (full = +/-0.05)',
                fill=(170, 170, 180))
    im.save(OUT)

    print(f'{STYLE}：4 个通道')
    print(f"{'ch':>3} {'目标 μ':>8} {'目标 σ':>7} {'模型 μ':>8} {'模型 σ':>7} "
          f"{'M':>3} {'c':>5} {'J':>7}")
    for (d, mt, st, mm, sm, M, c, j) in info:
        print(f'{d:>3} {mt:8.1f} {st:7.1f} {mm:8.1f} {sm:7.1f} {M:>3} {c:>5} {j:7.4f}')
    print(f'写出 {OUT}')
    print('MAXJ', max(i[7] for i in info))


if __name__ == '__main__':
    main()
