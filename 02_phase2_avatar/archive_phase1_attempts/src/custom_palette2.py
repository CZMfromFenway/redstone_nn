#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
头像定制 16 色 v2：k-means 最优 + 吸附到最近 MC 方块色
=========================================================
v1 的错误: 从 103 个候选里"换色"搜误差 —— 候选池太小且目标不对。
v2 的做法:
  1. 对头像像素做 k-means(16) -> 得到最优 16 色（量化误差下界）
  2. 扩大候选池: 收集**全部** MC 方块/物品纹理的众数色
  3. 把每个 k-means 中心吸附到最近的候选色（保证 MC 可获取）
  4. 报告吸附前后的误差差
  5. 排成有序路径（最近邻 + 2opt）
"""
import os, json, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
MC = r'D:\czm我的世界\.minecraft'


def collect_all_colors(limit_dirs=2000):
    """收集全部方块/物品纹理的众数色"""
    cols = {}
    nd = 0
    for dp, dn, fn in os.walk(MC):
        if 'textures' not in dp:
            continue
        pngs = [f for f in fn if f.endswith('.png')]
        if not pngs:
            continue
        nd += 1
        for f in pngs:
            try:
                im = Image.open(os.path.join(dp, f)).convert('RGB')
                if im.size[0] < 4 or im.size[1] < 4:
                    continue
                a = np.array(im).reshape(-1, 3)
                vals, cnt = np.unique(a, axis=0, return_counts=True)
                # 取前 3 个众数色（纹理常有几种主色）
                for k in np.argsort(-cnt)[:3]:
                    c = tuple(int(x) for x in vals[k])
                    if c not in cols:
                        cols[c] = os.path.splitext(f)[0]
            except Exception:
                pass
        if nd > limit_dirs:
            break
    return cols


def kmeans(px, C, seed=0, iters=50):
    rng = np.random.default_rng(seed)
    cen = px[rng.choice(len(px), C, replace=False)].copy().astype(np.float64)
    for _ in range(iters):
        lab = ((px[:, None] - cen[None]) ** 2).sum(2).argmin(1)
        for k in range(C):
            m = lab == k
            if m.sum():
                cen[k] = px[m].mean(0)
    lab = ((px[:, None] - cen[None]) ** 2).sum(2).argmin(1)
    return cen, float(((px - cen[lab]) ** 2).sum(1).mean())


def quant_err(px, pal):
    pal = np.asarray(pal, dtype=np.float64)
    lab = ((px[:, None] - pal[None]) ** 2).sum(2).argmin(1)
    return float(((px - pal[lab]) ** 2).sum(1).mean())


def order_path(P, start_idx=0):
    P = np.asarray(P, dtype=np.float64); n = len(P)
    used = np.zeros(n, bool); cur = start_idx; order = [cur]; used[cur] = True
    for _ in range(n - 1):
        d = ((P - P[cur]) ** 2).sum(1); d[used] = 1e18
        cur = int(d.argmin()); order.append(cur); used[cur] = True
    imp = True
    while imp:
        imp = False
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                a, b, c, e = order[i-1], order[i], order[j], order[j+1]
                d0 = np.linalg.norm(P[a]-P[b]) + np.linalg.norm(P[c]-P[e])
                d1 = np.linalg.norm(P[a]-P[c]) + np.linalg.norm(P[b]-P[e])
                if d1 + 1e-9 < d0:
                    order[i:j+1] = order[i:j+1][::-1]; imp = True
    return order


def main():
    print("=" * 100)
    print("头像定制 16 色 v2：k-means 最优 + 吸附到 MC 方块色")
    print("=" * 100)
    F = np.load(os.path.join(DATA, 'faces8_rgb.npy')).astype(np.float64)
    px = F.reshape(-1, 3)
    pal = json.load(open(os.path.join(DATA, 'palettes.json')))
    mc16 = np.array(pal['pal16'], dtype=np.float64)
    e_mc = quant_err(px, mc16)
    print(f"MC16 混凝土        量化误差 = {e_mc:8.1f}  (RMS {np.sqrt(e_mc/3):.1f})")

    cen, e_km = kmeans(px, 16, seed=1)
    print(f"自由 k-means 16    量化误差 = {e_km:8.1f}  (RMS {np.sqrt(e_km/3):.1f})  <- 下界")

    print(f"\n[收集 MC 全色候选池]")
    cols = collect_all_colors()
    CAND = np.array(list(cols.keys()), dtype=np.float64)
    print(f"  候选色数: {len(CAND)}")

    # 吸附: 每个中心 -> 最近候选色
    snapped = []
    for c in cen:
        i = int(((CAND - c) ** 2).sum(1).argmin())
        snapped.append(CAND[i])
    snapped = np.array(snapped)
    e_snap = quant_err(px, snapped)
    print(f"\n[吸附后] 量化误差 = {e_snap:8.1f}  (RMS {np.sqrt(e_snap/3):.1f})")
    print(f"  吸附代价: +{e_snap-e_km:.1f} ({(e_snap/e_km-1)*100:.1f}%)")
    print(f"  vs MC16 混凝土: 好 {e_mc/e_snap:.2f} 倍")

    # 去重（可能两个中心吸到同一色）
    uniq = []
    for c in snapped:
        if not any(np.linalg.norm(c - u) < 1e-6 for u in uniq):
            uniq.append(c)
    print(f"  去重后颜色数: {len(uniq)}")

    # 若不足 16 色，用 k-means 中心补
    P = np.array(uniq)
    if len(P) < 16:
        extra = []
        for c in cen:
            if len(P) + len(extra) >= 16:
                break
            extra.append(c)
        P = np.vstack([P, np.array(extra)[:16-len(P)]])

    order = order_path(P, start_idx=int(np.argmax(P.sum(1))))
    Q = P[order]
    names = [cols.get(tuple(int(round(x)) for x in c), '?') for c in Q]
    print(f"\n[有序化] 路径长度 = {np.linalg.norm(np.diff(Q,axis=0),axis=1).sum():.0f}")
    adj = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    print(f"  相邻距离: 最小 {adj.min():.0f} 中位 {np.median(adj):.0f} 最大 {adj.max():.0f}")

    print(f"\n[最终 16 色（有序）]")
    print(f"  {'#':>3} | {'RGB':<18} | 最近的 MC 纹理")
    print("  " + "-" * 60)
    for i, (c, nm) in enumerate(zip(Q, names)):
        print(f"  {i:>3} | ({c[0]:>3.0f},{c[1]:>3.0f},{c[2]:>3.0f})      | {nm}")

    json.dump({'palette_ordered': [[int(round(x)) for x in c] for c in Q],
               'names': names, 'quant_err': e_snap, 'free_kmeans_err': e_km,
               'mc16_err': e_mc},
              open(os.path.join(DATA, 'palette_avatar16.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: data/palette_avatar16.json")


if __name__ == '__main__':
    main()
