#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 8x8 头像定制 16 色调色板
=============================
目标: 从本地 MC 方块纹理中挑 16 种颜色，使头像量化误差最小，
      同时保证 16 色能排成一条**有序路径**（红石单调阶梯需要）。

方法:
  1. 收集 MC 全部方块纹理的众数色（去重、去近似重复）作为候选池
  2. 用 k-means 求出头像的最优 16 色（上界参考）
  3. 从候选池中贪心选 16 色逼近 k-means 解
  4. 用"最近邻插入"把 16 色排成有序路径（贪心 TSP 风格）
  5. 报告量化误差，并与 MC16 混凝土 / 自由 16 色对比
"""
import os, json, numpy as np
from PIL import Image
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
MC = r'D:\czm我的世界\.minecraft'


# ---------------------------------------------------------------- 候选池
def collect_block_colors(max_per_dir=400):
    """从本地 MC 资源包收集方块纹理众数色"""
    cols = {}
    seen_dirs = 0
    for dp, dn, fn in os.walk(MC):
        if 'resourcepacks' not in dp and 'textures' not in dp:
            continue
        pngs = [f for f in fn if f.endswith('.png')]
        if not pngs:
            continue
        seen_dirs += 1
        for f in pngs:
            if not f.startswith(('white_', 'light_gray_', 'gray_', 'black_', 'brown_',
                                 'red_', 'orange_', 'yellow_', 'lime_', 'green_',
                                 'cyan_', 'light_blue_', 'blue_', 'purple_',
                                 'magenta_', 'pink_')):
                continue
            try:
                im = Image.open(os.path.join(dp, f)).convert('RGB')
                a = np.array(im).reshape(-1, 3)
                vals, cnt = np.unique(a, axis=0, return_counts=True)
                c = vals[cnt.argmax()]
                cols[tuple(int(x) for x in c)] = f
            except Exception:
                pass
        if seen_dirs > max_per_dir:
            break
    return cols


# ---------------------------------------------------------------- k-means
def kmeans(px, C, seed=0, iters=40):
    rng = np.random.default_rng(seed)
    cen = px[rng.choice(len(px), C, replace=False)].copy().astype(np.float64)
    for _ in range(iters):
        lab = ((px[:, None] - cen[None]) ** 2).sum(2).argmin(1)
        for k in range(C):
            m = lab == k
            if m.sum():
                cen[k] = px[m].mean(0)
    lab = ((px[:, None] - cen[None]) ** 2).sum(2).argmin(1)
    err = float(((px - cen[lab]) ** 2).sum(1).mean())
    return cen, err


def quant_err(px, pal):
    lab = ((px[:, None] - np.asarray(pal)[None]) ** 2).sum(2).argmin(1)
    return float(((px - np.asarray(pal)[lab]) ** 2).sum(1).mean())


# ---------------------------------------------------------------- 有序化
def order_path(P, start_idx=None):
    """最近邻插入法排成有序路径（贪心 TSP）"""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    if n <= 2:
        return list(range(n))
    used = np.zeros(n, bool)
    cur = 0 if start_idx is None else start_idx
    order = [cur]; used[cur] = True
    for _ in range(n - 1):
        d = ((P - P[cur]) ** 2).sum(1)
        d[used] = 1e18
        cur = int(d.argmin())
        order.append(cur); used[cur] = True
    # 2-opt 改善
    imp = True
    while imp:
        imp = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c, e = order[j], order[j + 1] if j + 1 < n else None
                if e is None:
                    continue
                d0 = np.linalg.norm(P[a] - P[b]) + np.linalg.norm(P[c] - P[e])
                d1 = np.linalg.norm(P[a] - P[c]) + np.linalg.norm(P[b] - P[e])
                if d1 + 1e-9 < d0:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    imp = True
    return order


def path_len(P, order):
    Q = P[order]
    return float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())


def main():
    print("=" * 100)
    print("为 8x8 头像定制 16 色调色板")
    print("=" * 100)
    F = np.load(os.path.join(DATA, 'faces8_rgb.npy')).astype(np.float64)
    px = F.reshape(-1, 3)
    print(f"头像像素: {px.shape}")

    pal = json.load(open(os.path.join(DATA, 'palettes.json')))
    mc16 = np.array(pal['pal16'], dtype=np.float64)
    print(f"\n[参照]")
    print(f"  MC16 混凝土      量化误差 = {quant_err(px, mc16):8.1f}")

    cen_free, err_free = kmeans(px, 16, seed=1)
    print(f"  自由 k-means 16  量化误差 = {err_free:8.1f}   <- 理论上界")

    # 候选池
    print(f"\n[收集 MC 方块色候选池]")
    cols = collect_block_colors()
    print(f"  候选色数: {len(cols)}")
    CAND = np.array(list(cols.keys()), dtype=np.float64)
    # 去近似重复
    keep = []
    for i, c in enumerate(CAND):
        ok = True
        for k in keep:
            if np.linalg.norm(c - CAND[k]) < 12:
                ok = False; break
        if ok:
            keep.append(i)
    CAND = CAND[keep]
    print(f"  去近似重复后: {len(CAND)}")

    # 贪心：从 k-means 中心出发，逐步用候选色替换以降低误差
    print(f"\n[贪心搜索 16 色]")
    chosen = list(range(len(cen_free)))
    cur_pal = cen_free.copy()
    best_err = err_free
    # 迭代: 每次尝试用一个候选色替换一个当前色
    for it in range(12):
        improved = False
        for k in range(16):
            best_local = (best_err, None)
            for ci in range(len(CAND)):
                trial = cur_pal.copy(); trial[k] = CAND[ci]
                e = quant_err(px, trial)
                if e < best_local[0] - 1e-9:
                    best_local = (e, CAND[ci])
            if best_local[1] is not None:
                cur_pal[k] = best_local[1]
                best_err = best_local[0]
                improved = True
        print(f"  轮 {it+1}: 量化误差 = {best_err:.1f}", flush=True)
        if not improved:
            break

    print(f"\n[结果]")
    print(f"  定制 16 色       量化误差 = {best_err:8.1f}  (RMS {np.sqrt(best_err/3):.1f})")
    print(f"  自由 k-means     量化误差 = {err_free:8.1f}  (RMS {np.sqrt(err_free/3):.1f})")
    print(f"  MC16 混凝土      量化误差 = {quant_err(px, mc16):8.1f}  "
          f"(RMS {np.sqrt(quant_err(px, mc16)/3):.1f})")
    print(f"  -> 定制比 MC 混凝土好 {quant_err(px, mc16)/best_err:.2f} 倍")

    # 排成有序路径
    order = order_path(cur_pal, start_idx=int(np.argmax(cur_pal.sum(1))))
    Q = cur_pal[order]
    print(f"\n[有序化] 最近邻 + 2opt 路径长度 = {path_len(cur_pal, order):.0f}")
    print(f"  相邻色距离: 最小 {np.linalg.norm(np.diff(Q,axis=0),axis=1).min():.0f}  "
          f"中位 {np.median(np.linalg.norm(np.diff(Q,axis=0),axis=1)):.0f}  "
          f"最大 {np.linalg.norm(np.diff(Q,axis=0),axis=1).max():.0f}")

    name_of = {tuple(v): k for k, v in cols.items()}
    print(f"\n[最终 16 色（按有序路径）]")
    print(f"  {'#':>3} | {'RGB':<18} | 最近的 MC 纹理")
    print("  " + "-" * 62)
    final = []
    for i, c in enumerate(Q):
        key = tuple(int(round(x)) for x in c)
        nm = name_of.get(key, '?')
        final.append([int(round(x)) for x in c])
        print(f"  {i:>3} | ({c[0]:>3.0f},{c[1]:>3.0f},{c[2]:>3.0f})      | {nm}")

    json.dump({'palette_ordered': final, 'quant_err': best_err,
               'free_kmeans_err': err_free, 'mc16_err': quant_err(px, mc16),
               'names': [name_of.get(tuple(int(round(x)) for x in c), '?') for c in Q]},
              open(os.path.join(DATA, 'palette_avatar16.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: data/palette_avatar16.json")


if __name__ == '__main__':
    main()
