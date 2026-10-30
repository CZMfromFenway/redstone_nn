#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
笔画宽度：用合成笔画标定（替代错误的圆盘标定）
==================================================
方法:
  1. 取 MNIST 里中等复杂度的真实数字，提取其骨架（用形态学细化）
  2. 以该骨架为"中心线"，用已知半径 r 生成合成笔画
     (形态学膨胀 r 次, 结构元 = 近似圆盘)  -> 已知宽度的笔画
  3. 对合成笔画算 mean(d) 与 area/sum(1/d)
  4. 找出使 mean(d) 最接近真值的线性关系

实际测量:
  T_mean = 2 * mean(d) * k       （k 由标定得出）
也可直接报告 mean(d)（到最近背景的平均距离），它在几何上很直观。
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'


def load_idx(p):
    import struct
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def chamfer(mask):
    H, W = mask.shape
    INF = 1 << 20
    d = np.where(mask, INF, 0).astype(np.int64)
    d[0, :] = np.minimum(d[0, :], 3); d[-1, :] = np.minimum(d[-1, :], 3)
    d[:, 0] = np.minimum(d[:, 0], 3); d[:, -1] = np.minimum(d[:, -1], 3)
    for y in range(H):
        for x in range(W):
            if d[y, x] == 0: continue
            v = d[y, x]
            if y > 0:
                v = min(v, d[y-1, x] + 3)
                if x > 0: v = min(v, d[y-1, x-1] + 4)
                if x < W-1: v = min(v, d[y-1, x+1] + 4)
            if x > 0: v = min(v, d[y, x-1] + 3)
            d[y, x] = v
    for y in range(H-1, -1, -1):
        for x in range(W-1, -1, -1):
            if d[y, x] == 0: continue
            v = d[y, x]
            if y < H-1:
                v = min(v, d[y+1, x] + 3)
                if x > 0: v = min(v, d[y+1, x-1] + 4)
                if x < W-1: v = min(v, d[y+1, x+1] + 4)
            if x < W-1: v = min(v, d[y, x+1] + 3)
            d[y, x] = v
    return d / 3.0


DISK = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)
        if dy*dy + dx*dx <= 9]


def dilate(m, times):
    out = m.copy()
    for _ in range(times):
        nxt = out.copy()
        ys, xs = np.nonzero(out)
        for y, x in zip(ys, xs):
            for dy, dx in DISK:
                yy, xx = y+dy, x+dx
                if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]:
                    nxt[yy, xx] = True
        out = nxt
    return out


def thin(m):
    """Zhang-Suen 细化"""
    img = m.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2 = P[0:-2, 1:-1]; p3 = P[0:-2, 2:]; p4 = P[1:-1, 2:]
            p5 = P[2:, 2:]; p6 = P[2:, 1:-1]; p7 = P[2:, 0:-2]
            p8 = P[1:-1, 0:-2]; p9 = P[0:-2, 0:-2]
            B = p2+p3+p4+p5+p6+p7+p8+p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = np.zeros_like(B)
            for i in range(8):
                A += ((seq[i] == 0) & (seq[i+1] == 1)).astype(np.uint8)
            if step == 0:
                cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & \
                       ((p2*p4*p6) == 0) & ((p4*p6*p8) == 0)
            else:
                cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & \
                       ((p2*p4*p8) == 0) & ((p2*p6*p8) == 0)
            if cond.any():
                img[cond] = 0
                changed = True
    return img.astype(bool)


def main():
    tr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    B = tr[:200] > 127
    print("=" * 96)
    print("合成笔画标定")
    print("=" * 96)

    # 取 20 个真实数字做骨架 -> 合成笔画
    skels = []
    for m in B:
        if m.sum() < 40 or m.sum() > 200:
            continue
        s = thin(m)
        if s.sum() >= 10:
            skels.append(s)
        if len(skels) >= 20:
            break
    print(f"取样骨架 {len(skels)} 个，骨架长度 "
          f"{np.mean([s.sum() for s in skels]):.1f}")

    print(f"\n  {'膨胀次数':>8} | {'真实宽度':>8} | {'area':>6} | {'mean(d)':>8} | "
          f"{'2*mean(d)':>10} | {'area/sum(1/d)':>14}")
    print("  " + "-" * 70)
    calib = []
    for it in (0, 1, 2, 3):
        tw = 2 * it + 1
        md, wid = [], []
        for s in skels:
            syn = dilate(s, it)
            dt = chamfer(syn)
            md.append(dt[syn].mean())
            dd = np.maximum(dt[syn], 1e-9)
            wid.append(syn.sum() / (1.0 / dd).sum())
        mean_d = float(np.mean(md)); mean_w = float(np.mean(wid))
        print(f"  {it:>8} | {tw:>8} | "
              f"{np.mean([dilate(s,it).sum() for s in skels]):>6.0f} | "
              f"{mean_d:>8.3f} | {2*mean_d:>10.3f} | {mean_w:>14.3f}")
        calib.append((tw, mean_d, mean_w))

    # 线性拟合: 真实宽度 ~ a * 2*mean(d) + b
    X = np.array([[2*c[1], 1.0] for c in calib])
    Y = np.array([c[0] for c in calib])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    print(f"\n  拟合: 真实宽度 = {coef[0]:.4f} * 2*mean(d) + {coef[1]:.4f}")
    pred = X @ coef
    print(f"  拟合误差: " + ", ".join(f"{p:.2f} vs {y}" for p, y in zip(pred, Y)))

    X2 = np.array([[c[2], 1.0] for c in calib])
    coef2, *_ = np.linalg.lstsq(X2, Y, rcond=None)
    print(f"  拟合: 真实宽度 = {coef2[0]:.4f} * area/sum(1/d) + {coef2[1]:.4f}")

    # ---- 应用到 MNIST ----
    print(f"\n{'='*96}\nMNIST 实际测量\n{'='*96}")
    out = {}
    for name, imgs in [('train', tr[:8000]), ('test', load_idx(
            os.path.join(RAW, 't10k-images-idx3-ubyte'))) ]:
        Bm = imgs > 127
        md_all = []
        for m in Bm:
            if m.sum() == 0: continue
            dt = chamfer(m)
            md_all.append(dt[m].mean())
        md_all = np.array(md_all)
        w1 = coef[0] * 2 * md_all + coef[1]
        w2 = coef2[0] * (md_all * 0)  # 占位
        print(f"\n  {name}: {len(md_all)} 张")
        print(f"    mean(d) 均值 {md_all.mean():.3f}  "
              f"中位 {np.median(md_all):.3f}")
        print(f"    -> 2*mean(d) = {2*md_all.mean():.3f}")
        print(f"    -> 线性标定后笔画宽度 = {w1.mean():.2f} px  "
              f"(中位 {np.median(w1):.2f})")
        print(f"       分布 P10={np.percentile(w1,10):.2f} "
              f"P25={np.percentile(w1,25):.2f} "
              f"P50={np.percentile(w1,50):.2f} "
              f"P75={np.percentile(w1,75):.2f} "
              f"P90={np.percentile(w1,90):.2f}")
        print(f"    -> 简化口径 2*mean(d) = {2*md_all.mean():.2f} px")
        out[name] = {'mean_d': float(md_all.mean()),
                     'width_calib': float(w1.mean()),
                     'width_2meand': float(2*md_all.mean()),
                     'p10': float(np.percentile(w1,10)),
                     'p50': float(np.percentile(w1,50)),
                     'p90': float(np.percentile(w1,90))}

    out['calib'] = {'coef': [float(coef[0]), float(coef[1])],
                    'points': [[float(c[0]), float(c[1]), float(c[2])]
                               for c in calib]}
    json.dump(out, open(os.path.join(RES, 'stroke_width2.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/stroke_width2.json")


if __name__ == '__main__':
    main()
