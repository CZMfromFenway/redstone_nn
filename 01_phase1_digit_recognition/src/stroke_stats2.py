#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
MNIST 笔画粗细（修正版，含解析自检）
=======================================
口径: 对每个图案，用距离变换 d(p) = 点到最近背景的欧氏距离
      width = 面积 / 骨架长 ~= area / sum(1/max(d,1))
      等价地: width = 2 * area / sum( d / max(d,1)^2 )
  对半径 r 的圆形笔画，该式解析值 = 2.629*r （见自检），
  故再做一次比例校正 width / 2.629 * 2 = width * 0.7607 ... 
  为避免混淆，这里直接输出"参考量"并给出圆盘自检值，供解读。

实现: 自定义两遍 chamfer 距离变换（3-4 权重），无需 scipy/skimage。
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
A, B = 3, 4        # chamfer 权重（近似欧氏）
SA, SB = A / 3.0, B / 3.0


def load_idx(p):
    import struct
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def to15(im):
    n = len(im)
    a = im.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def chamfer(mask):
    """mask: (H,W) bool -> 距离变换 float (到最近 False 的近似欧氏距离)"""
    H, W = mask.shape
    INF = 1 << 20
    d = np.where(mask, INF, 0).astype(np.int64)
    # 边界外视为背景，把边界上的前景初始化为 1
    d[0, :] = np.minimum(d[0, :], A if True else INF)
    d[-1, :] = np.minimum(d[-1, :], A)
    d[:, 0] = np.minimum(d[:, 0], A)
    d[:, -1] = np.minimum(d[:, -1], A)
    # 前向
    for y in range(H):
        for x in range(W):
            if d[y, x] == 0:
                continue
            v = d[y, x]
            if y > 0:
                v = min(v, d[y-1, x] + A)
                if x > 0:
                    v = min(v, d[y-1, x-1] + B)
                if x < W - 1:
                    v = min(v, d[y-1, x+1] + B)
            if x > 0:
                v = min(v, d[y, x-1] + A)
            d[y, x] = v
    # 后向
    for y in range(H - 1, -1, -1):
        for x in range(W - 1, -1, -1):
            if d[y, x] == 0:
                continue
            v = d[y, x]
            if y < H - 1:
                v = min(v, d[y+1, x] + A)
                if x > 0:
                    v = min(v, d[y+1, x-1] + B)
                if x < W - 1:
                    v = min(v, d[y+1, x+1] + B)
            if x < W - 1:
                v = min(v, d[y, x+1] + A)
            d[y, x] = v
    return d / 3.0


def width_of(mask):
    """返回 (width_metric, area, skeleton_len)"""
    if mask.sum() == 0:
        return 0.0, 0, 0.0
    dt = chamfer(mask)
    dd = np.maximum(dt[mask], 1e-9)
    area = int(mask.sum())
    skel = float((1.0 / dd).sum())          # 骨架长度估计
    return area / skel, area, skel


def main():
    print("=" * 96)
    print("笔画粗细（修正版）")
    print("=" * 96)

    # ---- 自检：半径 r 的圆盘 ----
    print("\n[自检] 半径 r 的实心圆盘，理论 width = area/skel")
    print(f"  {'r':>4} | {'面积':>6} | {'骨架长':>8} | {'width':>7} | {'width/r':>8}")
    print("  " + "-" * 44)
    rr = []
    for r in (3, 4, 5, 6, 8, 10):
        n = 2 * r + 3
        yy, xx = np.mgrid[0:n, 0:n]
        m = ((yy - n // 2) ** 2 + (xx - n // 2) ** 2) <= r * r
        w, a, s = width_of(m)
        rr.append(w / r)
        print(f"  {r:>4} | {a:>6} | {s:>8.2f} | {w:>7.2f} | {w/r:>8.3f}")
    k = float(np.mean(rr))
    print(f"\n  -> width/r 平均 = {k:.4f}")
    print(f"  -> 真实半径 r_true = width / {k:.4f}，真实笔画宽度 = 2*r_true "
          f"= width * {2/k:.4f}")
    conv = 2.0 / k
    print(f"  -> 换算系数: 笔画宽度 = width * {conv:.4f}")

    tr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    te = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))

    out = {'conv': conv}
    for name, imgs, labs in [('train', tr, ytr), ('test', te, yte)]:
        print(f"\n{'-'*96}\n{name}\n{'-'*96}")
        B28 = imgs > 127
        ws = []; areas = []
        step = max(1, len(B28) // 4000)
        for m in B28[::step]:
            w, a, s = width_of(m)
            if a:
                ws.append(w); areas.append(a)
        ws = np.array(ws); areas = np.array(areas)
        print(f"  28x28 (>{127}), 抽样 {len(ws)} 张")
        print(f"    平均面积 {areas.mean():.1f} px")
        print(f"    width 均值 {ws.mean():.3f}  中位 {np.median(ws):.3f}")
        print(f"    => 28x28 上平均笔画宽度 {ws.mean()*conv:.2f} px "
              f"（中位 {np.median(ws)*conv:.2f}）")
        print(f"    分布: P10={np.percentile(ws,10)*conv:.2f} "
              f"P25={np.percentile(ws,25)*conv:.2f} "
              f"P50={np.percentile(ws,50)*conv:.2f} "
              f"P75={np.percentile(ws,75)*conv:.2f} "
              f"P90={np.percentile(ws,90)*conv:.2f}")
        out[name] = {'w28': float(ws.mean()), 'w28_conv': float(ws.mean()*conv),
                     'area28': float(areas.mean()),
                     'p10': float(np.percentile(ws,10)*conv),
                     'p50': float(np.percentile(ws,50)*conv),
                     'p90': float(np.percentile(ws,90)*conv)}

        # 15x15
        a15 = to15(imgs); B15 = a15 > 29.09
        ws15 = []
        for m in B15[::step]:
            w, a, s = width_of(m)
            if a:
                ws15.append(w)
        ws15 = np.array(ws15)
        print(f"  15x15 (阈值 29.09): 平均面积 "
              f"{B15.reshape(len(B15),-1).sum(1).mean():.1f} px")
        print(f"    width 均值 {ws15.mean():.3f} -> 笔画宽度 "
              f"{ws15.mean()*conv:.2f} px")
        out[name].update({'w15_conv': float(ws15.mean()*conv)})

        # 逐数字
        print(f"\n  逐数字（28x28）")
        print(f"    {'数字':>4} | {'面积':>6} | {'笔画宽度':>8}")
        print("    " + "-" * 26)
        for d in range(10):
            sel = B28[(labs == d)][::max(1, (labs == d).sum() // 400)]
            ww = [width_of(m)[0] for m in sel if m.sum()]
            am = [m.sum() for m in sel]
            print(f"    {d:>4} | {np.mean(am):>6.1f} | "
                  f"{np.mean(ww)*conv:>8.2f}")
            out.setdefault(f'{name}_digit{d}', {})['width'] = \
                float(np.mean(ww) * conv)

    json.dump(out, open(os.path.join(RES, 'stroke_thickness.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/stroke_thickness.json")


if __name__ == '__main__':
    main()
