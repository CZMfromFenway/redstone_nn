#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
MNIST 笔画粗细统计（训练集 + 测试集）
========================================
在 28x28 与 15x15（本模型用的下采样）两个分辨率上统计。

三种口径:
  A. 面积/骨架长  : 点亮像素数 / 骨架像素数  -> 平均直径（像素）
  B. 距离变换均值 : 每个点亮像素到最近背景的距离，均值 x 2
  C. 细化后骨架数 : 直接用 skeletonize 数骨架长度（若无 skimage 则用 A 近似）
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


def to15(im):
    n = len(im)
    a = im.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def try_scipy():
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt
    except Exception:
        return None


def skeleton_len_approx(mask):
    """无 skimage 时的骨架长度近似: 用击中击不中做一次细化迭代的替代
    这里用更稳的口径: 周长/2 的估计
    返回 (骨架长度估计)
    """
    # 用 4-连通边界数近似骨架: 每个 2x2 邻域内 (0,1) 或 (1,0) 的对角对
    m = mask.astype(np.uint8)
    a = m[:-1, :-1]; b = m[:-1, 1:]; c = m[1:, :-1]; d = m[1:, 1:]
    # 骨架长度 ~ 边界长度/2 ; 用周长近似
    peri = 0
    for x, y in ((a, b), (a, c), (b, d), (c, d)):
        peri += int((x != y).sum())
    return peri / 2.0


def main():
    print("=" * 96)
    print("MNIST 笔画粗细统计")
    print("=" * 96)
    tr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    te = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    print(f"训练集 {tr.shape}  测试集 {te.shape}")

    edt = try_scipy()
    print(f"scipy 距离变换可用: {edt is not None}")

    out = {}
    for name, imgs, labs, thr in [('train', tr, ytr, None), ('test', te, yte, None)]:
        print(f"\n{'-'*96}\n{name}\n{'-'*96}")
        # ---- 28x28 ----
        B28 = imgs > 127
        lit_mean = B28.reshape(len(B28), -1).sum(1).mean()
        print(f"  28x28: 平均点亮像素 {lit_mean:.1f} / 784 "
              f"({lit_mean/784*100:.1f}%)")
        # 距离变换
        if edt is not None:
            dt_vals = []
            for i in range(0, len(B28), 500):
                for m in B28[i:i+500]:
                    if m.sum() == 0:
                        continue
                    dt = edt(~m)
                    dt_vals.append(dt[m].mean())
            dt_mean = float(np.mean(dt_vals))
            print(f"  28x28: 距离变换均值 {dt_mean:.3f} px  -> 平均笔画宽度 "
                  f"~{2*dt_mean:.2f} px")
        # 面积/骨架
        sk = [skeleton_len_approx(m) for m in B28[:3000]]
        sk_mean = float(np.mean(sk))
        area_mean = float(B28[:3000].reshape(3000, -1).sum(1).mean())
        print(f"  28x28: 骨架长(周长/2)均值 {sk_mean:.2f}  "
              f"面积 {area_mean:.1f}  -> 面积/骨架 = {area_mean/sk_mean:.2f} px")

        # ---- 15x15（本模型输入）----
        a15 = to15(imgs)
        thr15 = to15(tr).mean()
        B15 = a15 > thr15
        lit15 = B15.reshape(len(B15), -1).sum(1).mean()
        print(f"  15x15: 二值化阈值 {thr15:.2f}")
        print(f"  15x15: 平均点亮像素 {lit15:.1f} / 225 ({lit15/225*100:.1f}%)")
        if edt is not None:
            dv = []
            for i in range(0, len(B15), 1000):
                for m in B15[i:i+1000]:
                    if m.sum() == 0:
                        continue
                    dt = edt(~m)
                    dv.append(dt[m].mean())
            d15 = float(np.mean(dv))
            print(f"  15x15: 距离变换均值 {d15:.3f} px -> 平均笔画宽度 "
                  f"~{2*d15:.2f} px")
        sk15 = [skeleton_len_approx(m) for m in B15[:3000]]
        skm15 = float(np.mean(sk15))
        am15 = float(B15[:3000].reshape(3000, -1).sum(1).mean())
        print(f"  15x15: 面积/骨架 = {am15/skm15:.2f} px")

        out[name] = {'lit28': float(lit_mean), 'lit15': float(lit15),
                     'thr15': float(thr15)}

    # 逐数字
    print(f"\n{'-'*96}\n逐数字（训练集, 28x28 > 127）\n{'-'*96}")
    B28 = tr > 127
    print(f"  {'数字':>4} | {'样本':>6} | {'平均点亮':>8} | {'面积/骨架':>9}")
    print("  " + "-" * 40)
    for d in range(10):
        sel = B28[ytr == d]
        a = sel.reshape(len(sel), -1).sum(1).mean()
        sk = np.mean([skeleton_len_approx(m) for m in sel[:500]])
        print(f"  {d:>4} | {len(sel):>6} | {a:>8.1f} | {a/sk:>9.2f}")
        out[f'digit{d}'] = {'train_lit28': float(a), 'ratio': float(a/sk)}

    json.dump(out, open(os.path.join(RES, 'stroke_thickness.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/stroke_thickness.json")


if __name__ == '__main__':
    main()
