#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
反对称分量的可行性分析
========================
分解（按调色板索引，索引有序故差值有颜色意义）:
  左半 s_i = x_i                 (i 在左半 32 个位置)
  右半 d_i = x_{M(i)} - x_i      (反对称差)

分析:
  1. d 的分布（取值、占比）        -> 需要几个层级
  2. 用 ±k 个层级表达 d 的覆盖率     -> 红石成本
  3. 三种重建的误差对比:
     (a) 原始 64 像素（无损基准）
     (b) 强制对称 (d=0)
     (c) 量化 d 到 ±k 层
  4. d 与 s 的相关性（能否独立编码）
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16

# 左半 32 个位置 与 其镜像位置
LEFT = [(r, c) for r in range(8) for c in range(4)]
LIDX = np.array([r * 8 + c for r, c in LEFT], dtype=int)
RIDX = np.array([r * 8 + (7 - c) for r, c in LEFT], dtype=int)
# 中线（c=3 和 c=4 互为镜像，属于同一对）——注意 c=3 与 c=4 是镜像对
print("镜像对示例:", list(zip(LIDX[:6], RIDX[:6])))


def main():
    print("=" * 96)
    print("反对称分量 d = 右 - 左 的可行性分析")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    n = len(lab)
    print(f"数据: {lab.shape}   调色板 {C} 色（索引有序）")

    s = lab[:, LIDX]                       # (n,32) 左半
    xr = lab[:, RIDX]                      # (n,32) 右半
    d = xr - s                             # (n,32) 反对称差
    print(f"\nd 的范围: [{d.min()}, {d.max()}]   |d| 均值 {np.abs(d).mean():.3f}")

    # ---- 1. d 的分布 ----
    vals, cnt = np.unique(d, return_counts=True)
    tot = d.size
    print(f"\n[1] d 的取值分布（共 {len(vals)} 种）")
    print(f"  {'d值':>5} | {'出现次数':>10} | {'占比':>8} | 累计")
    print("  " + "-" * 42)
    order = np.argsort(-cnt)
    cum = 0.0
    top = []
    for k in order[:16]:
        v, c = vals[k], cnt[k]
        cum += c / tot
        print(f"  {v:>5} | {c:>10} | {c/tot:>7.3%} | {cum:>7.3%}")
        top.append((int(v), int(c)))

    # ---- 2. 用 ±k 层级覆盖多少 ----
    print(f"\n[2] 若只保留 d ∈ [-k, +k]，覆盖率与重建损失")
    print(f"  {'k':>3} | {'覆盖比例':>9} | {'截断后平均绝对误差':>20} | 说明")
    print("  " + "-" * 62)
    rows_k = []
    for k in (0, 1, 2, 3, 5, 8, 15):
        mask = np.abs(d) <= k
        cov = float(mask.mean())
        dq = np.clip(d, -k, k)
        mae = float(np.abs(d - dq).mean())
        note = "强制对称" if k == 0 else ("建议" if k in (2, 3) else "")
        print(f"  {k:>3} | {cov:>8.2%} | {mae:>20.4f} | {note}")
        rows_k.append({'k': k, 'cover': cov, 'mae': mae})

    # ---- 3. 重建误差对比（还原成 RGB 后的 MSE）----
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.float64)
    F = np.load(os.path.join(DATA, 'faces8_rgb.npy')).astype(np.float64)
    print(f"\n[3] 在 RGB 空间的重建误差（对比原始头像）")
    print(f"  {'方案':<28} | {'像素MSE':>9} | {'RMS/255':>8} | {'逐像素准确率':>10}")
    print("  " + "-" * 66)

    def rebuild_sym_only(si):
        """强制对称: 左右都用左半"""
        X = np.zeros((len(si), 64), dtype=np.int64)
        X[:, LIDX] = si
        X[:, RIDX] = si
        return X

    def rebuild_with_d(si, di):
        X = np.zeros((len(si), 64), dtype=np.int64)
        X[:, LIDX] = si
        X[:, RIDX] = np.clip(si + di, 0, C - 1)
        return X

    def score(X):
        rec = pal[np.clip(X, 0, C - 1)].reshape(-1, 8, 8, 3)
        mse = float(((rec - F) ** 2).mean())
        acc = float((X == lab).mean())
        return mse, acc

    # (a) 无损
    mse_a, acc_a = score(lab)
    print(f"  {'(a) 原始 64 像素（基准）':<28} | {mse_a:>9.2f} | {np.sqrt(mse_a/3):>8.2f} | {acc_a:>10.4f}")
    for k in (0, 1, 2, 3, 5, 15):
        X = rebuild_with_d(s, np.clip(d, -k, k))
        mse, acc = score(X)
        tag = f"(c) d 量化到 ±{k}" if k > 0 else "(b) 强制对称 d=0"
        print(f"  {tag:<28} | {mse:>9.2f} | {np.sqrt(mse/3):>8.2f} | {acc:>10.4f}")

    # ---- 4. 独立性: d 与 s 的相关 ----
    print(f"\n[4] d 与 s 是否独立（决定能否分开编码）")
    # 逐位置的互信息近似：用相关系数
    cors = []
    for i in range(32):
        a, b = s[:, i].astype(float), d[:, i].astype(float)
        if a.std() > 1e-9 and b.std() > 1e-9:
            cors.append(abs(np.corrcoef(a, b)[0, 1]))
    print(f"  逐位置 |corr(s_i, d_i)| 均值 = {np.mean(cors):.3f}  "
          f"(0=完全独立，1=完全相关)")
    print(f"  -> {'可以分开编码' if np.mean(cors) < 0.3 else '有耦合，需注意'}")

    # ---- 5. 用一个共享的 d 分布 ----
    print(f"\n[5] 若所有位置共用同一个 d 分布（简化红石）")
    dv = d.reshape(-1)
    print(f"  全局 d 的 5 个最常见值: {[int(v) for v in vals[np.argsort(-cnt)[:5]]]}")
    print(f"  全局 |d|=0 占比 {np.mean(dv==0):.3%}   |d|<=1 占比 {np.mean(np.abs(dv)<=1):.3%}  "
          f"|d|<=2 占比 {np.mean(np.abs(dv)<=2):.3%}")
    # 若每个位置都用自己的众数 d
    mode_d = np.zeros(32, dtype=np.int64)
    for i in range(32):
        vv, cc = np.unique(d[:, i], return_counts=True)
        mode_d[i] = vv[cc.argmax()]
    X = rebuild_with_d(s, np.tile(mode_d, (len(s), 1)))
    mse, acc = score(X)
    print(f"  用每位置的众数 d 重建: MSE {mse:.2f}  RMS {np.sqrt(mse/3):.2f}  "
          f"逐像素准确率 {acc:.4f}")

    out = {'range': [int(d.min()), int(d.max())],
           'abs_mean': float(np.abs(d).mean()),
           'levels': rows_k,
           'corr_mean': float(np.mean(cors)),
           'dist': [[int(v), int(c)] for v, c in top]}
    json.dump(out, open(os.path.join(RES, 'antisym_analysis.json'), 'w'), indent=2)
    print(f"\n已写出: results/antisym_analysis.json")


if __name__ == '__main__':
    main()
