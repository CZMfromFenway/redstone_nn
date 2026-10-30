#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
蒸馏网络的正确用法：精修 / 循环生成
======================================
训练配置回顾: 输入=真实头像, 标签=教师从该头像邻域采样的颜色
  -> 学生学的是 "给定一张头像, 输出教师的细化结果"

用法 A (精修): 输入 = 近邻法生成的图 -> 学生输出更干净的结果
用法 B (循环): 反复应用学生（每次输入上一轮输出），看是否收敛到稳定头像
用法 C (温度扫描): 在用法 B 下扫温度

输出:
  distill_use_<档位>_refine.png    A: 近邻图 -> 精修
  distill_use_<档位>_iter<N>.png   B: 第 N 轮迭代结果
  distill_use_<档位>_real.png      真实对照
"""
import os, json, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16
LIDX = np.array([r * 8 + c for r in range(8) for c in range(4)], dtype=int)
RIDX = np.array([r * 8 + (7 - c) for r in range(8) for c in range(4)], dtype=int)
K_NB = 30; TEMP_T = 5.0


def net_input(x_flat, pos):
    xn = (x_flat.astype(np.float32) - (C - 1) / 2) / ((C - 1) / 2)
    ph = np.zeros((len(pos), 64), dtype=np.float32)
    ph[np.arange(len(pos)), pos] = 1.0
    return np.concatenate([xn, ph], 1)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'd15'
    dclip = 3 if tag == 'd3' else 15
    print("=" * 92)
    print(f"蒸馏网络的使用（精修/循环）  {tag}")
    print("=" * 92)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    j = json.load(open(os.path.join(RES, f'distill_{tag}_net.json')))
    W1 = np.array(j['W1'], dtype=np.float32); b1 = np.array(j['b1'], dtype=np.float32)
    W2 = np.array(j['W2'], dtype=np.float32); b2 = np.array(j['b2'], dtype=np.float32)
    print(f"载入网络 W1{W1.shape} W2{W2.shape}")

    # 教师（近邻）
    rng0 = np.random.default_rng(0)
    POOL = 20000
    sel = rng0.choice(len(lab), POOL, replace=False)
    S = lab[sel][:, LIDX]; D = lab[sel][:, RIDX] - lab[sel][:, LIDX]

    def teacher_gen(pi, seed, temp=TEMP_T):
        rr = np.random.default_rng(seed)
        ds = np.empty(POOL, dtype=np.int32); dd = np.empty(POOL, dtype=np.int32)
        for t in range(0, POOL, 5000):
            ds[t:t+5000] = (S[t:t+5000] != S[pi]).sum(1)
            dd[t:t+5000] = (D[t:t+5000] != D[pi]).sum(1)
        dist = ds + dd
        nb = np.argpartition(dist, K_NB)[:K_NB]
        w = np.exp(-dist[nb] / max(temp, 1e-6)); w /= w.sum()
        cum = np.cumsum(w)
        pick = np.clip((rr.random(32)[:, None] > cum[None, :]).sum(1), 0, K_NB - 1)
        sv = S[nb][pick, np.arange(32)]
        dv = np.clip(D[nb][pick, np.arange(32)], -dclip, dclip)
        X = np.zeros(64, dtype=np.int64)
        X[LIDX] = sv; X[RIDX] = np.clip(sv + dv, 0, C - 1)
        return X

    def student_refine(X, temp=1.0, steps=1, seed=0):
        """把 X 交给学生，逐位置刷新（可多轮）"""
        r = np.random.default_rng(seed)
        cur = X.copy()
        for _ in range(steps):
            new = cur.copy()
            for pos in range(64):
                Xin = net_input(cur, np.full(len(cur), pos))
                h = np.maximum(Xin @ W1 + b1, 0)
                lg = (h @ W2 + b2) / max(temp, 1e-6)
                mx = lg.max(1, keepdims=True)
                pr = np.exp(lg - mx); pr /= pr.sum(1, keepdims=True)
                new[:, pos] = [r.choice(C, p=pr[i]) for i in range(len(cur))]
            cur = new
        return cur

    def to_img(l): return pal[l].reshape(8, 8, 3)

    SC = 8
    def grid(imgs, cols=16):
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

    N = 32
    r = np.random.default_rng(2024)
    proto = r.choice(POOL, N, replace=False)

    # 教师生成（基准）
    T0 = np.array([teacher_gen(p, seed=i) for i, p in enumerate(proto)])
    Image.fromarray(grid([to_img(v) for v in T0])).save(
        os.path.join(RES, f'distill_use_{tag}_teacher.png'))
    print("  教师输出 -> distill_use_%s_teacher.png" % tag)

    # 用法 A: 精修
    for temp in (0.5, 1.0):
        Rf = student_refine(T0, temp=temp, steps=1, seed=5)
        Image.fromarray(grid([to_img(v) for v in Rf])).save(
            os.path.join(RES, f'distill_use_{tag}_refine_T{int(temp*10):02d}.png'))
        diff = float((Rf != T0).sum(1).mean())
        uc = np.mean([len(np.unique(v)) for v in Rf])
        print(f"  精修 T={temp}: 改动 {diff:.1f} 像素  颜色数 {uc:.2f}  "
              f"-> distill_use_{tag}_refine_T{int(temp*10):02d}.png")

    # 用法 B: 反复迭代（看是否收敛）
    X = T0.copy()
    for it in range(1, 5):
        X = student_refine(X, temp=0.8, steps=1, seed=100 + it)
        Image.fromarray(grid([to_img(v) for v in X])).save(
            os.path.join(RES, f'distill_use_{tag}_iter{it}.png'))
        dT = float((X != T0).sum(1).mean())
        uc = np.mean([len(np.unique(v)) for v in X])
        print(f"  迭代 {it}: 与教师输出差 {dT:.1f} 像素  颜色数 {uc:.2f}  "
              f"-> distill_use_{tag}_iter{it}.png", flush=True)

    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in lab[r2.choice(len(lab), N, replace=False)]])).save(
        os.path.join(RES, f'distill_use_{tag}_real.png'))
    print(f"  真实对照 -> distill_use_{tag}_real.png")


if __name__ == '__main__':
    main()
