#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
F_0 重对齐：干净复核
======================
目标: F = z + 128  （z=0 时 F=128）
实现: F_0(k) = F_0_old(k) - offset(k) + 128
      其中 offset(k) 由实测确定: offset(k) = mean(F_old - z)

验证:
  1. offset 实测值是否恒定（逐样本 F-z 是否常数）
  2. 平移后 F == z + 128（逐样本）
  3. 加"F<0 截断为 0"后 argmax 准确率
  4. min z 与 -128 的关系
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
TARGET = 128


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


def main():
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); bias = z['b'].astype(int)
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    Xin = (to15(Xte) > thr).astype(np.int64).reshape(len(Xte), -1)
    lit_all = Xin.astype(bool)
    Z = Xin.astype(np.float64) @ W + bias

    # ---- 构造每层的 F_old，并实测 offset ----
    info = []
    for k in range(10):
        parity = '奇数' if (k + 1) % 2 == 1 else '偶数'
        c_all = (15 + W[:, k]) if parity == '奇数' else (15 - W[:, k])
        Gs = np.zeros((len(Xin), 8), dtype=np.int64)
        gmaxs = []
        for gi, (r0, nr) in enumerate(GROUPS):
            pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
            cg = c_all[pxs]
            g0 = int(np.maximum(cg, 15).sum()); gmax = g0 - int(np.minimum(cg, 15).sum())
            gmaxs.append(gmax)
            lit = lit_all[:, pxs]
            out = np.where(lit, 15, cg[None, :]) if parity == '偶数' \
                else np.where(lit, cg[None, :], 15)
            Gs[:, gi] = g0 - out.sum(1)
        F0_old = sum(gmaxs)
        F_old = F0_old - Gs.sum(1)
        d = F_old - Z[:, k]
        assert d.max() - d.min() == 0, (k, d.min(), d.max())
        off = int(d[0])
        F0_new = F0_old - off + TARGET
        info.append({'k': k, 'layer': k + 1, 'parity': parity, 'Gs': Gs,
                     'F0_old': F0_old, 'offset': off, 'F0_new': F0_new,
                     'gmaxs': gmaxs,
                     'sum_gmax': F0_old})

    print("=" * 96)
    print("F_0 重对齐（目标 F = z + 128）")
    print("=" * 96)
    print(f"\n  {'层':>3}(数字) | {'奇偶':>4} | {'F_0(旧)':>8} | {'实测 offset':>10} | "
          f"{'F_0(新)':>8} | {'新 F 范围':>12}")
    print("  " + "-" * 70)
    for L in info:
        print(f"  {L['layer']:>3}({L['k']}) | {L['parity']:>4} | {L['F0_old']:>8} | "
              f"{L['offset']:>10} | {L['F0_new']:>8} | [0, {L['F0_new']}]")

    # ---- 新 F ----
    Fold = np.stack([L['F0_old'] - L['Gs'].sum(1) for L in info], 1)
    Fnew = np.stack([L['F0_new'] - L['Gs'].sum(1) for L in info], 1)
    Fclip = np.maximum(Fnew, 0)

    print(f"\n[验证1] 平移后 F == z + 128 ?")
    chk = Fnew - (Z + TARGET)
    print(f"  F - (z+128): 范围 [{chk.min()}, {chk.max()}]  "
          f"{'恒为 0 OK' if chk.max()==chk.min()==0 else '有偏差'}")

    print(f"\n[验证2] argmax 准确率")
    a_fold = float((Fold.argmax(1) == yte).mean())
    a_fnew = float((Fnew.argmax(1) == yte).mean())
    a_z = float((Z.argmax(1) == yte).mean())
    a_clip = float((Fclip.argmax(1) == yte).mean())
    print(f"  原网络 argmax(z)           {a_z:.4f}")
    print(f"  平移前 argmax(F_old)       {a_fold:.4f}   "
          f"（各层 offset 不同 -> 被破坏）")
    print(f"  平移后 argmax(F_new)       {a_fnew:.4f}")
    print(f"  加截断 argmax(max(F,0))    {a_clip:.4f}")
    print(f"  截断损失 {(a_fnew-a_clip)*100:+.4f} 个百分点")
    print(f"  平移后与 z 的 argmax 一致: "
          f"{'是' if np.array_equal(Fnew.argmax(1), Z.argmax(1)) else '否'}")
    print(f"  截断后与 z 的 argmax 一致: "
          f"{'是' if np.array_equal(Fclip.argmax(1), Z.argmax(1)) else '否'}")
    nd = int((Fclip.argmax(1) != Fnew.argmax(1)).sum())
    print(f"  预测被截断改变的样本数: {nd}")

    print(f"\n[验证3] 截断是否会触发")
    print(f"  z 的实测范围: [{Z.min():.0f}, {Z.max():.0f}]")
    print(f"  阈值 -128 -> z < -128 的个数: {int((Z < -TARGET).sum())}")
    print(f"  新 F 的实测范围: [{Fnew.min():.0f}, {Fnew.max():.0f}]")
    print(f"  F < 0 的个数: {int((Fnew < 0).sum())}")
    print(f"  F < 16 的个数: {int((Fnew < 16).sum())}")
    print(f"\n  结论: {'截断从未触发，准确率零损失' if (Fnew<0).sum()==0 else '截断有触发'}")
    print(f"  余量: 最小 F = {Fnew.min():.0f}，离 0 还有 {Fnew.min():.0f}")

    # 若担心真实世界分布更广，算安全边界
    print(f"\n[验证4] 安全边界")
    print(f"  z 最小 {Z.min():.0f} -> 需要 F_0 满足 F_0 - sum(G) >= 0")
    print(f"  实测最小 F = {Fnew.min():.0f}，即 z 最小值 {Z.min():.0f} 时 F = "
          f"{Z.min()+TARGET:.0f}")
    print(f"  要触发截断需要 z < -128，即比实测最小 {Z.min():.0f} 再低 "
          f"{-128-Z.min():.0f}")

    json.dump({'target': TARGET,
               'accuracy': {'z': a_z, 'F_old': a_fold, 'F_new': a_fnew, 'F_clip': a_clip},
               'layers': [{'layer': L['layer'], 'digit': L['k'], 'parity': L['parity'],
                           'F_0_old': L['F0_old'], 'offset': L['offset'],
                           'F_0_new': L['F0_new'], 'G_max': L['gmaxs']} for L in info],
               'z_min': int(Z.min()), 'z_max': int(Z.max()),
               'F_min': int(Fnew.min()), 'F_max': int(Fnew.max()),
               'clip_triggered': int((Fnew < 0).sum())},
              open(os.path.join(RES, 'mc_F0_align_v2.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_F0_align_v2.json")


if __name__ == '__main__':
    main()
