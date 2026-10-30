#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
重新对齐 F_0：使 z=0 时 F=128，并验证截断对 argmax 准确率的影响
==================================================================
目标: F = z + 128
  -> z = 0 时 F = 128
  -> z < -128 时 F 会被截断为 0（总线不能为负）

验证两项:
  1. 仅平移（不截断）时准确率是否不变（数学上应当不变）
  2. 加上"F<0 截断为 0"后，准确率损失多少
  3. 统计 z < -128 的样本比例（截断触发率）
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
    Z = Xin.astype(np.float64) @ W + bias            # (10000, 10)

    print("=" * 100)
    print("重新对齐 F_0 使 z=0 时 F=128，并验证截断影响")
    print("=" * 100)

    # 各层架构参数
    layers = []
    for k in range(10):
        ln = k + 1
        parity = '奇数' if ln % 2 == 1 else '偶数'
        c_all = (15 + W[:, k]) if parity == '奇数' else (15 - W[:, k])
        Gs = np.zeros((len(Xin), 8), dtype=np.int64)
        g0s, gmaxs = [], []
        for gi, (r0, nr) in enumerate(GROUPS):
            pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
            cg = c_all[pxs]
            g0 = int(np.maximum(cg, 15).sum()); gmax = g0 - int(np.minimum(cg, 15).sum())
            g0s.append(g0); gmaxs.append(gmax)
            lit = lit_all[:, pxs]
            out = np.where(lit, 15, cg[None, :]) if parity == '偶数' \
                else np.where(lit, cg[None, :], 15)
            Gs[:, gi] = g0 - out.sum(1)
        F0_old = sum(gmaxs)
        F_old = F0_old - Gs.sum(1)
        offset = float((F_old - Z[:, k]).mean())
        # 要求 F = z + 128  ->  F_0_new = F_0_old + (128 - offset)
        F0_new = int(round(F0_old + TARGET - offset))
        layers.append({'k': k, 'layer': ln, 'parity': parity, 'Gs': Gs,
                       'F0_old': F0_old, 'F0_new': F0_new, 'offset': offset,
                       'g0s': g0s, 'gmaxs': gmaxs})

    print(f"\n[各层 F_0 调整]")
    print(f"  {'层':>3} | {'奇偶':>4} | {'F_0(旧)':>8} | {'offset(旧)':>10} | "
          f"{'F_0(新)':>8} | {'新 offset':>9} | {'新 F 范围(理论)':>16}")
    print("  " + "-" * 82)
    for L in layers:
        new_off = L['F0_new'] - (L['F0_old'] - L['offset'])
        print(f"  {L['layer']:>3} | {L['parity']:>4} | {L['F0_old']:>8} | "
              f"{L['offset']:>10.1f} | {L['F0_new']:>8} | {new_off:>9.1f} | "
              f"[0, {L['F0_new']}]")

    # ---- 构造新的 F（含截断）----
    Fold = np.zeros((len(Xin), 10), dtype=np.int64)
    Fnew = np.zeros((len(Xin), 10), dtype=np.int64)
    Fclip = np.zeros((len(Xin), 10), dtype=np.int64)
    for L in layers:
        k = L['k']
        Fold[:, k] = L['F0_old'] - L['Gs'].sum(1)
        Fnew[:, k] = L['F0_new'] - L['Gs'].sum(1)
        Fclip[:, k] = np.maximum(Fnew[:, k], 0)

    print(f"\n[验证1] 仅平移（不截断）")
    acc_old = float((Fold.argmax(1) == yte).mean())
    acc_new = float((Fnew.argmax(1) == yte).mean())
    print(f"  平移前准确率 {acc_old:.4f}")
    print(f"  平移后准确率 {acc_new:.4f}   "
          f"{'不变 OK' if acc_old == acc_new else '变了！'}")
    print(f"  逐样本 argmax 是否完全一致: "
          f"{'是' if np.array_equal(Fold.argmax(1), Fnew.argmax(1)) else '否'}")

    print(f"\n[验证2] 加上 F<0 截断为 0")
    acc_clip = float((Fclip.argmax(1) == yte).mean())
    print(f"  截断后准确率 {acc_clip:.4f}")
    print(f"  相对平移后损失 {(acc_new-acc_clip)*100:+.3f} 个百分点")
    diff = (Fclip.argmax(1) != Fnew.argmax(1))
    print(f"  预测发生改变的样本数: {int(diff.sum())} / {len(yte)}  "
          f"({diff.mean():.3%})")
    if diff.sum() > 0:
        print(f"    其中原本判对、截断后判错的: "
              f"{int(((Fnew.argmax(1) == yte) & (Fclip.argmax(1) != yte)).sum())}")
        print(f"    原本判错、截断后判对的: "
              f"{int(((Fnew.argmax(1) != yte) & (Fclip.argmax(1) == yte)).sum())}")
        # 查看这些样本的 z 值
        bad = np.flatnonzero(((Fnew.argmax(1) == yte) & (Fclip.argmax(1) != yte)))
        if len(bad) > 0:
            print(f"\n    受影响样本的 z 值（真实类别 vs 预测类别）:")
            for i in bad[:5]:
                print(f"      样本{i}: 真实 {yte[i]}, z = {np.round(Z[i],1)}")

    print(f"\n[验证3] 截断触发统计")
    below = (Z < -TARGET)
    print(f"  z < -128 的元素数: {int(below.sum())} / {Z.size}  ({below.mean():.3%})")
    per_layer = below.sum(0)
    print(f"  按层统计 z < -128 的次数:")
    print(f"  {'层':>3}(数字) | {'比较数':>7} | {'< -128':>7} | {'比例':>7}")
    print("  " + "-" * 40)
    for k in range(10):
        print(f"  {k+1:>3}({k})   | {len(yte):>7} | {int(per_layer[k]):>7} | "
              f"{per_layer[k]/len(yte):>6.2%}")
    # 真实的 F 值分布（用新的 F）
    print(f"\n  新 F 值分布（全部 10 层 x 10000 样本 = {Fnew.size} 个值）:")
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print(f"    分位数: " + "  ".join(f"P{q}={np.percentile(Fnew,q):.1f}" for q in qs))
    print(f"    F < 0 的个数: {int((Fnew<0).sum())}  ({(Fnew<0).mean():.3%})")
    print(f"    F < 16 的个数: {int((Fnew<16).sum())}  ({(Fnew<16).mean():.3%})")

    # 逐样本比每个类别
    print(f"\n[验证4] 截断只影响被判为最小的那些类别吗？")
    # 对每个样本，检查截断把哪个类别的分数改小了
    affected = (Fnew < 0)
    if affected.sum() > 0:
        # 被截断的类别是否是该样本的 argmax？
        argmax_clip = Fclip.argmax(1); argmax_new = Fnew.argmax(1)
        # 被截断类的原始 F
        print(f"  被截断的 (样本,类别) 对: {int(affected.sum())}")
        # 这些类别是否是 argmax？
        is_arg = np.zeros_like(affected)
        is_arg[np.arange(len(Xin)), argmax_new] = True
        print(f"    其中是该样本 argmax 的: {int((affected & is_arg).sum())}")
        print(f"    -> 若为 0，则截断不影响任何 argmax 决策")

    print(f"\n{'='*100}")
    print("### 结论")
    print(f"  平移（不截断）准确率 {acc_new:.4f}  = 原准确率 {acc_old:.4f}")
    print(f"  加截断后准确率     {acc_clip:.4f}")
    print(f"  损失 {(acc_new-acc_clip)*100:+.3f} 个百分点")

    json.dump({'acc_before_shift': acc_old, 'acc_after_shift': acc_new,
               'acc_after_clip': acc_clip,
               'target': TARGET,
               'layers': [{'layer': L['layer'], 'F0_old': L['F0_old'],
                           'offset_old': L['offset'], 'F0_new': L['F0_new'],
                           'g0s': L['g0s'], 'gmaxs': L['gmaxs']} for L in layers],
               'z_below_-128': int(below.sum())},
              open(os.path.join(RES, 'mc_F0_align.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_F0_align.json")


if __name__ == '__main__':
    main()
