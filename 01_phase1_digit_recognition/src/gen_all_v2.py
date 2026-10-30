#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
生成全部 10 层容器放置函数（y 步长 = 2，层内 (x,z) 完全一致）
================================================================
层与 y:
  第 10 层(数字9) y = -1
  第  9 层(数字8) y = -3
  ...
  第  1 层(数字0) y = -19
  y = -(1 + 2*(10 - layer))

各层 (x,z) 布局完全相同，唯一区别:
  * y
  * 层间奇偶性 -> 容器值 c
        偶数层(第2,4,6,8,10): c = 15 - W
        奇数层(第1,3,5,7, 9): c = 15 + W

坐标:
  权重: x 沿列（奇行 0..-28，偶行 -1..-29），z 沿行
        z: 13=0, 14=-2, 11=-4, 12=-6, 9=-8, 10=-10, 7=-12, 8=-14,
           5=-16, 6=-18, 3=-20, 4=-22, 1=-24, 2=-26, 0=-30
  G_{1,0}: ~-30 <y> ~1 ; i+1 -> z-4
  F_0    : ~1 <y> ~-28
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DP = os.path.join(os.path.dirname(ROOT), 'datapack', 'tiny_cnn')
FUNC_DIR = os.path.join(DP, 'data', 'deploy_weights', 'function')
RES = os.path.join(ROOT, 'results')
ORDER = [13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0]
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
TARGET = 128


def fmt(x, y, z):
    def p(v):
        return '~' if v == 0 else f'~{v}'
    return f"{p(x)} {p(y)} {p(z)}"


def build_Z():
    Z, z = {}, 0
    for i, r in enumerate(ORDER):
        Z[r] = z
        if i < len(ORDER) - 1:
            z -= 4 if r == 2 else 2
    return Z


def x_start(row):
    return 0 if row % 2 == 1 else -1


def load_test():
    import struct
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    def li(p):
        with open(p, 'rb') as f:
            mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
            dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
            d = np.frombuffer(f.read(), dtype=np.uint8)
        return d.reshape(dims)
    def t15(im):
        n = len(im)
        a = im.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
        o = np.zeros((n, 15, 15), dtype=np.float32)
        o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
        o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
        return o
    Xte = li(RAW + r'\t10k-images-idx3-ubyte')
    yte = li(RAW + r'\t10k-labels-idx1-ubyte')
    thr = t15(li(RAW + r'\train-images-idx3-ubyte')).mean()
    B = (t15(Xte) > thr)
    return B.reshape(len(B), -1), yte


def main():
    zz = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = zz['W'].astype(int); bias = zz['b'].astype(int)
    Xin, yte = load_test()
    lit = Xin.astype(bool)
    Z = build_Z()
    os.makedirs(FUNC_DIR, exist_ok=True)

    layers = []
    for k in range(10):
        ln = k + 1
        parity = '偶数' if ln % 2 == 0 else '奇数'
        y = -(1 + 2 * (10 - ln))
        c_all = (15 - W[:, k]) if parity == '偶数' else (15 + W[:, k])
        G0, Gmax, Gsum = [], [], np.zeros(len(Xin), dtype=np.int64)
        for (r0, nr) in GROUPS:
            pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
            cg = c_all[pxs]
            g0 = int(np.maximum(cg, 15).sum())
            gmax = g0 - int(np.minimum(cg, 15).sum())
            G0.append(g0); Gmax.append(gmax)
            L = lit[:, pxs]
            out = np.where(L, 15, cg[None, :]) if parity == '偶数' \
                else np.where(L, cg[None, :], 15)
            Gsum += g0 - out.sum(1)
        F0_old = sum(Gmax)
        z = Xin.astype(np.int64) @ W[:, k] + bias[k]
        F_old = F0_old - Gsum
        off = F_old - z
        assert off.max() == off.min()
        offset = int(off[0])
        F0 = F0_old - offset + TARGET
        # 验证
        Fnew = F0 - Gsum
        assert (Fnew - (z + TARGET)).max() == (Fnew - (z + TARGET)).min() == 0
        layers.append({'k': k, 'layer': ln, 'parity': parity, 'y': y,
                       'c_all': c_all, 'G0': G0, 'Gmax': Gmax,
                       'F0_old': F0_old, 'offset': offset, 'F0': F0})

    print("=" * 104)
    print("10 层参数（y 步长 2；F_0 使 F = z + 128）")
    print("=" * 104)
    print(f"  {'层':>3}(数字) | {'奇偶':>4} | {'y':>5} | {'c范围':>9} | "
          f"{'F_0':>5} | {'G_10..G_80':>36}")
    print("  " + "-" * 100)
    for L in layers:
        gs = ' '.join(f"{g:>3}" for g in L['G0'])
        print(f"  {L['layer']:>3}({L['k']}) | {L['parity']:>4} | {L['y']:>5} | "
              f"[{int(L['c_all'].min()):>2},{int(L['c_all'].max()):>2}]   | "
              f"{L['F0']:>5} | {gs}")

    # ---- 生成 ----
    lines = ["# 全部 10 层容器放置 -- 由 gen_deploy.py 生成",
             "# 执行位置: ~ ~ ~",
             "# 层由 y 决定，y 步长 2:",
             "#   第10层(数字9) y=-1   第9层(数字8) y=-3   第8层(数字7) y=-5",
             "#   第7层(数字6) y=-7    第6层(数字5) y=-9   第5层(数字4) y=-11",
             "#   第4层(数字3) y=-13   第3层(数字2) y=-15  第2层(数字1) y=-17",
             "#   第1层(数字0) y=-19",
             "# 各层 (x,z) 布局完全相同；偶数层 c=15-W，奇数层 c=15+W",
             "# F_0 已对齐: F = z + 128（10 层可比）",
             ""]
    n = 0
    for L in layers:
        y = L['y']
        lines.append("")
        lines.append(f"# ===== 第 {L['layer']} 层（数字 {L['k']}，{L['parity']}层，"
                     f"y={y}） =====")
        lines.append(f"# c = {'15 - W' if L['parity']=='偶数' else '15 + W'}, "
                     f"信号 [{int(L['c_all'].min())}, {int(L['c_all'].max())}], "
                     f"F_0 = {L['F0']}")
        for r in ORDER:
            z = Z[r]; xs = x_start(r)
            lines.append(f"# 权重 像素行 {r} (z={z})")
            for col in range(14, -1, -1):
                x = xs - 2 * (14 - col)
                lines.append(f"execute at @s positioned {fmt(x,y,z)} run "
                             f"function vessel:barrel_{int(L['c_all'][r*15+col])}")
                n += 1
        lines.append(f"# G_{{i,0}}")
        for i in range(8):
            lines.append(f"execute at @s positioned {fmt(-30,y,1-4*i)} run "
                         f"function vessel:barrel_{L['G0'][i]}")
            n += 1
        lines.append(f"# F_0")
        lines.append(f"execute at @s positioned {fmt(1,y,-28)} run "
                     f"function vessel:barrel_{L['F0']}")
        n += 1

    path = os.path.join(FUNC_DIR, 'deploy_all_layers.mcfunction')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n[已生成] {path}")
    print(f"  指令总数 {n}  = 10 层 x (225 权重 + 8 G + 1 F_0)")

    # 抽样展示
    print(f"\n[第 10 层 前 6 行 / 偏置行]")
    with open(path, encoding='utf-8') as f:
        for ln in f:
            if 'y=' in ln or (ln.startswith('execute') and ('~-1 ' in ln or
                                                           '~-3 ~1' in ln or
                                                           '~1 ~-1 ~-28' in ln)):
                print('  ' + ln.rstrip())

    json.dump([{kk: (v.tolist() if isinstance(v, np.ndarray) else v)
                for kk, v in L.items()} for L in layers],
              open(os.path.join(RES, 'deploy_all_layers.json'), 'w'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
