# -*- coding: utf-8 -*-
"""对照：知识库《数模结合：从输入量化到非负累加量 N》+《激活与输出》 vs 三期编译器。

做法：用知识库里的公式**独立重写**一遍编译流程，然后与
03/src/compiler.py 的产物逐元素比较，统计每个量的最大绝对差。
不 import 编译器的中间结果来"抄"，只import它当参照。
"""
import json
import os
import sys

import numpy as np

ROOT = r'E:\redstone_nn'
sys.path.insert(0, os.path.join(ROOT, '03_phase3_quantized_network', 'src'))
import compiler as cmp3   # noqa: E402

MODEL = os.path.join(ROOT, '02_phase2_avatar', 'webapp',
                     'pca22_dim4w64L1_model.json')
XMIN, XMAX, INB = -3.5, 3.5, 8


# ----------------------------------------------------------------------
# 知识库版：完全按两页正文的公式写，不参考编译器源码的结构
# ----------------------------------------------------------------------
def kb_compile(layers, xmin, xmax, in_bits, w_ref):
    delta = np.full(layers[0]['w'].shape[1], (xmax - xmin) / 2 ** in_bits)
    a0 = np.full_like(delta, xmin)
    b_in = in_bits
    out = []
    for L in layers:
        w, th = L['w'], L['theta']
        m, n = w.shape

        # §三.2 缩放：分母看有没有 j
        g = np.abs(w) * delta[None, :]
        if L['comparable']:
            sigma = np.full(m, w_ref / (g.max() * 2 ** (b_in - 2)))
        else:
            sigma = w_ref / (g.max(axis=1) * 2 ** (b_in - 2))

        # §二 位权重
        W = np.rint(sigma[:, None, None] * w[:, :, None]
                    * delta[None, :, None]
                    * 2 ** np.arange(b_in)[None, None, :]).astype(np.int64)

        # §五 累加与非负化
        Cmin = np.minimum(W, 0).sum(axis=(1, 2))
        Nmax = np.abs(W).sum(axis=(1, 2))

        # §二 零点：P = round(-sigma*kappa - Cmin)
        kappa = th + (w * (a0 + 0.5 * delta)[None, :]).sum(axis=1)
        P = np.rint(-sigma * kappa - Cmin).astype(np.int64)

        # §3.1 / §3.2 输出
        if L['kind'] == 'relu':
            bY = L['out_bits']
            Gam = np.ceil((Nmax - P) / (2 ** bY - 1)).astype(np.int64)
            Gam = np.maximum(Gam, 1)                      # 编译器有这一钳
            Y = np.clip(np.floor_divide(Nmax - P, Gam), 0, 2 ** bY - 1)
            d_out = Gam / sigma
            rec = dict(kind='relu', sigma=sigma, W=W, Cmin=Cmin, Nmax=Nmax, P=P,
                       step=Gam, out_bits=bY, out_max=Y, delta_out=d_out)
        else:
            Om = np.rint(L['lam'] * sigma).astype(np.int64)
            bY = int(np.ceil(np.log2(Om.max() + 1)))
            Y = Om.copy()
            d_out = 1.0 / sigma
            rec = dict(kind='clip', sigma=sigma, W=W, Cmin=Cmin, Nmax=Nmax, P=P,
                       step=Om, out_bits=bY, out_max=Y, delta_out=d_out)
        out.append(rec)
        delta, a0, b_in = d_out, np.zeros(m), bY
    return out


def maxdiff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float('nan'), -1
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.max()), int((a != b).sum())


def main():
    mj = json.load(open(MODEL, encoding='utf-8'))
    raw = [dict(w=np.array(mj['gen']['layers'][0]['W']),
                theta=np.array(mj['gen']['layers'][0]['b']),
                kind='relu', comparable=False, out_bits=8, name='层1 4→64'),
           dict(w=np.array(mj['gen']['layers'][1]['W']),
                theta=np.array(mj['gen']['layers'][1]['b']),
                kind='clip', comparable=True,
                lam=np.full(32, 15.0), name='层2 64→32')]

    lines = []
    for w_ref in (882, 441):
        cmp_layers = [cmp3.Layer(r['w'], r['theta'], r['kind'],
                                 comparable=r['comparable'],
                                 out_bits=r.get('out_bits'),
                                 lam=r.get('lam'), name=r['name'])
                      for r in raw]
        ref = cmp3.compile_network(cmp_layers, XMIN, XMAX, INB,
                                   verbose=False, w_ref=w_ref)
        kb = kb_compile(raw, XMIN, XMAX, INB, w_ref)

        lines.append(f'=== w_ref = {w_ref} ===')
        for r, k in zip(ref, kb):
            lines.append(f"  [{r['name']}] kind={r['kind']}")
            for label, a, b in [
                ('sigma', r['sigma'], k['sigma']),
                ('W', r['W'], k['W']),
                ('Cmin', r['Cmin'], k['Cmin']),
                ('Nmax', r['Nmax'], k['Nmax']),
                ('P', r['P'], k['P']),
                ('step', r['c'] if r['kind'] == 'relu' else r['T'], k['step']),
                ('delta_out', r['delta_out'], k['delta_out']),
                ('out_bits', r['out_bits'], k['out_bits']),
                ('out_max', r['out_max'], k['out_max']),
            ]:
                d, n = maxdiff(a, b)
                lines.append(f'    {label:10s} max|Δ|={d:<12.6g} 不一致元素={n}')

        # 物理约束检查（知识库 §四 的三条）
        for k in kb:
            W = k['W']
            over = int(np.abs(W[:, :, :-1]).max())
            top = int(np.abs(W[:, :, -1]).max())
            # 单容器 V = 15+|W| ≤ 897 等价于 |W| ≤ 882；只数"超标的位"有多少个
            n_ring = int((np.abs(W) > 882).sum())
            lines.append(f"    约束: |W|非最高位={over} (应≤882)  最高位={top}  "
                         f"超 882 的位(需环)={n_ring} 个  "
                         f"N_max 最大={int(k['Nmax'].max())}")

    open(os.path.join(os.path.dirname(__file__), '_crosscheck_03.txt'),
         'w', encoding='utf-8').write('\n'.join(lines))
    print('done')


if __name__ == '__main__':
    main()
