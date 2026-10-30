#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 16 色 / 48 色调色板（全部来自 MC 染料色系）
==================================================
16 色: 混凝土 16 色（czm 给定的色环顺序）
48 色: 混凝土 16 + 羊毛 16 + 陶瓦 16
       各族内部按色环排序，族间按"同一色相"配对，使整体仍近似沿色环
"""
import os, json, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
MC = r'D:\czm我的世界\.minecraft'

ORDER16 = ['white', 'light_gray', 'gray', 'black', 'brown', 'red', 'orange', 'yellow',
           'lime', 'green', 'cyan', 'light_blue', 'blue', 'purple', 'magenta', 'pink']


def find_tex(family, root=MC):
    """找某族的 16 色纹理（用众数色）"""
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if not f.endswith('.png'):
                continue
            base = f[:-4]
            for nm in ORDER16:
                for suf in (f'{nm}_{family}', f'{nm}_{family}_top'):
                    if base == suf and nm not in out:
                        try:
                            im = Image.open(os.path.join(dp, f)).convert('RGB')
                            a = np.array(im).reshape(-1, 3)
                            vals, cnt = np.unique(a, axis=0, return_counts=True)
                            out[nm] = vals[cnt.argmax()].tolist()
                        except Exception:
                            pass
    return out


res = {}
for fam in ('concrete', 'wool', 'terracotta'):
    d = find_tex(fam)
    res[fam] = d
    print(f"{fam:>12}: 找到 {len(d)}/16 色")
    if d:
        print("             ", {k: tuple(v) for k, v in list(d.items())[:4]})

# 16 色 = 混凝土
pal16 = [res['concrete'][n] for n in ORDER16 if n in res['concrete']]
print(f"\n16 色调色板: {len(pal16)} 色")

# 48 色 = 三族交错（同一色相的三族相邻，保持色环的大方向）
pal48 = []
for n in ORDER16:
    for fam in ('concrete', 'wool', 'terracotta'):
        if n in res[fam]:
            pal48.append(res[fam][n])
print(f"48 色调色板: {len(pal48)} 色")

json.dump({'order16': ORDER16, 'pal16': pal16, 'pal48': pal48,
           'by_family': res},
          open(os.path.join(DATA, 'palettes.json'), 'w'), ensure_ascii=False, indent=2)
print(f"已写出: data/palettes.json")
