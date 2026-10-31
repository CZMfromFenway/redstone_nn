# -*- coding: utf-8 -*-
"""把 py_compile_reference.json 的结构压成一份小报告，写到文件里（不往 stdout 打大对象）。"""
import json
import numpy as np

SRC = r'E:\redstone_nn\03_phase3_quantized_network\results\py_compile_reference.json'
OUT = r'E:\redstone_nn\kb\20_数学\_calc\_schema.txt'


def shape_of(x, depth=0):
    if isinstance(x, list) and x and depth < 5:
        return (len(x),) + shape_of(x[0], depth + 1)
    if isinstance(x, list):
        return (0,)
    return ()


def main():
    d = json.load(open(SRC, encoding='utf-8'))
    lines = []
    lines.append('meta: ' + json.dumps(d['meta'], ensure_ascii=False))
    for L, v in d['layers'].items():
        lines.append('=' * 50)
        lines.append(L)
        for k, val in v.items():
            if isinstance(val, list):
                sh = shape_of(val)
                try:
                    arr = np.array(val)
                    if arr.dtype != object and arr.size:
                        lines.append(f'  {k:14s} shape={sh} dtype={arr.dtype} '
                                     f'min={arr.min()} max={arr.max()}')
                        continue
                except Exception:
                    pass
                lines.append(f'  {k:14s} shape={sh} (ragged)')
            else:
                s = repr(val)
                lines.append(f'  {k:14s} {type(val).__name__} = {s[:80]}')
    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
    print('written', OUT)


if __name__ == '__main__':
    main()
