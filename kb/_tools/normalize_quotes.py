#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把知识库正文里的 ASCII 双引号统一成中文引号。

不动的部分：
  - 代码围栏 ``` ... ``` 内的整行
  - 行内代码 `...`
  - 数学 $...$（含 $$...$$ 整行）
  - 90_外部/ 下的第三方稿件（保留其英文引号原文）

配对策略：以空行为界，每个"段落"内从开引号开始交替；段内落单的引号会被报出来。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # kb/
SKIP_DIRS = {'90_外部'}

OPEN, CLOSE = '\u201c', '\u201d'


def convert(text):
    out = []
    in_fence = False
    open_q = True
    odd = []

    for lineno, line in enumerate(text.split('\n'), 1):
        stripped = line.strip()

        if stripped.startswith('```'):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        if stripped == '':
            open_q = True          # 段落边界重置配对
            out.append(line)
            continue

        res = []
        in_code = False
        in_math = False
        for ch in line:
            if ch == '`':
                in_code = not in_code
                res.append(ch)
                continue
            if ch == '$' and not in_code:
                in_math = not in_math
                res.append(ch)
                continue
            if ch == '"' and not in_code and not in_math:
                res.append(OPEN if open_q else CLOSE)
                open_q = not open_q
                continue
            res.append(ch)
        if not open_q:
            odd.append(lineno)
                # 段内没闭合，记录但不强行补
        out.append(''.join(res))

    return '\n'.join(out), odd


def main():
    targets = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith('.md'):
                targets.append(os.path.join(dirpath, fn))

    total = 0
    for p in sorted(targets):
        src = open(p, encoding='utf-8').read()
        if '"' not in src:
            continue
        dst, odd = convert(src)
        n = src.count('"') - dst.count('"')
        if n:
            open(p, 'w', encoding='utf-8', newline='\n').write(dst)
            total += n
            rel = os.path.relpath(p, ROOT)
            extra = f'  段内未闭合行: {odd}' if odd else ''
            print(f'{rel:32s} 转换 {n:3d} 个{extra}')
    print(f'--- 共转换 {total} 个 ASCII 双引号 ---')


if __name__ == '__main__':
    sys.exit(main())
