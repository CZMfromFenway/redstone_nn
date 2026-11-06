# -*- coding: utf-8 -*-
"""检查知识库页面里数学公式的写法是否安全（$$ 配对、跨行、转义）。"""
import sys

for p in sys.argv[1:]:
    t = open(p, encoding='utf-8').read()
    n = t.count('$$')
    print(p)
    print('  $$ 个数:', n, '(偶数才配对)')
    print('  跨行公式（一行只有 $$）:',
          sum(1 for l in t.split('\n') if l.strip() == '$$'))
    print('  公式内换行符 \\\\ :', t.count('\\\\'))
    print('  行数:', t.count('\n') + 1)
