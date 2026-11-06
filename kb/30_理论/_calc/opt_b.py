# -*- coding: utf-8 -*-
"""扫描更宽的位宽范围，定位「单层输出逆回误差」的最小点。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_baseline2 as E

rows = E.run([4, 16, 64], [4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
for n in (4, 16, 64):
    pts = [(b, eo) for (nn, b, ex, ew, ey, eo) in rows if nn == n]
    bmin = min(pts, key=lambda z: z[1])
    print(f'n={n:2d}  b*={bmin[0]:2d}  eps_out={bmin[1]:.3e}')
    print('   ' + '  '.join('b%d:%.2e' % (b, e) for b, e in pts))
