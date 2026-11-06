# -*- coding: utf-8 -*-
"""量一下 (N^max - P) 与 n、b、W_ref 的关系。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_baseline2 as E

XMIN, DELTA, WREF = E.X_MIN, E.DELTA, E.WREF

print('n    b   (Nmax-P)/nW_ref    Nmax/(nW_ref)   E|w|/max|w|')
for n in (4, 8, 16, 32, 64):
    for b in (6, 10, 14):
        vals, nmax_r, ratio = [], [], []
        for t in range(20):
            rng = np.random.default_rng(500 + 17 * n + b + t)
            m = 16
            w = rng.normal(0, 1 / np.sqrt(n), size=(m, n))
            th = np.zeros(m)
            x = rng.uniform(XMIN, XMIN + DELTA, size=(64, n))
            r = E.layer_int(w, th, x, WREF, b, DELTA / 2 ** b)
            vals.append(np.mean(r['N'].max(0) - r['P']))
            nmax_r.append(np.mean(np.abs(r['W']).sum(axis=(1, 2))))
            ratio.append(np.mean(np.abs(w).sum(1) / np.abs(w).max(1)))
        print('%2d  %2d   %10.3f      %10.3f      %6.3f' % (
            n, b, np.mean(vals) / (n * WREF), np.mean(nmax_r) / (n * WREF), np.mean(ratio)))
