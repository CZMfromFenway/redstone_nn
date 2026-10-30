#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
通用平面生成器（任意 n, b）。

坐标公式（全部由 n, b 推出）
    W        = 2n + 6
    ODD      = 2i+3 (i=0..n-1)     EVEN = 2i+4
    颜色行   z = 3k+2 (k 偶) / 3k+3 (k 奇)      k = 0..b-1
    木桶行   z = 3k+1 (k 偶) / 3k+4 (k 奇)      k = 0..b-2
    链行     z = 6j+4                            j = 0..b//2-1
    MSB 行   zmsb = 3(b-1) + (2 if (b-1) 偶 else 3)
    灰环     朝上：z = 3b-gshift .. 3b-gshift+2   朝下：z = gshift .. gshift+2
    白总线   x = 2n+3        白环 x = 2n+3..2n+5
    配对区   z = zmsb+1 .. zmsb+7
    L        = zmsb + 8
"""
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cmp_sim import Board, DIRV  # noqa: E402

FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def classify_pair(Sa, Sb):
    lo, hi = min(Sa, Sb), max(Sa, Sb)
    if lo > 897:
        return "I_a" if Sa >= Sb else "I_b"
    if hi > 897:
        return "II_a" if Sa >= Sb else "II_b"
    return "III"


def geom(n, b, gshift=2):
    W = 2 * n + 6
    zmsb = 3 * (b - 1) + (2 if (b - 1) % 2 == 0 else 3)
    Jn = (zmsb - 2) // 6 + 1
    L = zmsb + 8
    return dict(W=W, L=L, zmsb=zmsb, xw=2 * n + 3,
                ODD=[2 * i + 3 for i in range(n)],
                EVEN=[2 * i + 4 for i in range(n)],
                zlast_chain=6 * (Jn - 1) + 4)


def build(n, b, S, K, P, inp, msb, wsign, grey_up=True, white_up=True,
          gshift=2, in_rev=False, par_swap=False):
    G = geom(n, b, gshift)
    W, L, zmsb, xw = G["W"], G["L"], G["zmsb"], G["xw"]
    ODD, EVEN = G["ODD"], G["EVEN"]
    g = Board(W, L)
    CE = EVEN if par_swap else ODD
    CO = ODD if par_swap else EVEN

    def X(i):
        return (n - 1 - i) if in_rev else i

    def COL(k, i):
        return (CE if k % 2 == 0 else CO)[X(i)]

    def C(p, fac, mode="compare", init=0):
        g.set_cmp(p, FLIP[fac], mode, init)

    # ---- 灰环
    _zlc = G["zlast_chain"]
    gz = (_zlc + 2 - gshift) if grey_up else gshift
    for (x, z, f) in ((1, gz, "E"), (2, gz + 1, "S"), (1, gz + 2, "W"), (0, gz + 1, "N")):
        C((x, z), f, "compare", K)
    for (x, z) in ((0, gz), (2, gz), (0, gz + 2), (2, gz + 2)):
        g.set_solid((x, z), "base")

    # ---- 白环（列 xw..xw+2）
    wz = (zmsb - 1) if white_up else 1
    for (x, z, f) in ((xw + 1, wz, "W"), (xw + 2, wz + 1, "N"),
                      (xw + 1, wz + 2, "E"), (xw, wz + 1, "S")):
        C((x, z), f, "compare", P)
    for (x, z) in ((xw, wz), (xw + 2, wz), (xw, wz + 2), (xw + 2, wz + 2)):
        g.set_solid((x, z), "white")

    # ---- 输入行 + 木桶（位 0..b-2）
    for k in range(b - 1):
        zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
        zb = 3 * k + 1 if k % 2 == 0 else 3 * k + 4
        for i in range(n):
            c = COL(k, i)
            on = (inp[i][k] == 1) != (wsign[i][k] > 0)
            g.set_solid((c, zc), "col")
            g.set_torch((c, zc), 15 if on else 0)
            g.set_barrel((c, zb), int(S[i][k]))

    # ---- 周期：cmp 行 + 减法链
    for j in range(zmsb // 6):
        z3, z4, z5 = 6 * j + 3, 6 * j + 4, 6 * j + 5
        for x in CE:
            C((x, z3), "N")
        for x in ODD + EVEN:
            C((x, z4), "W", "subtract")
        for x in CO:
            C((x, z5), "S")

    # ---- 左总线 x=2
    zlc = G["zlast_chain"]
    if grey_up:
        r0, r1 = 5, gz - 1
    else:
        r0, r1 = gz + 3, zlc
    for z in range(r0, r1 + 1):
        p = (2, z)
        if p in g.cmp:
            continue
        if z % 2 == 0:
            g.set_solid(p, "base")
        else:
            C(p, "S" if grey_up else "N")
    g.set_solid((2, 4), "base")            # 第一条链的取电块
    if grey_up:
        g.set_solid((2, zlc), "base")       # 最后一条链的取电块

    # ---- 右总线 x=xw
    for z in range(4, zlc + 2):
        p = (xw, z)
        if z % 2 == 0:
            C(p, "S" if white_up else "N",
              "subtract" if (z - 4) % 6 == 0 else "compare")
        else:
            g.set_solid(p, "white")

    # ---- MSB 行
    for i in range(n):
        on = (inp[i][b - 1] == 1) != (wsign[i][b - 1] > 0)
        g.set_solid((COL(b - 1, i), zmsb), "col")
        g.set_torch((COL(b - 1, i), zmsb), 15 if on else 0)

    # ---- 配对区
    zr = zmsb + 1
    for pi in range(n // 2):
        a, bb = 2 * pi, 2 * pi + 1
        if COL(b - 1, a) > COL(b - 1, bb):
            a, bb = bb, a
        xa, xb = COL(b - 1, a), COL(b - 1, bb)
        Sa, Sb = int(S[a][b - 1]), int(S[bb][b - 1])
        kind = classify_pair(Sa, Sb)
        if kind == "III":
            g.set_barrel((xa, zr), Sa)
            g.set_barrel((xb, zr), Sb)
            continue
        big_xa = Sa >= Sb
        g.set_barrel((xa + 1, zr), abs(Sa - Sb))
        C((xa, zr), "S"); C((xb, zr), "S")
        C((xa, zr + 1), "S"); C((xa + 1, zr + 1), "N"); C((xb, zr + 1), "S")
        for x in (xa, xb):
            for dz in (2, 4, 6):
                g.set_solid((x, zr + dz), "col")
                g.set_torch((x, zr + dz), 0)
        C((xa + 1, zr + 2), "W" if big_xa else "E", "subtract")
        C((xa, zr + 3) if big_xa else (xb, zr + 3), "S")
        smax = max(Sa, Sb)
        C((xa + 1, zr + 4), "E", "compare", smax)
        C((xa, zr + 5), "N", "compare", smax)
        C((xb, zr + 5), "S", "compare", smax)
        C((xa + 1, zr + 6), "W", "compare", smax)
    # 奇数 n：最后一个输入没有配对，单桶或单环
    if n % 2 == 1:
        i = n - 1
        x = COL(b - 1, i)
        Sv = int(S[i][b - 1])
        if Sv <= 897:
            g.set_barrel((x, zr), Sv)
        else:
            # 单独一个环：把 Sv 顶进 MSB 块
            # 向左展开的 3 列闭环：(x-2, x-1, x) x (zr+4 .. zr+6)
            xm, xc = x, x + 1      # 向右展开：只占白环的列，但 z 完全错开
            for (_x, _z) in ((xm, zr + 4), (x + 2, zr + 4), (xm, zr + 6), (x + 2, zr + 6),
                             (x, zr + 2)):
                g.set_solid((_x, _z), "col")
                g.set_torch((_x, _z), 0)
            # 注意 C() 内部做 FLIP：要输出 E 就传 W，要输出 N 就传 S
            C((xc, zr + 4), "W", "compare", Sv)   # 输出 E → (x+2, zr+4)
            C((x + 2, zr + 5), "N", "compare", Sv)  # 输出 S → (x+2, zr+6)
            C((xc, zr + 6), "E", "compare", Sv)   # 输出 W → (x, zr+6)
            C((x, zr + 5), "S", "compare", Sv)    # 输出 N → (x, zr+4)
            # 向上取电链：环块 (x,zr+4) → cmp(zr+3) → cmp(zr+1) → MSB 块
            C((x, zr + 3), "S", "compare", 0)     # 输出 N
            C((x, zr + 1), "S", "compare", 0)     # 输出 N
            C((x, zr), "S", "compare", 0)         # 输出 N → 充能 MSB 块
    return g
