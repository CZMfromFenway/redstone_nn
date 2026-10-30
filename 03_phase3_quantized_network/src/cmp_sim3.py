#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按 schem 原样块表建板，翻转与否做成开关，两种都跑。

主体 = rsnn.schem 的精确块表（n=4,b=8）
配对区 = renn_sr.schem 的精确块表（pair 0 / pair 1 两型）
修正：彩色方块同时注册为实体（上一版漏了，所以品红块在图上消失）
"""
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
from cmp_sim import Board, DIRV  # noqa: E402

FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def build(n, b, S, K, P, inp, flip, msb, wsign=None):
    W, L = 14, 32
    g = Board(W, L)
    if wsign is None:
        wsign = [[1] * b for _ in range(n)]
    ODD, EVEN = [2 * i + 3 for i in range(n)], [2 * i + 4 for i in range(n)]

    def C(p, fac, mode="compare", init=0):
        g.set_cmp(p, FLIP[fac] if flip else fac, mode, init)

    # ---- 灰环（中心 (1,25)）：schem 原样 facing，输出方向按开关
    for (x, z, fac) in ((1, 24, "E"), (2, 25, "S"), (1, 26, "W"), (0, 25, "N")):
        C((x, z), fac, "compare", init=K)
    for (x, z) in ((0, 24), (2, 24), (0, 26), (2, 26)):
        g.set_solid((x, z), "base")

    # ---- 白环（中心 (12,24)）
    for (x, z, fac) in ((12, 23, "E"), (13, 24, "S"), (12, 25, "W"), (11, 24, "N")):
        C((x, z), fac, "compare", init=P)
    for (x, z) in ((11, 23), (13, 23), (11, 25), (13, 25)):
        g.set_solid((x, z), "white")

    # ---- 输入行 + 木桶
    # 火把激活条件随 W 的符号反转：W>0 看 x=1，W<0 看 x=0
    for k in range(b - 1):
        col = ODD if k % 2 == 0 else EVEN
        zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
        zb = 3 * k + 1 if k % 2 == 0 else 3 * k + 4
        for i in range(n):
            active = (inp[i][k] == 1) != (wsign[i][k] > 0)
            g.set_solid((col[i], zc), "col")
            g.set_torch((col[i], zc), 15 if active else 0)
            g.set_barrel((col[i], zb), int(S[i][k]))

    # ---- 周期：cmp 行 + 减法链（schem: z3 全 north, z4 全 west-subtract, z5 北+x2/南+偶列）
    for j in range(4):
        z3, z4, z5 = 6 * j + 3, 6 * j + 4, 6 * j + 5
        if z3 > 3:
            C((2, z3), "S")      # z=3 是断头路，剪掉
        for x in ODD:
            C((x, z3), "N")
        for x in ODD + EVEN:
            C((x, z4), "W", "subtract")
        C((2, z5), "S")
        for x in EVEN:
            C((x, z5), "S")

    # ---- 左总线 x=2（schem: z=1 south，其余奇数 z north）
    for z in range(1, 24):
        if (2, z) in g.cmp:
            continue
        if z % 2 == 0:
            g.set_solid((2, z), "base")
        elif z >= 5:
            C((2, z), "S")          # 灰环在下方 ⇒ 灰线朝上流


    # ---- 右总线 x=11（schem: 偶数 z south，奇数 z 白）
    for z in range(4, 23):
        if z % 2 == 0:
            C((11, z), "S", "subtract" if z in (4, 10, 16, 22) else "compare")
        else:
            g.set_solid((11, z), "white")

    # ---- MSB 行
    for i in range(n):
        act7 = (inp[i][b - 1] == 1) != (wsign[i][b - 1] > 0)
        g.set_solid((EVEN[i], 24), "col")
        g.set_torch((EVEN[i], 24), 15 if act7 else 0)

    # ---- 配对区（**用翻转**：图验证 22/22 全部有源）
    def C2(p, fac, mode="compare", init=0):
        g.set_cmp(p, FLIP[fac], mode, init)

    for pi, (a, bb) in enumerate([(0, 1), (2, 3)]):
        xa, xb = EVEN[a], EVEN[bb]
        lo, hi = min(msb[a], msb[bb]), max(msb[a], msb[bb])
        case = "I" if lo > 882 else ("II" if hi > 882 else "III")
        if case == "III":
            g.set_barrel((xa, 25), int(S[a][b - 1]))
            g.set_barrel((xb, 25), int(S[bb][b - 1]))
            continue
        g.set_barrel((xa + 1, 25), abs(int(S[a][b - 1] - S[bb][b - 1])))
        C2((xa, 25), "S"); C2((xb, 25), "S")
        C2((xa, 26), "S"); C2((xa + 1, 26), "N"); C2((xb, 26), "S")
        for x in (xa, xb):
            g.set_solid((x, 27), "col")
            g.set_torch((x, 27), 0)
            g.set_solid((x, 29), "col")
            g.set_torch((x, 29), 0)
            g.set_solid((x, 31), "col")
            g.set_torch((x, 31), 0)
        big_at_xa = S[a][b - 1] >= S[bb][b - 1]
        # 大值在 xa ⇒ 减法朝 xb（差值结果落到 xb）；反之朝 xa
        C2((xa + 1, 27), "W" if big_at_xa else "E", "subtract")
        # z=28 的充电比较器必须放在"减法 rear 所在的那一侧"
        C2((xa, 28) if big_at_xa else (xb, 28), "S")
        Sa = int(max(S[a][b - 1], S[bb][b - 1]))       # 环里烧录的大值
        C2((xa + 1, 29), "E", "compare", Sa)
        C2((xa, 30), "N", "compare", Sa)
        C2((xb, 30), "S", "compare", Sa)
        C2((xa + 1, 31), "W", "compare", Sa)
    return g


if __name__ == "__main__":
    P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
    mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                        encoding="utf-8"))
    W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
    DELTA, WREF, B, N = 7.0, 882.0, 8, 4
    alpha = WREF / (np.abs(W1).max(1) * DELTA / 4)
    U = np.rint(alpha[:, None, None] * W1[:, :, None] * (DELTA / (1 << B))
                * (1 << np.arange(B))[None, None, :]).astype(np.int64)
    J = 8
    S = (15 + np.abs(U[J])).astype(int)
    msb = np.abs(U[J, :, B - 1]).tolist()
    np.random.seed(1)
    inp = np.random.randint(0, 2, size=(N, B))
    K = int(S.sum())
    Ntrue = int((U[J] * inp).sum() - np.minimum(0, U[J]).sum())
    print(f"K={K}  N_true={Ntrue}")

    for flip in (False, True):
        g = build(N, B, S, K, K // 2, inp, flip, msb, wsign=(U[J] > 0).astype(int))
        it, ok = g.run()
        nz = {p: c["state"] for p, c in g.cmp.items() if c["state"]}
        tag = "翻转" if flip else "不翻转"
        print()
        print(f"=== {tag} ===  迭代 {it} 收敛 {ok}  非零 {len(nz)}")
        for p in [(2, 21), (2, 3), (11, 22), (11, 4), (10, 4), (3, 4),
                  (3, 3), (1, 24), (12, 23)]:
            if p in g.cmp:
                c = g.cmp[p]
                print(f"   {str(p):10s} out={c['out']} {c['mode']:8s} "
                      f"= {c['state']}")
        # 各链末端
        print("   四条链末端:", [(10, z, g.cmp[(10, z)]["state"])
                                for z in (4, 10, 16, 22) if (10, z) in g.cmp])
    print("SIM3_DONE")
