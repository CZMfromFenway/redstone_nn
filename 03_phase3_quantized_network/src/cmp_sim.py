#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
比较器电路静态模拟器。

规则（按用户给定）
  1. 比较器的 facing 与信号输出方向相反：输出方向 = flip(facing)
  2. 比较器只有在 rear 满足下列之一时才有源：
       a) rear 是另一个比较器（且它朝我们输出）
       b) rear 方块被某个比较器充能（充能值 = 那个比较器的输出）
       c) rear 是容器（木桶）
       d) rear 方块的更后面是容器（穿透方块读容器）
     否则没有源 → 输出 0
  3. 比较器把输出值"充"到它指向的方块
  4. 彩色输入方块被火把充能时优先级最高 → 读它的比较器直接得 15
  5. 静态：迭代到不动点。环靠初值（烧录常数）维持
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}
DIRV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


class Board:
    def __init__(self, W, L):
        self.W, self.L = W, L
        self.solid = {}      # pos -> 类型字符串
        self.barrel = {}     # pos -> 信号值
        self.cmp = {}        # pos -> dict(out_dir, mode, state, init)
        self.torch = {}      # pos -> 15/0（彩色输入方块）
        self.ext = {}        # pos -> 外部注入的充能（模拟烧录常数/源）

    def set_solid(self, p, kind="solid"):
        self.solid[p] = kind

    def set_barrel(self, p, v):
        self.barrel[p] = v

    def set_cmp(self, p, out_dir, mode="compare", init=0):
        self.cmp[p] = dict(out=out_dir, mode=mode, state=init, init=init)

    def set_torch(self, p, v):
        self.torch[p] = v

    def inside(self, p):
        x, z = p
        return 0 <= x < self.W and 0 <= z < self.L

    # ---------- 读一个位置对"从 toward 方向来的比较器"提供的信号
    # side=True 时只有合法侧源：比较器 / 红石块 / 红石线 / 中继器
    #（本图里没有红石块与中继器，所以侧输入只认比较器）
    def read(self, p, toward, side=False):
        if not self.inside(p):
            return 0
        if side:
            if p in self.cmp:
                c = self.cmp[p]
                return c["state"] if DIRV[c["out"]] == toward else 0
            return 0
        # 4. 火把充能的彩色方块：优先级最高（只在其亮时截断到 15）
        if p in self.torch and self.torch[p] > 0:
            return self.torch[p]
        # a) rear 是比较器
        if p in self.cmp:
            c = self.cmp[p]
            # 它必须朝我们输出
            if DIRV[c["out"]] == toward:
                return c["state"]
            return 0
        # c) rear 是容器
        if p in self.barrel:
            return self.barrel[p]
        # b) rear 方块被比较器充能
        if p in self.solid:
            if p in self.ext:
                return self.ext[p]
            charged = []
            for q, c in self.cmp.items():
                qx, qz = q
                dx, dz = DIRV[c["out"]]
                if (qx + dx, qz + dz) == p:
                    charged.append(c["state"])
            if charged:
                return max(charged)
            # d) 穿透方块读后面的容器（后面 = 远离读取者的一侧）
            dx, dz = -toward[0], -toward[1]
            back = (p[0] + dx, p[1] + dz)
            if back in self.barrel:
                return self.barrel[back]
            return 0
        return 0

    def step(self):
        new = {}
        for p, c in self.cmp.items():
            ox, oz = DIRV[c["out"]]
            rear = (p[0] - ox, p[1] - oz)
            rv = self.read(rear, toward=DIRV[c["out"]])
            # 侧输入：垂直方向。读取者在本体 p，位置 p+s 处的方块要朝 -s 输出
            sd = [DIRV["N"] if ox else DIRV["E"], DIRV["S"] if ox else DIRV["W"]]
            sv = max(self.read((p[0] + s[0], p[1] + s[1]),
                               toward=(-s[0], -s[1]), side=True) for s in sd)
            if c["mode"] == "compare":
                v = rv if rv >= sv else 0
            else:
                v = max(0, rv - sv)
            new[p] = v
        changed = False
        for p, v in new.items():
            if self.cmp[p]["state"] != v:
                changed = True
        for p, v in new.items():
            self.cmp[p]["state"] = v
        return changed

    def run(self, maxit=200):
        for i in range(maxit):
            if not self.step():
                return i + 1, True
        return maxit, False


# ---------------- 版面
def build(channel_msb, S, grey_K, white_P, inp):
    W, L = 14, 32
    b = Board(W, L)
    ODD, EVEN = [3, 5, 7, 9], [4, 6, 8, 10]
    B, N = 8, 4

    # 灰环 K：中心 (1,25)
    for (x, z, d) in ((1, 24, "E"), (2, 25, "S"), (1, 26, "W"), (0, 25, "N")):
        # 输出方向：顺时针闭环 (1,24)->(2,24)->(2,25)->(2,26)->(1,26)->(0,26)->(0,25)->(0,24)->(1,24)
        b.set_cmp((x, z), d, "compare", init=grey_K)
    for (x, z) in ((0, 24), (2, 24), (0, 26), (2, 26)):
        b.set_solid((x, z), "base")

    # 白环 P：中心 (12,24)
    for (x, z, d) in ((12, 23, "W"), (13, 24, "N"), (12, 25, "E"), (11, 24, "S")):
        # 输出方向：闭环 (12,23)->(11,23)->(11,24)->(11,25)->(12,25)->(13,25)->(13,24)->(13,23)->(12,23)
        b.set_cmp((x, z), d, "compare", init=white_P)
    for (x, z) in ((11, 23), (13, 23), (11, 25), (13, 25)):
        b.set_solid((x, z), "white")

    # 输入行 + 木桶（位 0..6）
    for k in range(B - 1):
        col = ODD if k % 2 == 0 else EVEN
        zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
        zb = 3 * k + 1 if k % 2 == 0 else 3 * k + 4
        for i in range(N):
            b.set_solid((col[i], zc), "col")          # 彩色输入方块也是实体
            b.set_torch((col[i], zc), 15 * inp[i][k])
            b.set_barrel((col[i], zb), int(S[i][k]))

    # 周期：cmp 行 + 减法链
    for j in range(4):
        z3, z4, z5 = 6 * j + 3, 6 * j + 4, 6 * j + 5
        b.set_cmp((2, z3), "N")
        for x in ODD:
            b.set_cmp((x, z3), "S")
        for x in ODD + EVEN:
            b.set_cmp((x, z4), "E", "subtract")
        b.set_cmp((2, z5), "N")
        for x in EVEN:
            b.set_cmp((x, z5), "N")

    # 左总线 x=2：灰环在左下 ⇒ 信号自下而上（输出 N）
    for z in range(1, 24):
        p = (2, z)
        if p in b.cmp:
            continue
        if z % 2 == 0:
            b.set_solid(p, "base")
        else:
            b.set_cmp(p, "N")

    # 右总线 x=11，4 个减法（白环在下，信号向下汇入）
    SUBS = {4, 10, 16, 22}
    for z in range(4, 23):
        p = (11, z)
        if z in SUBS:
            b.set_cmp(p, "S", "subtract")
        elif z % 2 == 0:
            b.set_cmp(p, "S")
        else:
            b.set_solid(p, "white")

    # MSB 行 z=24
    for i in range(N):
        b.set_torch((EVEN[i], 24), 15 * inp[i][B - 1])

    # 配对区
    for pi, (a, bb) in enumerate([(0, 1), (2, 3)]):
        xa, xb = EVEN[a], EVEN[bb]
        lo, hi = min(channel_msb[a], channel_msb[bb]), max(channel_msb[a], channel_msb[bb])
        case = "I" if lo > 882 else ("II" if hi > 882 else "III")
        if case == "III":
            b.set_barrel((xa, 25), int(S[a][B - 1]))
            b.set_barrel((xb, 25), int(S[bb][B - 1]))
            continue
        hi_i = a if S[a][B - 1] >= S[bb][B - 1] else bb
        lo_i = bb if hi_i == a else a
        b.set_cmp((xa, 25), "N"); b.set_cmp((xb, 25), "N")
        b.set_barrel((xa + 1, 25), int(S[hi_i][B - 1] - S[lo_i][B - 1]))
        b.set_cmp((xa, 26), "N"); b.set_cmp((xb, 26), "N")
        if case == "I":
            b.set_cmp((xa + 1, 26), "S")
            b.set_cmp((xa, 27), "E"); b.set_cmp((xb, 27), "E")
            b.set_cmp((xa + 1, 27), "E", "subtract")
        b.set_cmp((xb, 28), "N")
        b.set_cmp((xa, 29), "E"); b.set_cmp((xb, 29), "E")
        b.set_cmp((xa + 1, 29), "W")
        b.set_cmp((xa, 30), "S"); b.set_cmp((xb, 30), "N")
        b.set_cmp((xa, 31), "E"); b.set_cmp((xb, 31), "E")
        b.set_cmp((xa + 1, 31), "E")
    return b


if __name__ == "__main__":
    import numpy as np
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

    inp = np.zeros((N, B), dtype=int)
    inp[0] = [1, 0, 1, 1, 0, 1, 0, 1]
    inp[1] = [0, 1, 1, 0, 1, 0, 1, 1]
    inp[2] = [1, 1, 0, 0, 1, 1, 0, 0]
    inp[3] = [0, 0, 1, 1, 0, 0, 1, 1]

    brd = build(msb, S, 0, 0, inp)
    print("比较器总数:", len(brd.cmp))
    it, ok = brd.run()
    print(f"[环初值 0] 迭代 {it} 次，收敛 {ok}，"
          f"非零 {sum(1 for c in brd.cmp.values() if c['state'])}")

    # 按烧录常数给两个环赋初值
    K = 15 * N * B + int(S.sum())
    P = 0
    brd2 = build(msb, S, K, P, inp)
    it2, ok2 = brd2.run()
    nz = {p: c["state"] for p, c in brd2.cmp.items() if c["state"]}
    print(f"[灰环烧 K={K}] 迭代 {it2} 次，收敛 {ok2}，非零 {len(nz)}")
    for p in [(1, 24), (2, 24), (2, 23), (2, 21), (2, 19), (2, 3),
              (3, 4), (5, 4), (9, 4), (10, 4), (11, 4),
              (11, 22), (11, 10), (11, 6), (12, 23)]:
        kind = ("cmp" if p in brd2.cmp else
                "base" if brd2.solid.get(p) else
                "barrel" if p in brd2.barrel else "air")
        if p in brd2.cmp:
            c = brd2.cmp[p]
            print(f"  {str(p):12s} cmp  out={c['out']} {c['mode']:8s} -> {c['state']}")
        else:
            print(f"  {str(p):12s} {kind:6s} "
                  f"读值={brd2.read(p, toward=DIRV['N'])}")
    print("SIM_DONE")
