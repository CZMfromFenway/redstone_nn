#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
把电路抽象成**有向图**并做诊断。

节点 = 每个方块/比较器
边：
   比较器 --出--> 它指向的方块（充能边）
   方块   --读--> 比较器（该比较器的 rear 是这个方块）
   容器   --读--> 比较器（rear 是容器）
   比较器 --读--> 比较器（rear 处是比较器且它朝我输出）

诊断两件事
   A. 每个比较器的 rear/side 到底连到什么（图上的入边）
   B. 环是不是真的闭环（每个节点的入边都指向环内）
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
sys.path.insert(0, HERE)
from cmp_sim import Board, DIRV  # noqa: E402
from cmp_sim3 import build  # noqa: E402

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

g = build(N, B, S, K, K, inp, True, msb)
g.run()


def kind(p):
    if p in g.cmp:
        return "cmp"
    if p in g.barrel:
        return "barrel"
    if p in g.torch:
        return "torch%d" % int(g.torch[p])
    if p in g.solid:
        return g.solid[p]
    return "air"


def chargers_of(p):
    out = []
    for q, c in g.cmp.items():
        d = DIRV[c["out"]]
        if (q[0] + d[0], q[1] + d[1]) == p:
            out.append(q)
    return out


print("=" * 96)
print("A. 配对区（品红下方）每个比较器的入边")
print("=" * 96)
targets = []
for pi in range(2):
    xa = [4, 8][pi]
    xb = [6, 10][pi]
    targets += [(xa + 1, 29), (xb, 30), (xa + 1, 31), (xa, 30),
                (xa, 25), (xb, 25), (xa, 26), (xa + 1, 26), (xb, 26),
                (xa + 1, 27), (xa, 28), (xa + 1, 31)]
for p in dict.fromkeys(targets):
    if p not in g.cmp:
        print(f"  {str(p):10s} 不是比较器（{kind(p)}）")
        continue
    c = g.cmp[p]
    d = DIRV[c["out"]]
    r = (p[0] - d[0], p[1] - d[1])
    chg = chargers_of(r)
    sd = [DIRV["N"] if d[0] else DIRV["E"], DIRV["S"] if d[0] else DIRV["W"]]
    sp = [(p[0] + s[0], p[1] + s[1]) for s in sd]
    print(f"  {str(p):10s} out={c['out']} {c['mode']:8s} = {int(c['state'])}")
    print(f"      rear {str(r):10s} = {kind(r):10s} "
          f"充能者 {chargers_of(r) if chg else '无'}")
    for s, sp_ in zip(sd, sp):
        print(f"      side {str(sp_):10s} = {kind(sp_):10s} "
              f"充能者 {chargers_of(sp_) if chargers_of(sp_) else '无'}")

print()
print("=" * 96)
print("B. 环的闭环性检查")
print("=" * 96)
for nm, ring in (
    ("灰环 K", [(1, 24), (2, 25), (1, 26), (0, 25)]),
    ("白环 P", [(12, 23), (13, 24), (12, 25), (11, 24)]),
    ("品红环 pair0", [(4, 29), (5, 29), (6, 30), (5, 31), (4, 30)]),
):
    print(f"\n  【{nm}】")
    for p in ring:
        if p not in g.cmp:
            print(f"    {str(p):10s} 不是比较器（{kind(p)}）  ← 环断开")
            continue
        c = g.cmp[p]
        d = DIRV[c["out"]]
        r = (p[0] - d[0], p[1] - d[1])
        ok = r in ring or any((q[0] + DIRV[g.cmp[q]["out"]][0],
                               q[1] + DIRV[g.cmp[q]["out"]][1]) == r
                              for q in ring if q in g.cmp)
        print(f"    {str(p):10s} out={c['out']} rear={str(r):10s} "
              f"({kind(r)})  环内? {'✓' if ok else '✗'}  = {int(c['state'])}")

print()
print("=" * 96)
print("C. 全局：没有入边的比较器（死节点）")
print("=" * 96)
dead = []
for p, c in g.cmp.items():
    d = DIRV[c["out"]]
    r = (p[0] - d[0], p[1] - d[1])
    has = (r in g.cmp and DIRV[g.cmp[r]["out"]] == d) or r in g.barrel \
        or chargers_of(r) or (r in g.torch and g.torch[r] > 0)
    if not has:
        # 穿透检查
        if r in g.solid:
            back = (r[0] - d[0], r[1] - d[1])
            if back in g.barrel:
                has = True
    if not has:
        dead.append(p)
print(f"  总数 {len(g.cmp)}，无入边 {len(dead)}")
for p in dead[:30]:
    print(f"    {p} out={g.cmp[p]['out']}")
print("GRAPH_DONE")
