#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
迭代剪枝 + 两种环位置变体。

剪枝判据（两类）
  1. 死比较器：它的输出落在"无人读"的位置
  2. 孤儿方块：既没有被任何比较器充能，也没有被任何比较器读作 rear
反复剪到不动点。

变体参数
  grey_up / white_up：总线是向上流（环在下方）还是向下流（环在上方）
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

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
ODD, EVEN = [3, 5, 7, 9], [4, 6, 8, 10]


def build(n, b, S, K, P, inp, msb, wsign, grey_up=True, white_up=True, gshift=2, in_rev=False, par_swap=False):
    W, L = 14, 32
    g = Board(W, L)
    B = b

    def C(p, fac, mode="compare", init=0):
        g.set_cmp(p, FLIP[fac], mode, init)

    # ---------------- 两个环的位置
    gz = (24 - gshift) if grey_up else (0 + gshift)
    wz = 23 if white_up else 1
    gdir = "up" if grey_up else "down"
    wdir = "up" if white_up else "down"

    # 灰环：四边依次 E/S/W/N（顺时针），朝上时中心 (1,25)，朝下时 (1,1)
    if gdir == "up":
        gr = [(1, gz, "E"), (2, gz + 1, "S"), (1, gz + 2, "W"), (0, gz + 1, "N")]
    else:
        gr = [(1, gz, "E"), (2, gz + 1, "S"), (1, gz + 2, "W"), (0, gz + 1, "N")]
    for (x, z, f) in gr:
        C((x, z), f, "compare", K)
    for (x, z) in ((0, gz), (2, gz), (0, gz + 2), (2, gz + 2)):
        g.set_solid((x, z), "base")

    # 白环：中心 (12,24) 或 (12,1)
    if wdir == "up":
        wr = [(12, wz, "W"), (13, wz + 1, "N"), (12, wz + 2, "E"), (11, wz + 1, "S")]
    else:
        wr = [(12, wz, "W"), (13, wz + 1, "N"), (12, wz + 2, "E"), (11, wz + 1, "S")]
    for (x, z, f) in wr:
        C((x, z), f, "compare", P)
    for (x, z) in ((11, wz), (13, wz), (11, wz + 2), (13, wz + 2)):
        g.set_solid((x, z), "white")

    # ---------------- 输入行 + 木桶
    def X(i):
        return (n - 1 - i) if in_rev else i
    # 列奇偶性：bit k 用哪一组列
    CE = EVEN if par_swap else ODD     # k 偶
    CO = ODD if par_swap else EVEN     # k 奇
    def COL(k, i):
        return (CE if k % 2 == 0 else CO)[X(i)]
    for k in range(b - 1):
        zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
        zb = 3 * k + 1 if k % 2 == 0 else 3 * k + 4
        for i in range(n):
            c = COL(k, i)
            on = (inp[i][k] == 1) != (wsign[i][k] > 0)
            g.set_solid((c, zc), "col")
            g.set_torch((c, zc), 15 if on else 0)
            g.set_barrel((c, zb), int(S[i][k]))

    # ---------------- 周期
    for j in range(4):
        z3, z4, z5 = 6 * j + 3, 6 * j + 4, 6 * j + 5
        for x in CE:
            C((x, z3), "N")
        for x in ODD + EVEN:
            C((x, z4), "W", "subtract")
        for x in CO:
            C((x, z5), "S")

    # ---------------- 左总线
    _r0 = 5 if grey_up else (gz + 3)
    _r1 = gz if grey_up else 23
    for z in range(_r0, _r1):
        p = (2, z)
        if p in g.cmp:
            continue
        if z % 2 == 0:
            g.set_solid(p, "base")
        else:
            C(p, "S" if grey_up else "N")
    if grey_up:
        g.set_solid((2, 4), "base")
    if (2, 4) not in g.solid and (2, 4) not in g.cmp:
        g.set_solid((2, 4), "base")

    # ---------------- 右总线
    for z in range(4, 23):
        p = (11, z)
        if z % 2 == 0:
            C(p, "S" if white_up else "N",
              "subtract" if z in (4, 10, 16, 22) else "compare")
        else:
            g.set_solid(p, "white")

    # ---------------- MSB 行
    for i in range(n):
        on = (inp[i][b - 1] == 1) != (wsign[i][b - 1] > 0)
        g.set_solid((COL(b - 1, i), 24), "col")
        g.set_torch((COL(b - 1, i), 24), 15 if on else 0)

    # ---------------- 配对区
    for pi, (a0, b0) in enumerate([(0, 1), (2, 3)]):
        # 映射到物理列后要重新排序，保证 xa < xb
        a, bb = a0, b0
        if COL(b - 1, a) > COL(b - 1, bb):
            a, bb = bb, a
        xa, xb = COL(b - 1, a), COL(b - 1, bb)
        lo, hi = min(msb[a], msb[bb]), max(msb[a], msb[bb])
        case = "I" if lo > 882 else ("II" if hi > 882 else "III")
        if case == "III":
            g.set_barrel((xa, 25), int(S[a][b - 1]))
            g.set_barrel((xb, 25), int(S[bb][b - 1]))
            continue
        big_xa = S[a][b - 1] >= S[bb][b - 1]
        g.set_barrel((xa + 1, 25), abs(int(S[a][b - 1] - S[bb][b - 1])))
        C((xa, 25), "S"); C((xb, 25), "S")
        C((xa, 26), "S"); C((xa + 1, 26), "N"); C((xb, 26), "S")
        for x in (xa, xb):
            for zz in (27, 29, 31):
                g.set_solid((x, zz), "col")
                g.set_torch((x, zz), 0)
        C((xa + 1, 27), "W" if big_xa else "E", "subtract")
        C((xa, 28) if big_xa else (xb, 28), "S")
        Sa = int(max(S[a][b - 1], S[bb][b - 1]))
        C((xa + 1, 29), "E", "compare", Sa)
        C((xa, 30), "N", "compare", Sa)
        C((xb, 30), "S", "compare", Sa)
        C((xa + 1, 31), "W", "compare", Sa)
    return g


def prune(g, verbose=False):
    """反复删除死比较器与孤儿方块，直到不动点。"""
    removed = {"cmp": [], "block": []}
    first = True
    while True:
        # --- 死比较器
        dead = []
        for p, c in g.cmp.items():
            d = DIRV[c["out"]]
            b = (p[0] + d[0], p[1] + d[1])
            used = False
            if b in g.cmp:
                cc = g.cmp[b]; bd = DIRV[cc["out"]]
                if (b[0] - bd[0], b[1] - bd[1]) == p:
                    used = True
                for s in [(-bd[1], bd[0]), (bd[1], -bd[0])]:
                    if (b[0] + s[0], b[1] + s[1]) == p and DIRV[c["out"]] == (-s[0], -s[1]):
                        used = True
            elif b in g.barrel or b in g.solid:
                for q, cc in g.cmp.items():
                    if q == p:
                        continue
                    if (q[0] - DIRV[cc["out"]][0], q[1] - DIRV[cc["out"]][1]) == b:
                        used = True
                        break
            if not used:
                dead.append(p)
        # --- 孤儿方块（非火把块、非环角）
        orphan = []
        for p in list(g.solid.keys()) + list(g.barrel.keys()):
            if p in g.torch and g.torch[p] > 0:
                continue
            charged = any((q[0] + DIRV[cc["out"]][0],
                           q[1] + DIRV[cc["out"]][1]) == p for q, cc in g.cmp.items())
            read = any((q[0] - DIRV[cc["out"]][0],
                        q[1] - DIRV[cc["out"]][1]) == p for q, cc in g.cmp.items())
            if not charged and not read:
                orphan.append(p)
        if not orphan:
            if not first:
                break
            first = False
            break
        for p in dead:
            g.cmp.pop(p, None)
            removed["cmp"].append(p)
        for p in orphan:
            g.solid.pop(p, None)
            g.barrel.pop(p, None)
            removed["block"].append(p)
    return removed


def render(g, fn, title):
    Wg, Lg = g.W, g.L
    PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
           (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
    C, PADL, PADT, LEG = 44, 60, 66, 250
    img = Image.new("RGB", (PADL + Wg*C + LEG, PADT + Lg*C + 12), (16, 16, 20))
    dr = ImageDraw.Draw(img)

    def T(xy, s, fill=(235, 235, 245), anchor="mm"):
        dr.text(xy, s, fill=fill, anchor=anchor)

    colbit = {}
    for k in range(8):
        col = ODD if k % 2 == 0 else EVEN
        zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
        for i in range(4):
            colbit[(col[i], zc)] = k
    for i in range(4):
        colbit[(4 + 2 * i, 24)] = 7

    for z in range(Lg):
        for x in range(Wg):
            p = (x, z)
            x0, y0 = PADL + x*C, PADT + z*C
            cx, cy = x0 + C//2, y0 + C//2
            if p in g.cmp:
                c = g.cmp[p]; sub = c["mode"] == "subtract"; v = int(c["state"])
                dr.rectangle([x0+1, y0+1, x0+C-2, y0+C-2],
                             fill=(72, 48, 62) if sub else (44, 44, 56),
                             outline=(215, 100, 100) if sub else (110, 110, 130))
                vv = DIRV[c["out"]]
                ac = (255, 95, 95) if sub else (255, 195, 85)
                dr.line([cx-vv[0]*8, cy-vv[1]*8-6, cx+vv[0]*8, cy+vv[1]*8-6],
                        fill=ac, width=2)
                dr.polygon([(cx+vv[0]*12, cy+vv[1]*12-6),
                            (cx+vv[0]*6+vv[1]*4, cy+vv[1]*6-vv[0]*4-6),
                            (cx+vv[0]*6-vv[1]*4, cy+vv[1]*6+vv[0]*4-6)], fill=ac)
                T((cx, cy+10), str(v), (120, 255, 150) if v else (110, 110, 120))
            elif p in g.barrel:
                dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3], fill=(150, 110, 70),
                             outline=(90, 60, 30))
                T((cx, cy), str(int(g.barrel[p])), (255, 245, 220))
            elif p in g.solid:
                if p in g.torch:
                    k = colbit.get(p, 0); on = g.torch[p] > 0
                    base = PAL[k % 8]
                    dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                                 fill=base if on else tuple(int(q*0.40) for q in base),
                                 outline=(215, 215, 215))
                    T((cx, cy), "15" if on else "·", (10, 10, 10) if on else (215, 215, 215))
                else:
                    wt = g.solid[p] == "white"
                    dr.rectangle([x0+2, y0+2, x0+C-3, y0+C-3],
                                 fill=(235, 235, 235) if wt else (110, 110, 120))
                    rv = g.read(p, toward=DIRV["N"])
                    if rv:
                        T((cx, cy), str(int(rv)), (10, 10, 10) if wt else (240, 240, 240))
    for x in range(Wg):
        T((PADL+x*C+C//2, PADT-18), str(x), (190, 190, 200))
    for z in range(Lg):
        T((6, PADT+z*C+C//2), f"z{z}", (190, 190, 200), anchor="lm")
    for t, s in enumerate(title):
        T((PADL+Wg*C+10, PADT+t*18), s, (228, 228, 238), anchor="la")
    img.save(fn)
    return os.path.getsize(fn)


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
    wsg = (U[J] > 0).astype(int)
    K = int(S.sum())
    W0 = 4 * K - 15 * N * B
    rng = np.random.default_rng(11)
    inp = rng.integers(0, 2, size=(N, B))
    Ntrue = int((U[J] * inp).sum() - np.minimum(0, U[J]).sum())

    print(f"K={K}  白环={W0}  N_true={Ntrue}")
    for (gu, wu, tag) in ((True, True, "灰上·白上"),
                          (False, False, "灰下·白下"),
                          (True, False, "灰上·白下"),
                          (False, True, "灰下·白上")):
        g = build(N, B, S, K, W0, inp, msb, wsg, grey_up=gu, white_up=wu)
        rm = {"cmp": [], "block": []}
        for _p in ([(2, 0), (2, 1), (2, 2), (2, 3)] if gu else []):
            if _p in g.solid:
                g.solid.pop(_p); rm["block"].append(_p)
            if _p in g.cmp:
                g.cmp.pop(_p); rm["cmp"].append(_p)
        g.run()
        _port = (11, 4) if wu else (11, 22)
        head = int(g.cmp.get(_port, {}).get("state", -1))
        fn = os.path.join(RESULTS, f"variant_{int(gu)}{int(wu)}.png")
        render(g, fn, [f"灰线{'上' if gu else '下'}  白线{'上' if wu else '下'}",
                       f"端口 {_port} = {head}", f"N = {Ntrue}",
                       f"剪掉 cmp {len(rm['cmp'])} 个", f"剪掉 block {len(rm['block'])} 个"])
        print(f"  {tag}: 头部={head:6d}  N={Ntrue:6d}  "
              f"{'OK' if head == Ntrue else 'X'}   剪掉 cmp{len(rm['cmp'])} "
              f"block{len(rm['block'])}")
    print("VAR_DONE")
