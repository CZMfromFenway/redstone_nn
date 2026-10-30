#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
回到 renn_sr.schem 的原始块表，把品红配对区抽成有向图，两种朝向约定各算一次。

判定标准很简单：**哪种约定下每个比较器都能找到 rear 源**
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
g = json.load(open(os.path.join(RESULTS, "renn_sr_grid.json"), encoding="utf-8"))
G, W, L = g["grid"], g["W"], g["L"]

DIRV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def kind(x, z):
    if not (0 <= x < W and 0 <= z < L):
        return "out"
    v = G[z][x]
    if v == "minecraft:air":
        return "air"
    if v.startswith("minecraft:comparator"):
        return "cmp"
    if v.startswith("minecraft:barrel"):
        return "barrel"
    return "solid"


def facing(x, z):
    v = G[z][x]
    if not v.startswith("minecraft:comparator"):
        return None
    f = v.split("facing=")[1].split(",")[0]
    return {"north": "N", "south": "S", "east": "E", "west": "W"}[f]


def is_sub(x, z):
    return "subtract" in G[z][x]


# 先精确列出配对区
print("=" * 100)
print("renn_sr.schem 配对区原始块表（pair0: x=4..6, pair1: x=8..10）")
print("=" * 100)
for z in range(23, L):
    items = []
    for x in range(3, 11):
        v = G[z][x]
        if v == "minecraft:air":
            continue
        if v.startswith("minecraft:comparator"):
            f = facing(x, z)
            items.append(f"{x}:{'减' if is_sub(x, z) else '比'}{f[0].upper()}")
        elif v.startswith("minecraft:barrel"):
            items.append(f"{x}:桶")
        elif "magenta" in v:
            items.append(f"{x}:品")
        else:
            items.append(f"{x}:{v.split(':')[1][:4]}")
    print(f"  z={z:2d}  " + "  ".join(items))


def analyse(flip, xs, zs):
    """返回 (valid, invalid, detail)"""
    valid, invalid, detail = 0, 0, []
    for z in zs:
        for x in xs:
            if kind(x, z) != "cmp":
                continue
            f = facing(x, z)
            out = FLIP[f] if flip else f
            d = DIRV[out]
            r = (x - d[0], z - d[1])
            k = kind(*r)
            src = None
            if k == "cmp":
                # 那个比较器必须朝我输出
                rf = facing(*r)
                rout = FLIP[rf] if flip else rf
                if DIRV[rout] == d:
                    src = f"cmp@{r}"
            elif k == "barrel":
                src = f"桶@{r}"
            elif k == "solid":
                # 被比较器充能？
                ch = []
                for zz in zs:
                    for xx in xs:
                        if kind(xx, zz) != "cmp":
                            continue
                        ff = facing(xx, zz)
                        oo = FLIP[ff] if flip else ff
                        dd = DIRV[oo]
                        if (xx + dd[0], zz + dd[1]) == r:
                            ch.append((xx, zz))
                if ch:
                    src = f"充能@{r} by {ch}"
                else:
                    back = (r[0] - d[0], r[1] - d[1])
                    if kind(*back) == "barrel":
                        src = f"穿透桶@{back}"
            if src:
                valid += 1
                detail.append(((x, z), out, "OK", src, k))
            else:
                invalid += 1
                detail.append(((x, z), out, "无源", f"rear={r}({k})", k))
    return valid, invalid, detail


print()
print("=" * 100)
print("两种约定下，配对区比较器的 rear 源")
print("=" * 100)
XS = list(range(4, 11))
ZS = list(range(24, L))
for flip in (False, True):
    v, iv, det = analyse(flip, XS, ZS)
    print(f"\n  【{'翻转' if flip else '不翻转'}】 有源 {v}  无源 {iv}")
    for (p, out, st, src, k) in det:
        mark = "✓" if st == "OK" else "✗"
        print(f"    {str(p):10s} out={out}  {mark} {st:4s}  {src}")

print()
print("=" * 100)
print("环的闭环检查（pair0 下环 z=29..31）")
print("=" * 100)
ring = [(4, 29), (5, 29), (6, 29), (6, 30), (6, 31), (5, 31), (4, 31), (4, 30)]
for flip in (False, True):
    print(f"\n  【{'翻转' if flip else '不翻转'}】")
    for p in ring:
        x, z = p
        k = kind(x, z)
        if k != "cmp":
            print(f"    {str(p):10s} {k:6s}（导线/角块）")
            continue
        f = facing(x, z)
        out = FLIP[f] if flip else f
        d = DIRV[out]
        r = (x - d[0], z - d[1])
        rk = kind(*r)
        src = "?"
        if rk == "cmp":
            rf = facing(*r)
            rout = FLIP[rf] if flip else rf
            src = "OK(cmp)" if DIRV[rout] == d else f"朝向不符({rout}≠{out})"
        elif rk == "solid":
            ch = [q for q in ring if kind(*q) == "cmp"
                  and (q[0] + DIRV[(FLIP[facing(*q)] if flip else facing(*q))][0],
                       q[1] + DIRV[(FLIP[facing(*q)] if flip else facing(*q))][1]) == r]
            src = f"充能by{ch}" if ch else "无充能者"
        else:
            src = rk
        print(f"    {str(p):10s} out={out} rear={str(r):10s} {src}")
print("RS_DONE")
