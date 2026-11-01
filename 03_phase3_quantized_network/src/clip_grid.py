# -*- coding: utf-8 -*-
"""把 data/relu.schem（8×1×26，单层）打成字符表。"""
import gzip
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_census import R, read_varints  # noqa

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SYM = {
    "minecraft:comparator[facing=east,mode=compare,powered=false]": "a",
    "minecraft:comparator[facing=west,mode=compare,powered=false]": "b",
    "minecraft:comparator[facing=south,mode=compare,powered=false]": "c",
    "minecraft:comparator[facing=north,mode=compare,powered=false]": "d",
    "minecraft:comparator[facing=east,mode=subtract,powered=false]": "e",
    "minecraft:comparator[facing=south,mode=subtract,powered=false]": "f",
    "minecraft:red_concrete": "R",
    "minecraft:orange_concrete": "O",
    "minecraft:yellow_concrete": "Y",
    "minecraft:lime_concrete": "L",
    "minecraft:cyan_concrete": "C",
    "minecraft:blue_concrete": "B",
    "minecraft:purple_concrete": "P",
    "minecraft:magenta_concrete": "M",
    "minecraft:white_concrete": "W",
    "minecraft:redstone_wall_torch[facing=north,lit=true]": "t",
    "minecraft:air": ".",
}


def load(path):
    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw)
    r.u1()
    r.name()
    top = r.payload(10)
    key = [k for k in top if k != ""][0]
    sch = top[key]
    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    pal = {v: k for k, v in sch["Blocks"]["Palette"].items()}
    idx = read_varints(sch["Blocks"]["Data"])
    ents = {(tuple(e["Pos"]), e.get("Data", {}).get("OutputSignal")) for e in sch["Blocks"]["BlockEntities"]}
    grid = {}
    for y in range(H):
        for z in range(L):
            for x in range(W):
                grid[(x, y, z)] = pal[idx[(y * L + z) * W + x]]
    return W, H, L, grid, dict((p, s) for p, s in ents)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "relu.schem")
    W, H, L, grid, ents = load(path)
    print(f"{os.path.basename(path)}  {W}x{H}x{L}")
    print("图例 a=compare读东 b=compare读西 c=compare读南 d=compare读北 e=subtract读东 f=subtract读南")
    print("     R O Y L C B P M = 八色 W=白 t=火把 . =空气")
    print()
    print("      " + "".join(str(x) for x in range(W)))
    for z in range(L):
        row = "".join(SYM.get(grid[(x, 0, z)], "?") for x in range(W))
        print(f" z={z:2d} {row}")
    print()
    print("比较器的 OutputSignal（非零才列）:")
    nz = [(p, s) for p, s in sorted(ents.items()) if s]
    print("  ", nz if nz else "全部为 0（环未烧）")
    print()
    print("每个 x 列里出现的符号序列（沿 z）:")
    for x in range(W):
        col = "".join(SYM.get(grid[(x, 0, z)], "?") for z in range(L))
        print(f"  x={x}: {col}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
