# -*- coding: utf-8 -*-
"""把 data/relu.schem 每比特（3 行）的接线打出来：每个比较器读谁、写谁。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_grid import load, SYM  # noqa

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DIRV = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
NAME = {v: k for k, v in DIRV.items()}


def kind(cell):
    if cell is None:
        return "界外"
    if cell == "minecraft:air":
        return "空气"
    if "comparator" in cell:
        fac = cell.split("facing=")[1].split(",")[0]
        mode = cell.split("mode=")[1].split(",")[0]
        return f"比较器[{mode},读{ {'north':'北','south':'南','east':'东','west':'西'}[fac] }]"
    if "torch" in cell:
        fac = cell.split("facing=")[1].split(",")[0] if "facing=" in cell else "-"
        return f"火把[{fac}]"
    if "concrete" in cell:
        return "块" + SYM.get(cell, "?")
    return cell


def main():
    path = os.path.join(ROOT, "data", "relu.schem")
    W, H, L, grid, ents = load(path)
    g = lambda x, z: grid.get((x, 0, z))
    print("=" * 96)
    print("每比特的接线（除最后一段 z=22..24 只有 7 个块，其余 8 段结构相同）")
    print("=" * 96)
    for k in range(8):
        z0 = 3 * k + 1
        print(f"\n--- 第 {k} 比特：z = {z0}, {z0+1}, {z0+2}（色 {SYM[g(2,z0)]}）---")
        for z in (z0, z0 + 1, z0 + 2):
            for x in range(W):
                c = g(x, z)
                if c is None or "comparator" not in c:
                    continue
                fac = c.split("facing=")[1].split(",")[0]
                mode = c.split("mode=")[1].split(",")[0]
                dx, dz = DIRV[fac]
                rx, rz = x + dx, z + dz          # rear = facing 方向
                fx, fz = x - dx, z - dz          # front = 反方向
                print(f"  ({x},{z}) {mode:8s} 读{ {'north':'北','south':'南','east':'东','west':'西'}[fac] }"
                      f" ← ({rx},{rz}) {kind(g(rx,rz)):16s} 输出 → ({fx},{fz}) {kind(g(fx,fz))}")
    print()
    print("=" * 96)
    print("火把与白块的位置")
    print("=" * 96)
    for z in range(L):
        for x in range(W):
            c = g(x, z)
            if c and ("torch" in c or "white_concrete" in c):
                print(f"  ({x},{z})  {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
