# -*- coding: utf-8 -*-
"""把一份 schem 打成"俯视表格"：每个格子给一个短码 + 数值，方便读电路。

短码：
    c<方向><值>  比较器（compare）      s<方向><值>  比较器（subtract）
    T            火把（亮）             t            火把（灭）
    R O Y L C B P M W  混凝土（红橙黄绿青蓝紫白）
    ?            其它
用法：python src/schem_grid.py <schem> [--values]
"""
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schem_verify_stack import R, varints  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DIR = {"north": "N", "south": "S", "east": "E", "west": "W"}
COL = {"red": "R", "orange": "O", "yellow": "Y", "lime": "L", "cyan": "C",
       "blue": "B", "purple": "P", "magenta": "M", "white": "W",
       "light_gray": "g", "pink": "k"}


def main():
    path = sys.argv[1]
    show_val = "--values" in sys.argv
    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw); r.u1(); r.name()
    sch = r.payload(10)["Schematic"]
    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    blk = sch["Blocks"]
    pal = blk["Palette"]
    inv = {v: k for k, v in pal.items()}
    idx = varints(blk["Data"])
    be = {(e["Id"], tuple(e["Pos"])): e.get("Data", {}) for e in blk.get("BlockEntities", [])}

    print(f"{os.path.basename(path)}  {W} x {H} x {L}")

    if H > 1:
        ys = sorted({i // (W * L) for i in range(len(idx))})
        print("高度 > 1，逐层打印。")
    for y in range(H):
        print(f"\n--- y = {y} ---")
        hdr = "     " + "".join("%-9d" % x for x in range(W))
        print(hdr)
        for z in range(L):
            row = []
            for x in range(W):
                st = inv[idx[(y * L + z) * W + x]]
                if st == "minecraft:air":
                    row.append("    .    ")
                    continue
                if "comparator" in st:
                    mode = "s" if "subtract" in st else "c"
                    d = st.split("facing=")[1].split(",")[0]
                    v = be.get(("minecraft:comparator", (x, y, z)), {}).get("OutputSignal", 0)
                    row.append(("%s%-6s" % (mode + DIR[d], v) if show_val
                                else "%-4s" % (mode + DIR[d] + str(v))))
                elif "wall_torch" in st:
                    row.append("%-8s" % ("T" if "lit=true" in st else "t"))
                elif "redstone_torch" in st:
                    row.append("%-8s" % ("T^" if "lit=true" in st else "t^"))
                else:
                    m = st.replace("minecraft:", "").replace("_concrete", "")
                    row.append("%-8s" % COL.get(m, "?"))
            print("z=%2d " % z + "".join(row))


if __name__ == "__main__":
    main()
