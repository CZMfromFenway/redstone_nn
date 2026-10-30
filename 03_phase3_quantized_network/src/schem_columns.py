# -*- coding: utf-8 -*-
"""按 (x,z) 列打印一份 .schem 的竖直结构：每一列里 y 从低到高是什么方块。

用途：看火把塔这类"只在 y 方向延伸、占地不变"的结构。

用法：python src/schem_columns.py <schem> [--min-y N] [--quiet-air]
"""
import gzip
import struct
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class R:
    def __init__(s, b):
        s.b, s.i = b, 0

    def u1(s):
        v = s.b[s.i]; s.i += 1; return v

    def u2(s):
        v = struct.unpack(">H", s.b[s.i:s.i + 2])[0]; s.i += 2; return v

    def name(s):
        n = s.u2(); v = s.b[s.i:s.i + n].decode("utf-8", "replace"); s.i += n; return v

    def payload(s, t):
        if t == 1:
            v = struct.unpack(">b", s.b[s.i:s.i + 1])[0]; s.i += 1; return v
        if t == 2:
            v = struct.unpack(">h", s.b[s.i:s.i + 2])[0]; s.i += 2; return v
        if t == 3:
            v = struct.unpack(">i", s.b[s.i:s.i + 4])[0]; s.i += 4; return v
        if t == 4:
            v = struct.unpack(">q", s.b[s.i:s.i + 8])[0]; s.i += 8; return v
        if t == 5:
            v = struct.unpack(">f", s.b[s.i:s.i + 4])[0]; s.i += 4; return v
        if t == 6:
            v = struct.unpack(">d", s.b[s.i:s.i + 8])[0]; s.i += 8; return v
        if t == 7:
            n = struct.unpack(">i", s.b[s.i:s.i + 4])[0]; s.i += 4
            v = s.b[s.i:s.i + n]; s.i += n; return v
        if t == 8:
            n = s.u2(); v = s.b[s.i:s.i + n].decode("utf-8", "replace"); s.i += n; return v
        if t == 9:
            et = s.u1(); n = struct.unpack(">i", s.b[s.i:s.i + 4])[0]; s.i += 4
            return [s.payload(et) for _ in range(n)]
        if t == 10:
            d = {}
            while True:
                tt = s.u1()
                if tt == 0:
                    return d
                nm = s.name(); d[nm] = s.payload(tt)
        if t == 11:
            n = struct.unpack(">i", s.b[s.i:s.i + 4])[0]; s.i += 4
            v = list(struct.unpack(">" + "i" * n, s.b[s.i:s.i + 4 * n])); s.i += 4 * n; return v
        if t == 12:
            n = struct.unpack(">i", s.b[s.i:s.i + 4])[0]; s.i += 4
            v = list(struct.unpack(">" + "q" * n, s.b[s.i:s.i + 8 * n])); s.i += 8 * n; return v
        raise ValueError("tag %d @ %d" % (t, s.i))


def varints(bd):
    out, val, sh = [], 0, 0
    for b in bd:
        val |= (b & 0x7F) << sh; sh += 7
        if not (b & 0x80):
            out.append(val); val, sh = 0, 0
    return out


def load(path):
    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw); r.u1(); r.name()
    sch = r.payload(10)["Schematic"]
    return sch, raw, r


def main():
    path = sys.argv[1]
    show_air = "--quiet-air" not in sys.argv
    min_y = 0
    if "--min-y" in sys.argv:
        min_y = int(sys.argv[sys.argv.index("--min-y") + 1])

    sch, raw, r = load(path)
    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    blk = sch["Blocks"]
    pal = blk["Palette"]
    inv = {v: k for k, v in pal.items()}
    data = varints(blk["Data"])
    be = blk.get("BlockEntities", [])

    print(f"{path}")
    print(f"  尺寸 W={W} H={H} L={L}   格数 {len(data)}   读到 {r.i}/{len(raw)}")
    print(f"  调色板 {len(pal)}:")
    for k, v in sorted(pal.items(), key=lambda kv: kv[1]):
        print(f"    [{v:2d}] {k}")
    print(f"  BlockEntities {len(be)}")
    if be:
        c = Counter(e.get("Id", "?") for e in be)
        print(f"    种类: {dict(c)}")

    def at(x, y, z):
        return inv[data[(y * L + z) * W + x]]

    print("\n按列（y 从低到高，只显示非空气段）：")
    for z in range(L):
        for x in range(W):
            col = [at(x, y, z) for y in range(H)]
            if all(v == "minecraft:air" for v in col):
                continue
            runs = []
            y = 0
            while y < H:
                if col[y] == "minecraft:air" and not show_air:
                    y += 1
                    continue
                name, y0 = col[y], y
                while y + 1 < H and col[y + 1] == name:
                    y += 1
                runs.append((y0, y, name))
                y += 1
            seg = "  ".join(
                (f"y{a}..{b} {nm.replace('minecraft:', '')}" if b > a
                 else f"y{a} {nm.replace('minecraft:', '')}") for a, b, nm in runs if b >= min_y)
            print(f"  (x={x:2d}, z={z:2d}) | {seg}")

    print("\n每种方块的 y 分布：")
    for k in sorted(set(inv[i] for i in data)):
        ys = [d // (W * L) for d, i in enumerate(data) if inv[i] == k]
        if not ys:
            continue
        print(f"  {k:52s} 共 {len(ys):5d}   y ∈ [{min(ys)}, {max(ys)}]"
              f"   不同 y 数 {len(set(ys))}")


if __name__ == "__main__":
    main()
