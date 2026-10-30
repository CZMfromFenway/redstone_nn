# -*- coding: utf-8 -*-
"""把层1 的 64 个通道竖直堆叠成一份 .schem。

约定
    通道 ch 的平面在 y = 2*ch + 1（小通道在下）
    比较器下方补一个白色混凝土支撑方块，位于 y = 2*ch
    因此相邻通道 y 差 2，总高度 = 2*64 = 128
    抽掉了火把塔：不放任何火把
"""
import gzip
import json
import os
import struct
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"E:\redstone_nn\03_phase3_quantized_network"
COLORS = ["red", "orange", "yellow", "lime", "cyan", "blue", "purple", "magenta"]
FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}
FULL = {"N": "north", "S": "south", "E": "east", "W": "west"}
SUPPORT = "minecraft:white_concrete"
SLOTS, MAXC = 27, 64


# ---------------- NBT ----------------
def _n(n):
    b = n.encode("utf-8")
    return struct.pack(">H", len(b)) + b


def t_byte(n, v):
    return b"\x01" + _n(n) + struct.pack(">b", v)


def t_short(n, v):
    return b"\x02" + _n(n) + struct.pack(">h", v)


def t_int(n, v):
    return b"\x03" + _n(n) + struct.pack(">i", v)


def t_long(n, v):
    return b"\x04" + _n(n) + struct.pack(">q", v)


def t_str(n, v):
    b = v.encode("utf-8")
    return b"\x08" + _n(n) + struct.pack(">H", len(b)) + b


def t_barr(n, d):
    return b"\x07" + _n(n) + struct.pack(">i", len(d)) + bytes(d)


def t_iarr(n, v):
    return b"\x0b" + _n(n) + struct.pack(">i", len(v)) + b"".join(struct.pack(">i", x) for x in v)


def t_list(n, et, items):
    return b"\x09" + _n(n) + bytes([et]) + struct.pack(">i", len(items)) + b"".join(items)


def t_comp(n, payload):
    return b"\x0a" + _n(n) + payload + b"\x00"


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return out


def items_for(sig):
    """信号 → [(slot, count)]，maxStack=1 的木斧，每格 ≤64。"""
    if sig <= 0:
        return [], 0
    n = -(-((sig - 1) * SLOTS) // 14)
    if n > SLOTS * MAXC:
        return None, n
    out, need, slot = [], n, 0
    while need > 0:
        c = min(MAXC, need)
        out.append((slot, c))
        need -= c
        slot += 1
    return out, n


def main():
    d = json.load(open(os.path.join(ROOT, "results", "planes64.json"), encoding="utf-8"))
    W, L, b = d["W"], d["L"], d["b"]
    planes = d["planes"]
    NC = len(planes)
    H = 2 * NC

    cells = {}          # (x,y,z) -> blockstate
    ents = []           # BlockEntity 载荷
    unreach = []

    for p in planes:
        ch = p["ch"]
        y = 2 * ch + 1
        ysup = 2 * ch
        for c in p["cells"]:
            x, z = c["x"], c["z"]
            if c["t"] == "cmp":
                mode = "subtract" if c["mode"] == "subtract" else "compare"
                pw = "true" if (c["st"] or 0) > 0 else "false"
                cells[(x, y, z)] = (f"minecraft:comparator[facing={FULL[FLIP[c['out']]]},"
                                    f"mode={mode},powered={pw}]")
                cells[(x, ysup, z)] = SUPPORT          # 比较器下方补支撑
                dat = t_str("id", "minecraft:comparator") + t_int("OutputSignal", int(c["st"] or 0))
                ents.append(t_str("Id", "minecraft:comparator")
                            + t_iarr("Pos", [x, y, z])
                            + t_comp("Data", dat) + b"\x00")
            elif c["t"] == "barrel":
                cells[(x, y, z)] = "minecraft:barrel[facing=up,open=false]"
                it, ntot = items_for(int(c["v"] or 0))
                if it is None:
                    unreach.append((ch, x, z, c["v"]))
                    it = []
                payload = b""
                for (sl, cnt) in it:
                    one = (t_int("count", cnt) + t_byte("Slot", sl)
                           + t_str("id", "minecraft:wooden_axe"))
                    payload += one + b"\x00"
                dat = (t_str("id", "minecraft:barrel")
                       + b"\x09" + _n("Items") + b"\x0a"
                       + struct.pack(">i", len(it)) + payload
                       + t_str("CustomName", '"signal: %d, item: %dx64"'
                               % (int(c["v"] or 0), ntot)))
                ents.append(t_str("Id", "minecraft:barrel")
                            + t_iarr("Pos", [x, y, z])
                            + t_comp("Data", dat) + b"\x00")
            else:
                cc = c["c"] or ""
                if cc.startswith("col"):
                    st = f"minecraft:{COLORS[int(cc[3:]) % 8]}_concrete"
                elif cc == "white":
                    st = "minecraft:white_concrete"
                elif cc == "wire":
                    st = "minecraft:pink_concrete"        # 环内导线（原品红，与位7 撞色故改）
                else:
                    st = "minecraft:light_gray_concrete"
                cells[(x, y, z)] = st

    # ---- 调色板 + Data
    palette = {"minecraft:air": 0}
    for st in cells.values():
        if st not in palette:
            palette[st] = len(palette)
    data = bytearray()
    for y in range(H):
        for z in range(L):
            for x in range(W):
                data += varint(palette[cells.get((x, y, z), "minecraft:air")])

    pal_payload = b"".join(t_int(k, v) for k, v in palette.items())
    we = (t_str("Version", "7.3.8") + t_str("EditingPlatform", "enginehub:fabric")
          + t_iarr("Origin", [0, 0, 0])
          + t_comp("Platforms", t_comp("enginehub:fabric",
                                        t_str("Name", "Fabric-Official")
                                        + t_str("Version", "7.3.8"))))
    payload = (t_int("Version", 3) + t_int("DataVersion", 3953)
               + t_short("Width", W) + t_short("Height", H) + t_short("Length", L)
               + t_iarr("Offset", [0, 0, 0])
               + t_comp("Metadata", t_long("Date", 0) + t_comp("WorldEdit", we))
               + t_comp("Blocks", t_comp("Palette", pal_payload)
                        + t_barr("Data", data)
                        + t_list("BlockEntities", 10, ents))
               + t_int("PaletteMax", len(palette)))
    blob = gzip.compress(b"\x0a" + b"\x00\x00" + t_comp("Schematic", payload) + b"\x00", 9)

    out = os.path.join(ROOT, "results", "schem", "layer1_stack64.schem")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(blob)

    ncmp = sum(1 for v in cells.values() if "comparator" in v)
    nbar = sum(1 for v in cells.values() if "barrel" in v)
    # 支撑方块只在偶数层（y = 2*ch）。白色混凝土同时也是白环/白总线，所以按 y 数，不按材质数。
    nsup = sum(1 for (x, y, z), v in cells.items() if y % 2 == 0 and v == SUPPORT)
    print(f"写出 {out}  {os.path.getsize(out)/1024:.1f} KB")
    print(f"  尺寸 {W}x{H}x{L}   通道数 {NC}   调色板 {len(palette)}")
    print(f"  比较器 {ncmp}   容器 {nbar}   支撑方块 {nsup}（y 为偶数层）")
    print(f"  BlockEntities {len(ents)}")
    if unreach:
        print(f"  不可达信号 {len(unreach)} 个，最大 {max(u[3] for u in unreach)}")


if __name__ == "__main__":
    main()
