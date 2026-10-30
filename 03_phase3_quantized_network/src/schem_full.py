# -*- coding: utf-8 -*-
"""层1 完整网络：64 片平面 + 32 根火把塔 → 一份 .schem。

层号约定（y_world 是"平面第 ch 层"的世界坐标，导出时 +2）：
    y_schem = 0, 1   ← 塔底（y_world = -2, -1）
    y_schem = 2ch+2  ← 支撑层 / 塔的偶层（A、tap、relay）
    y_schem = 2ch+3  ← 平面层 / 塔的奇层（比较器、容器、彩色块、carrier、B）
    总高 H = 2*64 + 2 = 130

用法：python src/schem_full.py
输出：results/schem/layer1_full.schem
"""
import gzip
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schem_stack64 import (COLORS, FLIP, FULL, SUPPORT, items_for,  # noqa: E402
                           t_barr, t_byte, t_comp, t_iarr, t_int,
                           t_list, t_long, t_short, t_str, varint, _n)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(HERE, "..")
WIRE = "minecraft:pink_concrete"
YOFF = 2                       # y_schem = y_world + YOFF
MARK_EVERY = 8                 # 每 8 个通道，支撑块换成浅灰混凝土，方便定位
MARK_BLOCK = "minecraft:light_gray_concrete"


def support_of(ch):
    return MARK_BLOCK if (ch % MARK_EVERY == 0) else SUPPORT


def wall_torch(facing, lit):
    return "minecraft:redstone_wall_torch[facing=%s,lit=%s]" % (
        {"N": "north", "S": "south", "E": "east", "W": "west"}[facing],
        "true" if lit else "false")


def stand_torch(lit):
    return "minecraft:redstone_torch[lit=%s]" % ("true" if lit else "false")


def main():
    planes_path = (sys.argv[1] if len(sys.argv) > 1
                   else os.path.join(ROOT, "results", "planes64.json"))
    towers_path = (sys.argv[2] if len(sys.argv) > 2
                   else os.path.join(ROOT, "results", "towers32.json"))
    out = (sys.argv[3] if len(sys.argv) > 3
           else os.path.join(ROOT, "results", "schem", "layer1_full.schem"))
    d = json.load(open(planes_path, encoding="utf-8"))
    tw = json.load(open(towers_path, encoding="utf-8"))
    planes, W, L, b = d["planes"], d["W"], d["L"], d["b"]
    NCH = len(planes)
    H = 2 * NCH + YOFF

    cells = {}          # (x, y, z) -> blockstate
    ents = []           # BlockEntity 载荷
    unreach = []
    clash = []

    def put(x, y, z, st, what):
        if (x, y, z) in cells:
            clash.append((x, y, z, cells[(x, y, z)], st, what))
            return
        cells[(x, y, z)] = st

    # ---------- 平面 ----------
    for p in planes:
        ch = p["ch"]
        y = 2 * ch + 1 + YOFF
        ysup = y - 1
        for c in p["cells"]:
            x, z = c["x"], c["z"]
            if c["t"] == "cmp":
                mode = "subtract" if c["mode"] == "subtract" else "compare"
                pw = "true" if (c["st"] or 0) > 0 else "false"
                put(x, y, z, "minecraft:comparator[facing=%s,mode=%s,powered=%s]"
                    % (FULL[FLIP[c["out"]]], mode, pw), "cmp")
                put(x, ysup, z, support_of(ch), "support")
                dat = t_str("id", "minecraft:comparator") + t_int("OutputSignal", int(c["st"] or 0))
                ents.append(t_str("Id", "minecraft:comparator")
                            + t_iarr("Pos", [x, y, z])
                            + t_comp("Data", dat) + b"\x00")
            elif c["t"] == "barrel":
                put(x, y, z, "minecraft:barrel[facing=up,open=false]", "barrel")
                it, ntot = items_for(int(c["v"] or 0))
                if it is None:
                    unreach.append((ch, x, z, c["v"]))
                    it = []
                payload = b""
                for (sl, cnt) in it:
                    payload += (t_int("count", cnt) + t_byte("Slot", sl)
                                + t_str("id", "minecraft:wooden_axe")) + b"\x00"
                dat = (t_str("id", "minecraft:barrel")
                       + b"\x09" + _n("Items") + b"\x0a" + struct.pack(">i", len(it)) + payload
                       + t_str("CustomName", '"signal: %d, item: %dx64"' % (int(c["v"] or 0), ntot)))
                ents.append(t_str("Id", "minecraft:barrel")
                            + t_iarr("Pos", [x, y, z])
                            + t_comp("Data", dat) + b"\x00")
            else:
                cc = c["c"] or ""
                if cc.startswith("col"):
                    st = "minecraft:%s_concrete" % COLORS[int(cc[3:]) % 8]
                elif cc == "white":
                    st = "minecraft:white_concrete"
                elif cc == "wire":
                    st = WIRE
                else:
                    st = "minecraft:light_gray_concrete"
                put(x, y, z, st, "plane-block")

    # ---------- 塔 ----------
    for T in tw["towers"]:
        colr = "minecraft:%s_concrete" % COLORS[T["k"] % 8]
        tx, bx, tz, bz = (T["cells"]["tap"][0], T["cells"]["A"][0],
                          T["cells"]["tap"][1], T["cells"]["B"][1])
        rz = T["cells"]["B"][1]
        for ch in range(NCH):
            y = 2 * ch + YOFF
            yo = y + 1
            put(tx, y, tz, wall_torch(T["tap_face"], T["tap"][ch]), "tap")
            put(bx, y, tz, colr, "A")
            put(bx, y, rz, wall_torch(T["relay_face"], T["tap"][ch]), "relay")
            put(bx, yo, bz, colr, "B")
            if T["flip"][ch]:
                put(bx, yo, tz, stand_torch(T["carrier"][ch]), "carrier-stand")
            else:
                put(bx, yo, tz, wall_torch(T["carrier_face"], T["carrier"][ch]), "carrier-wall")
        # 塔底（y_schem 0, 1）
        if T["bottom_inv"]:
            put(bx, 0, tz, colr, "pad-block")
            put(bx, 1, tz, stand_torch(True), "pad-stand")
        else:
            put(bx, 0, tz, colr, "pad-block")
            put(bx, 1, rz, colr, "pad-block")
            put(bx, 0, rz, wall_torch(T["relay_face"], True), "pad-torch")
            put(bx, 1, tz, wall_torch(T["carrier_face"], False), "pad-torch")

    if clash:
        print(f"✗ 位置冲突 {len(clash)} 处：")
        for c in clash[:6]:
            print("   (%d,%d,%d) %s vs %s [%s]" % (c[0], c[1], c[2], c[3], c[4], c[5]))
        return 1

    # ---------- 写 schem ----------
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
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(blob)

    ncmp = sum(1 for v in cells.values() if "comparator" in v)
    nbar = sum(1 for v in cells.values() if "barrel" in v)
    nstand = sum(1 for v in cells.values() if "redstone_torch" in v)
    nwall = sum(1 for v in cells.values() if "wall_torch" in v)
    ncol = sum(1 for v in cells.values() if "_concrete" in v and any(
        v == "minecraft:%s_concrete" % cn for cn in COLORS))
    print(f"写出 {out}  {os.path.getsize(out)/1024:.1f} KB")
    print(f"  尺寸 {W} x {H} x {L}   通道 {NCH}   塔 {len(tw['towers'])}   调色板 {len(palette)}")
    print(f"  方块总数 {len(cells)}：比较器 {ncmp}  容器 {nbar}  直立火把 {nstand}  "
          f"壁面火把 {nwall}  位色彩块 {ncol}")
    print(f"  BlockEntities {len(ents)}")
    if unreach:
        print(f"  ✗ 不可达容器 {len(unreach)} 个，最大 {max(u[3] for u in unreach)}")
        return 1
    print("  不可达容器 0 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
