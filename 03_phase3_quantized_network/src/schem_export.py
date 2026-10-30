# -*- coding: utf-8 -*-
"""把 board_dump.json 写成 Sponge v3 的 .schem（每个测试用例一个文件）。

方块映射
    cmp(out=方向)  →  minecraft:comparator[facing=FLIP[out],mode=…,powered=false]
    barrel         →  minecraft:barrel[facing=up,open=false]
    blk base       →  minecraft:light_gray_concrete
    blk white      →  minecraft:white_concrete
    blk wire       →  minecraft:magenta_concrete       （环内导线：被比较器充能的实心块）
    blk colK       →  minecraft:<颜色>_concrete         （输入块；火把按用例放置）
    torch          →  minecraft:redstone_torch[lit=true] 放在输入块的侧向空格
"""
import gzip
import json
import os
import re
import struct
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"E:\redstone_nn\03_phase3_quantized_network"
OUTDIR = os.path.join(ROOT, "results", "schem")
COLORS = ["red", "orange", "yellow", "lime", "cyan", "blue", "purple", "magenta"]
FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}
FULL = {"N": "north", "S": "south", "E": "east", "W": "west"}
DV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


# ---------------- NBT ----------------
def _name(n):
    b = n.encode("utf-8")
    return struct.pack(">H", len(b)) + b


def tag_byte(n, v):
    return b"\x01" + _name(n) + struct.pack(">b", v)


def tag_short(n, v):
    return b"\x02" + _name(n) + struct.pack(">h", v)


def tag_int(n, v):
    return b"\x03" + _name(n) + struct.pack(">i", v)


def tag_long(n, v):
    return b"\x04" + _name(n) + struct.pack(">q", v)


def tag_string(n, v):
    b = v.encode("utf-8")
    return b"\x08" + _name(n) + struct.pack(">H", len(b)) + b


def tag_bytearray(n, data):
    return b"\x07" + _name(n) + struct.pack(">i", len(data)) + bytes(data)


def tag_intarray(n, vals):
    return b"\x0b" + _name(n) + struct.pack(">i", len(vals)) + b"".join(
        struct.pack(">i", v) for v in vals)


def tag_list(n, elem_type, payloads):
    return (b"\x09" + _name(n) + bytes([elem_type])
            + struct.pack(">i", len(payloads)) + b"".join(payloads))


def tag_compound(n, payload):
    return b"\x0a" + _name(n) + payload + b"\x00"


SLOTS, MAXC = 27, 64          # 木斧 maxStack = 1，所以 Σcount 就是 fill
DENOM = SLOTS                 # signal = 1 + floor(14*fill/27)


def sig_of(fill):
    return 0 if fill == 0 else 1 + (14 * fill) // DENOM


def items_for(sig):
    """信号 → [(slot, count, id)]，木斧每格最多 64 把。sig > 897 返回 None。"""
    if sig == 0:
        return [], 0
    n = -(-((sig - 1) * DENOM) // 14)          # ceil((sig-1)*27/14)
    if n > SLOTS * MAXC:
        return None, n
    out, need, slot = [], n, 0
    while need > 0:
        c = min(MAXC, need)
        out.append((slot, c, "minecraft:wooden_axe"))
        need -= c
        slot += 1
    return out, n


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


# ---------------- 板 → 方块 ----------------
def build_cells(case):
    """返回 {(x,z): blockstate} 和 [(x,z,torch_dir)]"""
    W, L = case["W"], case["L"]
    cells = {}
    cmp_out = {}
    cmp_st = {}
    blocks = {}
    for c in case["cells"]:
        p = (c["x"], c["z"])
        if c["t"] == "cmp":
            cmp_out[p] = c["out"]
            cmp_st[p] = c["st"] or 0
        elif c["t"] == "barrel":
            cells[p] = "minecraft:barrel[facing=up,open=false]"
        else:
            blocks[p] = c["c"] or "base"

    for p, out in cmp_out.items():
        mode = "subtract" if any(
            c["t"] == "cmp" and (c["x"], c["z"]) == p and c["mode"] == "subtract"
            for c in case["cells"]) else "compare"
        pw = "true" if cmp_st.get(p, 0) > 0 else "false"
        cells[p] = (f"minecraft:comparator[facing={FULL[FLIP[out]]},"
                    f"mode={mode},powered={pw}]")

    for p, cc in blocks.items():
        if cc == "base":
            cells[p] = "minecraft:light_gray_concrete"
        elif cc == "white":
            cells[p] = "minecraft:white_concrete"
        elif cc == "wire":
            cells[p] = "minecraft:magenta_concrete"
        elif cc.startswith("col"):
            k = int(cc[3:]) % 8
            cells[p] = f"minecraft:{COLORS[k]}_concrete"
        else:
            cells[p] = "minecraft:light_gray_concrete"

    # 火把：放在输入块侧向空格
    torches = []
    for c in case["cells"]:
        if c["ik"] and c["t"] == "blk" and (c["c"] or "").startswith("col") and c["torch"] > 0:
            p = (c["x"], c["z"])
            for d, (dx, dz) in DV.items():
                q = (p[0] + dx, p[1] + dz)
                if q not in cells and q not in blocks and 0 <= q[0] < W and 0 <= q[1] < L:
                    torches.append((q[0], q[1]))
                    break
    return cells, torches


def block_entities(cells, case):
    """写 BlockEntities：比较器带 OutputSignal，容器带 Items。y=0 层。"""
    out = []
    unreach = []
    cmap = {(c["x"], c["z"]): c for c in case["cells"]}
    for (x, z), st in cells.items():
        c = cmap.get((x, z))
        if "comparator" in st:
            sig = c["st"] if (c and c["st"] is not None) else 0
            dat = (tag_string("id", "minecraft:comparator")
                   + tag_int("OutputSignal", int(sig)))
            ent = (tag_string("Id", "minecraft:comparator")
                   + tag_intarray("Pos", [x, 0, z])
                   + tag_compound("Data", dat))
            out.append(ent + b"\x00")
        elif "barrel" in st:
            sig = c["v"] if (c and c["v"] is not None) else 0
            it, ntot = items_for(sig)
            if it is None:
                unreach.append((x, z, sig))
                it = []; ntot = 0
            payload_items = b""
            for (sl, cnt, iid) in it:
                one = (tag_int("count", cnt)
                       + tag_byte("Slot", sl)
                       + tag_string("id", iid))
                payload_items += one + b"\x00"
            dat = (tag_string("id", "minecraft:barrel")
                   + b"\x09" + _name("Items") + b"\x0a"
                   + struct.pack(">i", len(it)) + payload_items
                   + tag_string("CustomName",
                                '"signal: %d, item: %dx64"' % (sig, ntot)))
            ent = (tag_string("Id", "minecraft:barrel")
                   + tag_intarray("Pos", [x, 0, z])
                   + tag_compound("Data", dat))
            out.append(ent + b"\x00")
    return out, unreach


def write_schem(cells, torches, W, L, path, name, ents=None):
    palette = {}
    for st in cells.values():
        palette.setdefault(st, len(palette))
    for _ in torches:
        palette.setdefault("minecraft:redstone_torch[lit=true]", len(palette))
    # air 必须在调色板里
    palette.setdefault("minecraft:air", len(palette))

    H = 2
    data = bytearray()
    for y in range(H):
        for z in range(L):
            for x in range(W):
                if y == 0:
                    st = cells.get((x, z), "minecraft:air")
                else:
                    st = "minecraft:air"
                data += varint(palette[st])
    # 火把放在 y=1
    torch_set = set(torches)
    if torches:
        data = bytearray()
        for y in range(H):
            for z in range(L):
                for x in range(W):
                    if y == 1 and (x, z) in torch_set:
                        st = "minecraft:redstone_torch[lit=true]"
                    elif y == 0:
                        st = cells.get((x, z), "minecraft:air")
                    else:
                        st = "minecraft:air"
                    data += varint(palette[st])

    pal_payload = b""
    for st, idx in palette.items():
        pal_payload += tag_int(st, idx)

    # WorldEdit 的 Sponge v3：Palette / Data / BlockEntities 放在 Blocks 子复合里
    blocks = b""
    blocks += tag_compound("Palette", pal_payload)
    blocks += tag_bytearray("Data", data)
    blocks += tag_list("BlockEntities", 10, ents or [])

    payload = b""
    payload += tag_int("Version", 3)
    payload += tag_int("DataVersion", 3953)
    payload += tag_short("Width", W)
    payload += tag_short("Height", H)
    payload += tag_short("Length", L)
    payload += tag_intarray("Offset", [0, 0, 0])
    we = (tag_string("Version", "7.3.8")
          + tag_string("EditingPlatform", "enginehub:fabric")
          + tag_intarray("Origin", [0, 0, 0])
          + tag_compound("Platforms",
                         tag_compound("enginehub:fabric",
                                      tag_string("Name", "Fabric-Official")
                                      + tag_string("Version", "7.3.8"))))
    payload += tag_compound("Metadata",
                            tag_long("Date", 0) + tag_compound("WorldEdit", we))
    payload += tag_compound("Blocks", blocks)
    payload += tag_int("PaletteMax", len(palette))

    # 根(空名) { Schematic { … } }
    inner = tag_compound("Schematic", payload)
    blob = gzip.compress(b"\x0a" + b"\x00\x00" + inner + b"\x00", 9)
    with open(path, "wb") as f:
        f.write(blob)
    return os.path.getsize(path), len(palette)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = json.load(open(os.path.join(ROOT, "results", "board_dump.json"), encoding="utf-8"))
    for case in d["cases"]:
        cells, torches = build_cells(case)
        safe = {"C0_全0": "C0_all0", "C1_全1": "C1_all1",
                "C2_1010": "C2_1010", "C3_0101": "C3_0101"}.get(case["name"])
        if safe is None:
            safe = re.sub(r"[^A-Za-z0-9_]", "", case["name"]) or "case"
        p = os.path.join(OUTDIR, f"layer1_ch0_{safe}.schem")
        ents, unreach = block_entities(cells, case)
        size, npal = write_schem(cells, torches, case["W"], case["L"], p,
                                 case["name"], ents)
        print(f"  {case['name']:10s} -> {os.path.basename(p)}  "
              f"{case['W']}×{H_}×{case['L']}  {size/1024:.1f} KB  调色板 {npal}  "
              f"比较器 {sum(1 for v in cells.values() if 'comparator' in v)}  "
              f"容器 {sum(1 for v in cells.values() if 'barrel' in v)}  "
              f"火把 {len(torches)}"
              + (f"  不可达信号 {len(unreach)} 个（最大 {max(u[2] for u in unreach)}）" if unreach else ""))


H_ = 2

if __name__ == "__main__":
    main()
