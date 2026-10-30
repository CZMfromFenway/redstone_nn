# -*- coding: utf-8 -*-
"""生成一根火把塔的独立 .schem（2×2 × 全高），用于游戏内验证。

截面（局部坐标，x = 平面列方向，z = 行方向）
    x=0  槽列（该位自己的列：上槽 CE=2i，下槽 CO=2i+1）
    x=1  塔列
    z=0  槽行（输入块所在行）
    z=1  读入行（读入比较器所在行）

每个通道 ch（平面在 y = 2ch+1）塔加 5 个方块：
    (0, 2ch,   0) tap 火把（壁面，朝塔列）
    (1, 2ch,   0) A(ch) 彩色块
    (1, 2ch,   1) relay 火把（壁面，朝 A）
    (1, 2ch+1, 0) carrier 火把（壁面朝 B / 直立搭 A）
    (1, 2ch+1, 1) B(ch) 彩色块

塔底（y = -2, -1）：
    s(0)=1（不需要反相）→ 之字形双火把：块(-2,z0) + 壁面火把(-2,z1) + 块(-1,z1) + 壁面火把(-1,z0)
    s(0)=0（需要反相）  → 直立单火把：块(-2,z0) + 直立火把(-1,z0)

导出的 schem 里 y_schem = y_world + 2，所以 H = 128 + 2 = 130。

用法：python src/tower_gen.py [i] [k] [输出路径]
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schem_stack64 import (t_barr, t_byte, t_comp, t_iarr, t_int,  # noqa: E402
                           t_list, t_long, t_short, t_str, varint)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(HERE, "..")
COLORS = ["red", "orange", "yellow", "lime", "cyan", "blue", "purple", "magenta"]
FACING = {("W"): "west", ("E"): "east", ("N"): "north", ("S"): "south"}


def slots(b):
    J = (b + 1) // 2
    out = []
    for j in range(J):
        out.append({"c": j, "Z": -6 * j + 2, "up": False})
        out.append({"c": j, "Z": -6 * j - 2, "up": True})
    out.sort(key=lambda s: (abs(s["Z"]), -s["Z"]))
    SLOT = [None] * b
    for i in range(b):
        SLOT[b - 1 - i] = out[i]
    return SLOT


def wall_torch(facing, lit):
    return "minecraft:redstone_wall_torch[facing=%s,lit=%s]" % (FACING[facing],
                                                               "true" if lit else "false")


def stand_torch(lit):
    return "minecraft:redstone_torch[lit=%s]" % ("true" if lit else "false")


def plan_tower(i, k, b=8, n=4):
    """返回该塔的几何与每通道状态。"""
    ref = json.load(open(os.path.join(ROOT, "results", "py_compile_reference.json"),
                         encoding="utf-8"))
    Wm = ref["layers"]["L1"]["8"]["W"]
    SLOT = slots(b)
    s = SLOT[k]
    chain = -6 * s["c"]
    if s["up"]:
        blockRow, readRow = chain - 2, chain - 1
        slotCol, otherCol = 2 * i, 2 * i + 1
    else:
        blockRow, readRow = chain + 2, chain + 1
        slotCol, otherCol = 2 * i + 1, 2 * i

    wcol = [Wm[ch][i][k] for ch in range(64)]
    # 真实符号：取该输入该通道绝对值最大的那个位的符号。
    # 零权重格子（实测 26 处）读 15 恒定、对算术不敏感，用真实符号可让
    # 同一输入 8 个位的翻相高度完全一致（与 tower_spec.py 一致）。
    sign = []
    for ch in range(64):
        row = Wm[ch][i]
        best = max(range(b), key=lambda kk: abs(row[kk]))
        sign.append(1 if row[best] > 0 else 0)
    flip = [False] * 64
    for ch in range(63):
        flip[ch] = (sign[ch + 1] != sign[ch])
    XBIT = 0                                   # 独立塔没有总线，等价于 X=0 的空闲态
    tap = [1 if XBIT != sign[ch] else 0 for ch in range(64)]
    car = [(tap[ch] if flip[ch] else 1 - tap[ch]) for ch in range(64)]

    # 面朝方向（与绝对坐标一致）
    tap_f = "W" if otherCol > slotCol else "E"
    car_f = "N" if readRow > blockRow else "S"     # carrier 壁面朝 B（读入行那侧）
    rel_f = "S" if readRow > blockRow else "N"     # relay 朝 A（槽行那侧）
    return dict(i=i, k=k, up=s["up"], chain=chain, blockRow=blockRow, readRow=readRow,
                slotCol=slotCol, otherCol=otherCol, wcol=wcol, sign=sign, flip=flip,
                tap=tap, car=car, tap_f=tap_f, car_f=car_f, rel_f=rel_f,
                bottom_inv=(sign[0] == 0), n=n, b=b)


def build_cells(T):
    """局部坐标 (x, y_world, z) -> blockstate。
    x/z 用"最小角对齐"：lx = rx - min(slotCol, otherCol)，lz = Z - min(blockRow, readRow)。
    所以局部轴与世界轴同向（纯平移、不镜像），facing 直接可用，贴进平面时对齐最小角即可。"""
    x0 = min(T["slotCol"], T["otherCol"])
    z0 = min(T["blockRow"], T["readRow"])
    lx_slot = T["slotCol"] - x0
    lx_tow = T["otherCol"] - x0
    lz_blk = T["blockRow"] - z0
    lz_rd = T["readRow"] - z0
    cells = {}
    colr = "minecraft:%s_concrete" % COLORS[T["k"] % 8]

    for ch in range(64):
        y = 2 * ch
        yo = y + 1
        cells[(lx_slot, y, lz_blk)] = wall_torch(T["tap_f"], T["tap"][ch])    # tap
        cells[(lx_tow, y, lz_blk)] = colr                                     # A(ch)
        cells[(lx_tow, y, lz_rd)] = wall_torch(T["rel_f"], T["tap"][ch])      # relay
        if T["flip"][ch]:
            cells[(lx_tow, yo, lz_blk)] = stand_torch(T["car"][ch])           # 直立 carrier
        else:
            cells[(lx_tow, yo, lz_blk)] = wall_torch(T["car_f"], T["car"][ch])  # 壁面 carrier
        cells[(lx_tow, yo, lz_rd)] = colr                                     # B(ch)

    # 塔底
    if T["bottom_inv"]:
        # 直立单火把：块(-2,槽行) + 直立火把(-1,槽行)
        # 空闲态（总线未接）：块未充能 ⇒ 直立火把亮 ⇒ A(0) 被充能 ⇒ tap(0) 灭，与 X=0 自洽
        cells[(lx_tow, -2, lz_blk)] = colr
        cells[(lx_tow, -1, lz_blk)] = stand_torch(True)
    else:
        # 之字形双火把：块(-2,槽行) + 壁面火把(-2,读入行) + 块(-1,读入行) + 壁面火把(-1,槽行)
        cells[(lx_tow, -2, lz_blk)] = colr
        cells[(lx_tow, -2, lz_rd)] = wall_torch(T["rel_f"], True)
        cells[(lx_tow, -1, lz_rd)] = colr
        cells[(lx_tow, -1, lz_blk)] = wall_torch(T["car_f"], False)
    return cells


def write_schem(cells, out, ymin=-2, ymax=127):
    W = max(c[0] for c in cells) + 1
    L = max(c[2] for c in cells) + 1
    H = ymax - ymin + 1

    grid = {}
    for (x, y, z), st in cells.items():
        grid[(x, y - ymin, z)] = st

    palette = {"minecraft:air": 0}
    for st in grid.values():
        if st not in palette:
            palette[st] = len(palette)
    data = bytearray()
    for y in range(H):
        for z in range(L):
            for x in range(W):
                data += varint(palette[grid.get((x, y, z), "minecraft:air")])

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
                        + t_list("BlockEntities", 10, []))
               + t_int("PaletteMax", len(palette)))
    blob = gzip.compress(b"\x0a" + b"\x00\x00" + t_comp("Schematic", payload) + b"\x00", 9)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(blob)
    return W, H, L, len(palette), len(grid)


def main():
    i_in = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    k_in = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = (sys.argv[3] if len(sys.argv) > 3
           else os.path.join(ROOT, "results", "schem", "tower_i%d_k%d.schem" % (i_in, k_in)))

    T = plan_tower(i_in, k_in)
    cells = build_cells(T)
    W, H, L, npal, ncell = write_schem(cells, out)

    print(f"塔 (i={i_in}, k={k_in})  {'上槽' if T['up'] else '下槽'}  位色={COLORS[k_in % 8]}")
    print(f"  槽列 rx={T['slotCol']}  塔列 rx={T['otherCol']}  "
          f"槽行 Z={T['blockRow']}  读入行 Z={T['readRow']}")
    print(f"  朝向: tap={T['tap_f']}  carrier={T['car_f']}  relay={T['rel_f']}")
    print(f"  塔底: {'直立单火把（需反相）' if T['bottom_inv'] else '之字形双火把'}")
    print(f"  翻相（直立）{sum(T['flip'])} 处: {[c for c in range(64) if T['flip'][c]]}")
    print(f"  写出 {out}  {os.path.getsize(out)/1024:.1f} KB")
    print(f"  尺寸 {W} x {H} x {L}   方块 {ncell}   调色板 {npal}（y_schem = y_world + 2）")


if __name__ == "__main__":
    main()
