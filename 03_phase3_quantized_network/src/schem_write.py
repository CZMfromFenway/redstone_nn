#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
把模拟器里的 Board 写成 Sponge v3 的 .schem 文件。

板子的内部约定            →  Minecraft 块状态
    ("cmp", out_dir)      →  minecraft:comparator[facing=FLIP[out_dir],mode=...,powered=false]
    barrel                →  minecraft:barrel[facing=up,open=false]
    base                  →  minecraft:light_gray_concrete
    white                 →  minecraft:white_concrete
    "col"（输入/环内导线） →  对应颜色的混凝土
"""
import gzip
import json
import struct

FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}
FULL = {"N": "north", "S": "south", "E": "east", "W": "west"}
BITCOL = ["red", "orange", "yellow", "lime", "cyan", "blue", "purple", "magenta"]


def _tag_string(name, val):
    """TAG_String"""
    nb = val.encode("utf-8")
    nm = name.encode("utf-8")
    return b"\x08" + struct.pack(">H", len(nm)) + nm + struct.pack(">H", len(nb)) + nb


def _tag_int(name, val):
    nm = name.encode("utf-8")
    return b"\x03" + struct.pack(">H", len(nm)) + nm + struct.pack(">i", val)


def _tag_short(name, val):
    nm = name.encode("utf-8")
    return b"\x02" + struct.pack(">H", len(nm)) + nm + struct.pack(">h", val)


def _tag_byte_array(name, data):
    nm = name.encode("utf-8")
    return (b"\x07" + struct.pack(">H", len(nm)) + nm
            + struct.pack(">i", len(data)) + bytes(data))


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return out


def write_schem(board, path, warm=None):
    """board 是一个 cmp_sim.Board。warm 是 {pos: 值}，仅用于注释。"""
    W, L = board.W, board.L

    # ---- 1. 收集每个格子的块状态
    cells = {}
    for z in range(L):
        for x in range(W):
            cells[(x, z)] = "minecraft:air"
    for p in board.solid:
        cells[p] = ("minecraft:white_concrete" if board.solid[p] == "white"
                    else "minecraft:light_gray_concrete")
    for p in board.barrel:
        cells[p] = "minecraft:barrel[facing=up,open=false]"
    for p in board.torch:
        # 彩色输入块：颜色由调用方记在 board.torch_color 里
        k = getattr(board, "torch_color", {}).get(p, 7)
        cells[p] = f"minecraft:{BITCOL[k % 8]}_concrete"
    for p, c in board.cmp.items():
        mode = "subtract" if c["mode"] == "subtract" else "compare"
        cells[p] = (f"minecraft:comparator[facing={FULL[FLIP[c['out']]]},"
                    f"mode={mode},powered=false]")

    # ---- 2. 调色板
    pal, data = {}, []
    for z in range(L):
        for x in range(W):
            s = cells[(x, z)]
            if s not in pal:
                pal[s] = len(pal)
            data.append(pal[s])
    # 空气必须存在
    if "minecraft:air" not in pal:
        pal["minecraft:air"] = len(pal)

    # ---- 3. NBT
    body = bytearray()
    body += _tag_string("Name", "")          # 根的占位名字
    # 用手写方式拼 Schematic compound
    comp = bytearray()
    comp += _tag_int("Version", 3)
    comp += _tag_int("DataVersion", 3953)
    comp += _tag_short("Width", W)
    comp += _tag_short("Height", 1)
    comp += _tag_short("Length", L)

    # Palette compound
    pname = "Palette".encode("utf-8")
    pbody = bytearray()
    for k, v in pal.items():
        pbody += _tag_int(k, v)
    pbody += b"\x00"
    blocks = b"\x0a" + struct.pack(">H", 5) + b"Blocks" + b"\x0a" \
        + struct.pack(">H", len(pname)) + pname + bytes(pbody)
    # Data（varint 字节数组）
    vb = bytearray()
    for v in data:
        vb += _varint(v)
    blocks += _tag_byte_array("Data", vb)
    blocks += b"\x09" + struct.pack(">H", len("BlockEntities".encode())) \
        + b"BlockEntities" + b"\x0a" + struct.pack(">i", 0)
    comp += blocks
    comp += b"\x00"

    root = b"\x0a" + struct.pack(">H", 0) + b"" \
        + b"\x0a" + struct.pack(">H", len("Schematic")) + b"Schematic" + bytes(comp)
    root += b"\x00"

    with gzip.open(path, "wb") as f:
        f.write(root)
    return len(pal), sum(1 for v in data if v != pal["minecraft:air"])
