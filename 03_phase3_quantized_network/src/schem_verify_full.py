# -*- coding: utf-8 -*-
"""回读 layer1_full.schem，与 planes64.json + towers32.json 逐格对账。

检查：
  1 NBT 完整性、len(Data) == W×H×L
  2 尺寸 14 × 130 × 31（平面在 y = 2ch+3，支撑/塔偶层在 2ch+2，塔底在 0/1）
  3 平面每一个 cell 的方块状态
  4 塔每一个方块（tap/A/relay/carrier/B + 塔底）
  5 容器信号反解 ≤ 897、不可达 0
  6 powered 一致性
  7 火把计数

用法：python src/schem_verify_full.py [schem路径]
"""
import gzip
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schem_verify_stack import (COLORS, FLIP, FULL, SUPPORT, R,  # noqa: E402
                                sig_of_items, varints)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(HERE, "..")
YOFF = 2
WIRE = "minecraft:pink_concrete"
MARK_EVERY = 8
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
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(ROOT, "results", "schem", "layer1_full.schem"))
    planes_path = (sys.argv[2] if len(sys.argv) > 2
                   else os.path.join(ROOT, "results", "planes64.json"))
    towers_path = (sys.argv[3] if len(sys.argv) > 3
                   else os.path.join(ROOT, "results", "towers32.json"))
    d = json.load(open(planes_path, encoding="utf-8"))
    tw = json.load(open(towers_path, encoding="utf-8"))
    planes, pW, pL = d["planes"], d["W"], d["L"]
    NCH = len(planes)

    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw); r.u1(); r.name()
    sch = r.payload(10)["Schematic"]
    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    blk = sch["Blocks"]
    pal = blk["Palette"]
    inv = {v: k for k, v in pal.items()}
    idx = varints(blk["Data"])
    be = blk.get("BlockEntities", [])

    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    def at(x, y, z):
        return inv[idx[(y * L + z) * W + x]]

    chk(r.i == len(raw), "NBT 未读到末尾 %d/%d" % (r.i, len(raw)))
    chk(len(idx) == W * H * L, "Data 长度 %d ≠ %d" % (len(idx), W * H * L))
    print("1 完整性 %d/%d 字节，Data %d = %dx%dx%d  %s"
          % (r.i, len(raw), len(idx), W, H, L, "✓" if not fails else "✗"))

    ok2 = (W == pW and L == pL and H == 2 * NCH + YOFF)
    chk(ok2, "尺寸 %dx%dx%d，期望 %dx%dx%d" % (W, H, L, pW, 2 * NCH + YOFF, pL))
    print("2 尺寸 %d x %d x %d（平面链 y=2ch+3）  %s" % (W, H, L, "✓" if ok2 else "✗"))

    ent = {(e["Id"], tuple(e["Pos"])): e.get("Data", {}) for e in be}

    # ---- 平面 ----
    mism = 0
    sig_max, unreach = 0, 0
    for p in planes:
        ch = p["ch"]
        y = 2 * ch + 1 + YOFF
        for c in p["cells"]:
            x, z = c["x"], c["z"]
            if c["t"] == "cmp":
                mode = "subtract" if c["mode"] == "subtract" else "compare"
                pw = "true" if (c["st"] or 0) > 0 else "false"
                want = "minecraft:comparator[facing=%s,mode=%s,powered=%s]" % (
                    FULL[FLIP[c["out"]]], mode, pw)
                dat = ent.get(("minecraft:comparator", (x, y, z)))
                if at(x, y, z) != want:
                    mism += 1
                elif dat is None or int(dat.get("OutputSignal", -1)) != int(c["st"] or 0):
                    mism += 1
                if at(x, y - 1, z) != support_of(ch):
                    mism += 1
            elif c["t"] == "barrel":
                dat = ent.get(("minecraft:barrel", (x, y, z)))
                if at(x, y, z) != "minecraft:barrel[facing=up,open=false]":
                    mism += 1
                elif dat is None:
                    mism += 1
                else:
                    sig, _ = sig_of_items(dat.get("Items", []))
                    sig_max = max(sig_max, sig)
                    if sig > 897:
                        unreach += 1
                    if sig != int(c["v"] or 0):
                        mism += 1
            else:
                cc = c["c"] or ""
                if cc.startswith("col"):
                    want = "minecraft:%s_concrete" % COLORS[int(cc[3:]) % 8]
                elif cc == "white":
                    want = "minecraft:white_concrete"
                elif cc == "wire":
                    want = WIRE
                else:
                    want = "minecraft:light_gray_concrete"
                if at(x, y, z) != want:
                    mism += 1
    chk(mism == 0, "平面有 %d 处不一致" % mism)
    print("3 平面逐格：%d 处不一致  %s" % (mism, "✓" if mism == 0 else "✗"))

    # ---- 塔 ----
    tmism = 0
    nstand = nwall = 0
    for T in tw["towers"]:
        colr = "minecraft:%s_concrete" % COLORS[T["k"] % 8]
        tx, tz = T["cells"]["tap"]
        bx = T["cells"]["A"][0]
        bz = T["cells"]["A"][1]
        rz = T["cells"]["B"][1]
        for ch in range(NCH):
            y = 2 * ch + YOFF
            want = {
                (tx, y, tz): wall_torch(T["tap_face"], T["tap"][ch]),
                (bx, y, tz): colr,
                (bx, y, rz): wall_torch(T["relay_face"], T["tap"][ch]),
                (bx, y + 1, bz): colr,
            }
            if T["flip"][ch]:
                want[(bx, y + 1, tz)] = stand_torch(T["carrier"][ch])
            else:
                want[(bx, y + 1, tz)] = wall_torch(T["carrier_face"], T["carrier"][ch])
            for (x, yy, z), st in want.items():
                if at(x, yy, z) != st:
                    tmism += 1
        if T["bottom_inv"]:
            pad = {(bx, 0, tz): colr, (bx, 1, tz): stand_torch(True)}
        else:
            pad = {(bx, 0, tz): colr, (bx, 0, rz): wall_torch(T["relay_face"], True),
                   (bx, 1, rz): colr, (bx, 1, tz): wall_torch(T["carrier_face"], False)}
        for (x, yy, z), st in pad.items():
            if at(x, yy, z) != st:
                tmism += 1
    chk(tmism == 0, "塔有 %d 处不一致" % tmism)
    print("4 塔逐格：%d 处不一致  %s" % (tmism, "✓" if tmism == 0 else "✗"))

    # ---- 容器 / powered / 火把 ----
    chk(unreach == 0, "不可达容器 %d" % unreach)
    print("5 容器：最大信号 %d ≤ 897，不可达 %d  %s"
          % (sig_max, unreach, "✓" if unreach == 0 else "✗"))

    cnt = Counter()
    for i in idx:
        s = inv[i]
        if "comparator" in s:
            cnt["cmp"] += 1
            if "powered=true" in s:
                cnt["cmp_on"] += 1
        elif "barrel" in s:
            cnt["barrel"] += 1
        elif "redstone_torch" in s:
            cnt["stand"] += 1
        elif "wall_torch" in s:
            cnt["wall"] += 1
    nz = sum(1 for k, v in ent.items() if k[0] == "minecraft:comparator"
             and int(v.get("OutputSignal", 0)) > 0)
    ok6 = cnt["cmp_on"] == nz
    chk(ok6, "powered=true %d ≠ OutputSignal>0 %d" % (cnt["cmp_on"], nz))
    print("6 powered 一致性 true %d / OutputSignal>0 %d  %s"
          % (cnt["cmp_on"], nz, "✓" if ok6 else "✗"))
    print("7 计数：比较器 %d  容器 %d  直立火把 %d  壁面火把 %d  总非空 %d"
          % (cnt["cmp"], cnt["barrel"], cnt["stand"], cnt["wall"],
             sum(cnt[k] for k in ("cmp", "barrel", "stand", "wall"))
             + sum(1 for i in idx if "_concrete" in inv[i])))

    print()
    if fails:
        print("✗ 失败 %d 项：" % len(fails))
        for f in fails[:10]:
            print("   " + f)
        return 1
    print("回读对账全部通过 ✓  （%dx%dx%d，BlockEntities %d）" % (W, H, L, len(be)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
