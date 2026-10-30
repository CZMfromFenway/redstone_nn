# -*- coding: utf-8 -*-
"""回读 layer1_stack64.schem，与 planes64.json 逐格对账。

检查项
    1  NBT 读到文件末尾、len(Data) == W×H×L
    2  尺寸与 planes64 一致（14 × 128 × 31）
    3  每一个 cell 的方块状态与 planes64 一致
    4  比较器 BlockEntity 的 OutputSignal == 仿真器 state
    5  容器 Items 反解出的信号 == 仿真器 val（用 signal = 1 + floor(14·fill/27)）
    6  每个比较器下方（y-1）有支撑方块
    7  容器信号 ≤ 897（不可达容器数必须为 0）
    8  powered=true 的个数 == OutputSignal>0 的个数
    9  火把数（当前阶段应为 0，等火把塔）

用法：python src/schem_verify_stack.py [schem路径]
"""
import gzip
import json
import os
import struct
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
COLORS = ["red", "orange", "yellow", "lime", "cyan", "blue", "purple", "magenta"]
FULL = {"N": "north", "S": "south", "E": "east", "W": "west"}
FLIP = {"N": "S", "S": "N", "E": "W", "W": "E"}
SUPPORT = "minecraft:white_concrete"
SLOTS = 27
SIG_MAX = 897


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


def sig_of_items(items):
    """容器 Items → 信号强度（木斧 maxStack=1，每格 ≤64）：1 + floor(14·fill/27)"""
    fill = 0
    for it in items:
        fill += it.get("count", it.get("Count", 0))
    if fill <= 0:
        return 0, 0
    return 1 + (14 * fill) // SLOTS, fill


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(ROOT, "results", "schem", "layer1_stack64.schem"))
    d = json.load(open(os.path.join(ROOT, "results", "planes64.json"), encoding="utf-8"))
    planes = d["planes"]
    pW, pL = d["W"], d["L"]

    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw)
    assert r.u1() == 10, "根标签不是 TAG_Compound"
    r.name()
    sch = r.payload(10)["Schematic"]
    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    blk = sch["Blocks"]
    pal = blk["Palette"]
    inv = {v: k for k, v in pal.items()}
    idx = varints(blk["Data"])
    be = blk["BlockEntities"]

    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    # ---- 1 完整性 ----
    chk(r.i == len(raw), "NBT 没有读到末尾：%d/%d" % (r.i, len(raw)))
    chk(len(idx) == W * H * L, "Data 长度 %d ≠ W×H×L %d" % (len(idx), W * H * L))
    print("1 完整性：NBT %d/%d 字节，Data %d = %dx%dx%d  %s"
          % (r.i, len(raw), len(idx), W, H, L, "✓" if not fails else "✗"))

    # ---- 2 尺寸 ----
    expH = 2 * len(planes)
    ok2 = (W == pW and L == pL and H == expH)
    chk(ok2, "尺寸不一致 %dx%dx%d，期望 %dx%dx%d" % (W, H, L, pW, expH, pL))
    print("2 尺寸：%d x %d x %d（通道 %d，y=2ch+1）  %s" % (W, H, L, len(planes), "✓" if ok2 else "✗"))

    # 建索引
    def bi(x, y, z):
        return idx[(y * L + z) * W + x]

    ent = {}
    for e in be:
        ent[(e["Id"], tuple(e["Pos"]))] = e.get("Data", {})

    # ---- 3/4/5 逐格对账 ----
    ncmp = nbar = nblk = 0
    mism = 0
    sig_max = 0
    unreach = 0
    tot_cells = 0
    torch = 0
    for p in planes:
        ch, y = p["ch"], 2 * p["ch"] + 1
        for c in p["cells"]:
            tot_cells += 1
            x, z = c["x"], c["z"]
            st = inv[bi(x, y, z)]
            if c["t"] == "cmp":
                ncmp += 1
                mode = "subtract" if c["mode"] == "subtract" else "compare"
                pw = "true" if (c["st"] or 0) > 0 else "false"
                want = ("minecraft:comparator[facing=%s,mode=%s,powered=%s]"
                        % (FULL[FLIP[c["out"]]], mode, pw))
                if st != want:
                    mism += 1
                    if mism <= 3:
                        fails.append("ch%d (%d,%d) 方块状态 %s ≠ %s" % (ch, x, z, st, want))
                dat = ent.get(("minecraft:comparator", (x, y, z)))
                if dat is None:
                    mism += 1
                    fails.append("ch%d (%d,%d) 缺比较器 BlockEntity" % (ch, x, z))
                elif int(dat.get("OutputSignal", -1)) != int(c["st"] or 0):
                    mism += 1
                    fails.append("ch%d (%d,%d) OutputSignal %s ≠ 仿真 %s"
                                 % (ch, x, z, dat.get("OutputSignal"), c["st"]))
                if inv[bi(x, y - 1, z)] != SUPPORT:
                    mism += 1
                    fails.append("ch%d (%d,%d) 下方无支撑方块" % (ch, x, z))
            elif c["t"] == "barrel":
                nbar += 1
                if st != "minecraft:barrel[facing=up,open=false]":
                    mism += 1
                    if mism <= 6:
                        fails.append("ch%d (%d,%d) 容器方块状态 %s" % (ch, x, z, st))
                dat = ent.get(("minecraft:barrel", (x, y, z)))
                if dat is None:
                    mism += 1
                    fails.append("ch%d (%d,%d) 缺容器 BlockEntity" % (ch, x, z))
                else:
                    sig, fill = sig_of_items(dat.get("Items", []))
                    sig_max = max(sig_max, sig)
                    if sig > SIG_MAX:
                        unreach += 1
                    if sig != int(c["v"] or 0):
                        mism += 1
                        if mism <= 9:
                            fails.append("ch%d (%d,%d) 容器信号 %d（fill=%d）≠ 仿真 %d"
                                         % (ch, x, z, sig, fill, c["v"]))
            else:
                nblk += 1
                cc = c["c"] or ""
                if cc.startswith("col"):
                    want = "minecraft:%s_concrete" % COLORS[int(cc[3:]) % 8]
                elif cc == "white":
                    want = "minecraft:white_concrete"
                elif cc == "wire":
                    want = "minecraft:pink_concrete"
                else:
                    want = "minecraft:light_gray_concrete"
                if st != want:
                    mism += 1
                    if mism <= 6:
                        fails.append("ch%d (%d,%d) 块 %s ≠ %s" % (ch, x, z, st, want))
    print("3 逐格对账：%d 个 cell（比较器 %d / 容器 %d / 其它 %d），不一致 %d 处  %s"
          % (tot_cells, ncmp, nbar, nblk, mism, "✓" if mism == 0 else "✗"))

    # ---- 7 不可达容器 ----
    chk(unreach == 0, "不可达容器 %d 个" % unreach)
    print("4 容器可达性：最大信号 %d ≤ %d，不可达 %d 个  %s"
          % (sig_max, SIG_MAX, unreach, "✓" if unreach == 0 else "✗"))

    # ---- 8 powered vs OutputSignal ----
    from collections import Counter as C2
    pc = C2()
    for i in idx:
        s = inv[i]
        if "comparator" in s:
            pc["true" if "powered=true" in s else "false"] += 1
    nz = sum(1 for (k, pos), v in ent.items()
             if k == "minecraft:comparator" and int(v.get("OutputSignal", 0)) > 0)
    ok8 = pc["true"] == nz
    chk(ok8, "powered=true %d ≠ OutputSignal>0 %d" % (pc["true"], nz))
    print("5 powered 一致性：true %d / false %d，OutputSignal>0 %d  %s"
          % (pc["true"], pc["false"], nz, "✓" if ok8 else "✗"))

    # ---- 9 火把 ----
    torch = sum(1 for i in idx if "torch" in inv[i])
    print("6 火把：%d 个（本阶段不放火把，输入恒为 X=0；等火把塔）" % torch)

    print()
    if fails:
        print("✗ 失败 %d 项，前 12 条：" % len(fails))
        for f in fails[:12]:
            print("   " + f)
        return 1
    print("回读对账全部通过 ✓  （尺寸 %dx%dx%d，BlockEntities %d）" % (W, H, L, len(be)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
