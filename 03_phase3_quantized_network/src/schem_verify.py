# -*- coding: utf-8 -*-
"""校验生成的 .schem：NBT 完整回读 + 比较器 powered 与 OutputSignal 是否一致。"""
import gzip
import struct
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class R:
    def __init__(s, b):
        s.b, s.i = b, 0

    def u1(s):
        v = s.b[s.i]; s.i += 1; return v

    def u2(s):
        v = struct.unpack(">H", s.b[s.i:s.i + 2])[0]; s.i += 2; return v

    def u4(s):
        v = struct.unpack(">I", s.b[s.i:s.i + 4])[0]; s.i += 4; return v

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
            n = s.u4(); v = s.b[s.i:s.i + n]; s.i += n; return v
        if t == 8:
            n = s.u2(); v = s.b[s.i:s.i + n].decode("utf-8", "replace"); s.i += n; return v
        if t == 9:
            et = s.u1(); n = s.u4()
            return [s.payload(et) for _ in range(n)]
        if t == 10:
            d = {}
            while True:
                tt = s.u1()
                if tt == 0:
                    return d
                nm = s.name()
                d[nm] = s.payload(tt)
        if t == 11:
            n = s.u4()
            v = [struct.unpack(">i", s.b[s.i + 4 * k:s.i + 4 * k + 4])[0] for k in range(n)]
            s.i += 4 * n; return v
        if t == 12:
            n = s.u4()
            v = [struct.unpack(">q", s.b[s.i + 8 * k:s.i + 8 * k + 8])[0] for k in range(n)]
            s.i += 8 * n; return v
        raise ValueError(f"tag {t} @ {s.i}")


def expand(bd):
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
    return raw, r, sch


base = r"E:\redstone_nn\03_phase3_quantized_network\results\schem"
for nm in ("C0_all0", "C1_all1", "C2_1010", "C3_0101"):
    path = rf"{base}\layer1_ch0_{nm}.schem"
    raw, r, sch = load(path)
    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    blk = sch["Blocks"]
    pal = blk["Palette"]
    inv = {v: k for k, v in pal.items()}
    idx = expand(blk["Data"])
    be = blk["BlockEntities"]

    # 比较器方块状态的 powered 分布
    pcount = Counter()
    for i in idx:
        st = inv[i]
        if "comparator" in st:
            pcount[("powered=true" in st)] += 1

    # BlockEntity 里的 OutputSignal
    sig = {}
    for e in be:
        if e.get("Id") == "minecraft:comparator":
            p = tuple(e["Pos"])
            sig[p] = e.get("Data", {}).get("OutputSignal", 0)

    nz = sum(1 for v in sig.values() if v > 0)
    complete = (r.i == len(raw)) and (len(idx) == W * H * L)
    print(f"{nm}: {W}x{H}x{L} 格数{len(idx)} 读到 {r.i}/{len(raw)} "
          f"{'✓' if complete else '✗'}  BE={len(be)}")
    print(f"   比较器 {sum(pcount.values())} 个: powered=true {pcount[True]}, powered=false {pcount[False]}")
    print(f"   OutputSignal>0 的 {nz} 个，最大 {max(sig.values()) if sig else 0}")
    # 对照：powered=true 的方块数与 OutputSignal>0 的数应一致
    flag = "✓" if pcount[True] == nz else "✗ 不一致"
    print(f"   powered=true({pcount[True]}) vs OutputSignal>0({nz})  {flag}")
