# -*- coding: utf-8 -*-
"""学习一期工程 hand_write_digital.schem（WorldEdit 的 Sponge v3）。"""
import gzip
import json
import struct
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

P = r"E:\redstone_nn\03_phase3_quantized_network\data\hand_write_digital.schem"


class R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def u1(self):
        v = self.b[self.i]; self.i += 1; return v

    def s(self):
        n = struct.unpack(">H", self.b[self.i:self.i + 2])[0]; self.i += 2
        v = self.b[self.i:self.i + n].decode("utf-8", "replace"); self.i += n; return v

    def payload(self, t):
        if t == 1: return self.u1()
        if t == 2:
            v = struct.unpack(">h", self.b[self.i:self.i + 2])[0]; self.i += 2; return v
        if t == 3:
            v = struct.unpack(">i", self.b[self.i:self.i + 4])[0]; self.i += 4; return v
        if t == 4:
            v = struct.unpack(">q", self.b[self.i:self.i + 8])[0]; self.i += 8; return v
        if t == 5:
            v = struct.unpack(">f", self.b[self.i:self.i + 4])[0]; self.i += 4; return v
        if t == 6:
            v = struct.unpack(">d", self.b[self.i:self.i + 8])[0]; self.i += 8; return v
        if t == 7:
            n = self.i4() if False else struct.unpack(">i", self.b[self.i:self.i + 4])[0]
            self.i += 4
            v = self.b[self.i:self.i + n]; self.i += n; return v
        if t == 8: return self.s()
        if t == 9:
            et = self.u1(); n = struct.unpack(">i", self.b[self.i:self.i + 4])[0]; self.i += 4
            return [self.payload(et) for _ in range(n)]
        if t == 10:
            d = {}
            while True:
                tt = self.u1()
                if tt == 0: return d
                nm = self.s()
                d[nm] = self.payload(tt)
        if t == 11:
            n = struct.unpack(">i", self.b[self.i:self.i + 4])[0]; self.i += 4
            return [struct.unpack(">i", self.b[self.i + 4 * k:self.i + 4 * k + 4])[0]
                    for k in range(n)]
        if t == 12:
            n = struct.unpack(">i", self.b[self.i:self.i + 4])[0]; self.i += 4
            return [struct.unpack(">q", self.b[self.i + 8 * k:self.i + 8 * k + 8])[0]
                    for k in range(n)]
        raise ValueError(f"tag {t}")


def expand_varints(bd):
    out = []
    val, shift = 0, 0
    for b in bd:
        val |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            out.append(val); val, shift = 0, 0
    return out


def main():
    raw = gzip.decompress(open(P, "rb").read())
    r = R(raw)
    assert r.u1() == 10
    _ = r.s()
    root = r.payload(10)["Schematic"]
    W, H, L = root["Width"], root["Height"], root["Length"]
    print(f"尺寸 {W}×{H}×{L} = {W*H*L}   DataVersion={root['DataVersion']}")

    blk = root["Blocks"]
    print("Blocks 子键:", list(blk.keys()))
    pal = blk["Palette"]
    inv = {v: k for k, v in pal.items()}
    data = blk["Data"]
    idx = expand_varints(data)
    print(f"调色板 {len(pal)} 项，Data {len(data)} 字节 → {len(idx)} 个索引"
          f"（应 = {W*H*L}）")
    print("\n调色板：")
    for st, i in sorted(pal.items(), key=lambda kv: kv[1]):
        print(f"  {i:3d}  {st}")

    cnt = Counter(inv[i] for i in idx)
    print("\n各方块数量：")
    for st, n in cnt.most_common():
        print(f"  {n:6d}  {st}")

    be = blk.get("BlockEntities", [])
    print(f"\nBlockEntities 数量 = {len(be)}")
    if be:
        print("  类型分布:", dict(Counter(e.get("Id", "?") for e in be)))
        print("\n  前 2 个完整样本：")
        for e in be[:2]:
            print(json.dumps(e, ensure_ascii=False, indent=4)[:2500])

    # 找比较器/中继器的位置，看有没有 signal 之类
    print("\n比较器/中继器样本（前 6）：")
    shown = 0
    for i, idv in enumerate(idx):
        st = inv[idv]
        if "comparator" in st or "repeater" in st:
            y = i // (L * W)
            z = (i // W) % L
            x = i % W
            print(f"  ({x},{y},{z})  {st}")
            shown += 1
            if shown >= 6:
                break


if __name__ == "__main__":
    main()
